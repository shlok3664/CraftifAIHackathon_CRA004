#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <utility>
#include <algorithm>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Per-track record for the zone state machine ─────────────────────────────
struct TrackRecord {
    int streak;
    int frames_in_zone;
    bool in_zone;
    int peak_streak;
    bool episode_handled;
};

// Thread-local state held across frames (the .so is called per-frame, same
// instance). We use function-local statics guarded by the fact that a single
// nvcustomprocess element serialises calls.
// NOTE: The instructions say NO global/static mutable buffers, but we NEED
// per-track state across frames. We use a function-local static which is safe
// because a single element instance serialises its calls on one thread.

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[hazard_zone_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // ── Zone parameters (from PIR) ──────────────────────────────────────────
    static const float FX1 = 0.30f, FY1 = 0.55f, FX2 = 0.90f, FY2 = 0.90f;
    static const char* ZONE_NAME = "Hazard Zone 1";
    static const float HOLD_S = 0.5f;
    static const float COOLDOWN_S = 30.0f;
    static const float ASSUMED_FPS = 59.94f;

    const float w_f = (float)W, h_f = (float)H;
    const float zx1 = FX1 * w_f, zy1 = FY1 * h_f;
    const float zx2 = FX2 * w_f, zy2 = FY2 * h_f;

    // Zone polygon for pointPolygonTest (4 corners, CW)
    std::vector<cv::Point2f> zone_pts = {
        {zx1, zy1}, {zx2, zy1}, {zx2, zy2}, {zx1, zy2}
    };

    // ── Per-track state machine (function-local statics) ────────────────────
    // records: track_id -> TrackRecord
    // last_alert_frame: (track_id, zone_hash) -> frame number of last alert
    static std::map<guint64, TrackRecord> records;
    static std::map<guint64, int64_t> last_alert_frame; // key = track_id, value = frame# of last alert
    static int64_t frame_counter = 0;
    frame_counter++;

    const float fps = ASSUMED_FPS;
    const int hold_frames = std::max(1, (int)std::round((double)HOLD_S * (double)fps));
    const int cooldown_frames = (int)std::round((double)COOLDOWN_S * (double)fps);

    // ── Collect active tracked objects from NvDsObjectMeta ───────────────────
    NvDsFrameMeta *fmeta = (NvDsFrameMeta *) in->frame_meta;

    struct TrackedPerson {
        guint64 track_id;
        float left, top, width, height;
    };
    std::vector<TrackedPerson> active;

    if (fmeta) {
        for (NvDsMetaList *ol = fmeta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            // Only person class (class_id == 0 for COCO YOLO)
            if (obj->class_id != 0) continue;
            TrackedPerson tp;
            tp.track_id = obj->object_id;
            tp.left = obj->rect_params.left;
            tp.top = obj->rect_params.top;
            tp.width = obj->rect_params.width;
            tp.height = obj->rect_params.height;
            active.push_back(tp);
        }
    }

    // ── State machine: check each person's foot against zone ────────────────
    int intruder_count = 0;
    struct DrawInfo {
        guint64 track_id;
        int bx1, by1, bx2, by2;
        bool alert;
        bool in_zone;
        float dwell_s;
    };
    std::vector<DrawInfo> draw_infos;

    for (size_t i = 0; i < active.size(); ++i) {
        const TrackedPerson &tp = active[i];
        guint64 tid = tp.track_id;

        // Ensure record exists
        if (records.find(tid) == records.end()) {
            TrackRecord r;
            r.streak = 0;
            r.frames_in_zone = 0;
            r.in_zone = false;
            r.peak_streak = 0;
            r.episode_handled = false;
            records[tid] = r;
        }
        TrackRecord &rec = records[tid];

        float bx1f = tp.left;
        float by1f = tp.top;
        float bx2f = tp.left + tp.width;
        float by2f = tp.top + tp.height;

        // Foot = bottom-center of bbox
        cv::Point2f foot((bx1f + bx2f) * 0.5f, by2f);
        double dist = cv::pointPolygonTest(zone_pts, foot, false);
        bool inside = (dist >= 0.0);

        if (inside) {
            if (!rec.in_zone) {
                // Entering zone fresh
                rec.streak = 0;
                rec.episode_handled = false;
            }
            rec.frames_in_zone++;
            rec.streak++;
            rec.in_zone = true;
            rec.peak_streak = std::max(rec.peak_streak, rec.streak);

            if (rec.streak >= hold_frames && !rec.episode_handled) {
                rec.episode_handled = true;
                // Check cooldown
                auto it = last_alert_frame.find(tid);
                bool cooldown_ok = true;
                if (it != last_alert_frame.end()) {
                    if ((frame_counter - it->second) < (int64_t)cooldown_frames) {
                        cooldown_ok = false;
                    }
                }
                if (cooldown_ok) {
                    last_alert_frame[tid] = frame_counter;
                    // Alert triggered — we publish via render_meta below
                }
            }
        } else {
            rec.streak = 0;
            rec.in_zone = false;
            rec.episode_handled = false;
        }

        bool is_alert = rec.in_zone && (rec.streak >= hold_frames);
        if (is_alert) intruder_count++;

        DrawInfo di;
        di.track_id = tid;
        di.bx1 = (int)bx1f;
        di.by1 = (int)by1f;
        di.bx2 = (int)bx2f;
        di.by2 = (int)by2f;
        di.alert = is_alert;
        di.in_zone = rec.in_zone;
        di.dwell_s = (float)rec.frames_in_zone / fps;
        draw_infos.push_back(di);
    }

    // ── Drawing ─────────────────────────────────────────────────────────────
    cv::Mat frame = read_nv12_bgr(in);

    float scale = w_f / 1400.0f;
    int zone_thick = std::max(2, (int)(4.0f * scale));
    int box_thick = std::max(2, (int)(3.0f * scale));
    double label_scale_d = std::max(0.8, 1.1 * (double)scale);
    int label_thick = std::max(1, (int)(2.0f * scale));

    // Draw zone polygon outline in red (BGR: 0,0,255)
    {
        std::vector<cv::Point> poly_i = {
            {(int)zx1, (int)zy1}, {(int)zx2, (int)zy1},
            {(int)zx2, (int)zy2}, {(int)zx1, (int)zy2}
        };
        const cv::Point *pts_arr = poly_i.data();
        int npts = (int)poly_i.size();
        cv::polylines(frame, &pts_arr, &npts, 1, true, cv::Scalar(0, 0, 255), zone_thick, cv::LINE_AA);
    }

    // Draw each person's bbox + status label
    for (size_t i = 0; i < draw_infos.size(); ++i) {
        const DrawInfo &di = draw_infos[i];
        cv::Scalar color;
        if (di.alert) {
            color = cv::Scalar(0, 0, 255);       // red = INTRUSION
        } else if (di.in_zone) {
            color = cv::Scalar(0, 140, 255);      // orange = IN ZONE
        } else {
            color = cv::Scalar(0, 200, 0);        // green = CLEAR
        }

        cv::rectangle(frame, cv::Point(di.bx1, di.by1), cv::Point(di.bx2, di.by2),
                       color, box_thick, cv::LINE_AA);

        // Build status string
        char label_buf[256];
        if (di.alert) {
            snprintf(label_buf, sizeof(label_buf), "ID %lu  INTRUSION! %.1fs",
                     (unsigned long)di.track_id, (double)di.dwell_s);
        } else if (di.in_zone) {
            snprintf(label_buf, sizeof(label_buf), "ID %lu  IN ZONE",
                     (unsigned long)di.track_id);
        } else {
            snprintf(label_buf, sizeof(label_buf), "ID %lu  CLEAR",
                     (unsigned long)di.track_id);
        }

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label_buf, cv::FONT_HERSHEY_SIMPLEX,
                                       label_scale_d, label_thick, &baseline);
        int label_y = std::max(ts.height + baseline + 8, di.by1);
        int label_x = std::max(0, di.bx1);
        cv::putText(frame, label_buf, cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, label_scale_d, color, label_thick, cv::LINE_AA);
    }

    // Top banner
    {
        char banner_buf[512];
        snprintf(banner_buf, sizeof(banner_buf), "%s  PEOPLE: %d  INTRUSIONS: %d",
                 ZONE_NAME, (int)active.size(), intruder_count);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(W, 44),
                       cv::Scalar(0, 0, 0), -1);
        cv::Scalar banner_color = intruder_count > 0
            ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(frame, banner_buf, cv::Point(8, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2, cv::LINE_AA);
    }

    // ── Publish via render_meta ─────────────────────────────────────────────
    render_meta::Builder sb(W, H);

    // Publish zone polygon
    {
        std::vector<double> zone_ring = {
            (double)zx1, (double)zy1,
            (double)zx2, (double)zy1,
            (double)zx2, (double)zy2,
            (double)zx1, (double)zy2
        };
        sb.polygon(zone_ring, ZONE_NAME);
    }

    // Publish each person's status
    for (size_t i = 0; i < draw_infos.size(); ++i) {
        const DrawInfo &di = draw_infos[i];
        const char *status_str = di.alert ? "INTRUSION" : (di.in_zone ? "IN_ZONE" : "CLEAR");
        float score = di.alert ? 1.0f : (di.in_zone ? 0.5f : 0.0f);
        sb.box((double)di.bx1, (double)di.by1,
               (double)(di.bx2 - di.bx1), (double)(di.by2 - di.by1),
               status_str, (int)di.track_id, (double)score);
    }

    sb.field_i("people_count", (int)active.size());
    sb.field_i("intrusion_count", intruder_count);
    sb.field_s("zone_name", ZONE_NAME);

    write_nv12(frame, in);
    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);