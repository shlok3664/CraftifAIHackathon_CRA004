// mqtt_publish.cpp -- publish detections over MQTT, for the `nvcustomprocess`
// plugin.
//
// This library does one thing. It is not a file writer with a network option
// bolted on: everything here -- rate limiting, topics, status reporting -- exists
// because the destination is a message bus with something listening at the far
// end. (For a .json on disk, see ../../video_json_export.)
//
// The payload is frame_json.h, the same shape that pipeline writes to file, so a
// subscriber and a file reader agree about what a detection looks like.
//
// The element sits AFTER nvtracker, which is what makes `id` a persistent object
// id rather than a per-frame index, and it is a PASS-THROUGH rather than a sink,
// so the video can carry on to a display or an encoder with no tee. It reads
// metadata only and never touches a pixel.
//
// libmosquitto, not Paho: mosquitto ships installed in the DeepStream image
// (/usr/local, with pkg-config and headers) and is the same library DeepStream's
// own libnvds_mqtt_proto.so uses, so there is nothing to install. Paho C++ also
// works for publishing but cannot be shut down from a dlopen'd plugin's static
// destructor -- it aborts with heap corruption, or segfaults and then hangs in
// GStreamer's fault handler.
//
// CONFIGURATION (env vars -- the plugin has no config-file property)
//   MQTT_HOST        127.0.0.1
//   MQTT_PORT        1883
//   MQTT_TOPIC       ds/detections
//   MQTT_QOS         0        0, 1 or 2
//   MQTT_CLIENT_ID   ds-mqtt-pipeline   must be unique per broker connection
//   MQTT_USER        unset    username, if the broker requires one
//   MQTT_PASSWORD    unset
//   MQTT_KEEPALIVE   60       seconds
//   MQTT_INTERVAL    1        publish every Nth frame
//   MQTT_SKIP_EMPTY  0        1 = do not publish frames with no objects
//   MQTT_PAYLOAD     frame    frame  = one message per frame
//                             object = one message per object, to <topic>/<label>
//   MQTT_MIN_CONF    0.0      drop objects below this detector confidence.
//                             An object whose payload is NOT its bbox -- a
//                             classifier result, an instance mask -- has no
//                             detector confidence and is kept regardless.
//   MQTT_STATUS      0.5      seconds between status lines (0 = silent)
//
// PER-TASK TOPICS (both unset = one topic, exactly as this file always behaved)
//   MQTT_TOPIC_TEMPLATE  a topic carrying a literal `{task}` hole, e.g.
//                        `pipegen/{task}/cam0/yolov8n`
//   MQTT_TOPIC_TASKS     `stream:task` pairs, e.g. `objects:detection,shapes:pose`
//
//   A multi-task pipeline -- a detector AND a pose model on one chain -- used to
//   publish everything to ONE topic named after whichever task the PIR listed
//   first, so a subscriber reading `pipegen/detection/...` was also receiving
//   skeletons with no way to know from the topic that it was. Nothing was lost;
//   the NAME under-described the traffic.
//
//   With these set, each due frame is published once per task whose part of the
//   payload is actually present on that frame: boxes go to the objects task's
//   topic, published shapes go to the shapes task's topic. The library cannot
//   name those tasks itself -- it can see that shapes were published, not that
//   the author called them "pose" -- so the names come from the PIR through
//   MQTT_TOPIC_TASKS. `stream` is one of objects / shapes / segmentation.
//
//   A ROUTED MESSAGE CARRIES ONLY ITS OWN TASK'S PART OF THE PAYLOAD, plus the
//   frame header every message has. That is what makes the topics worth having:
//   selectivity is the whole point of a topic, and a subscriber that wants
//   everything takes the wildcard and joins on (source_id, frame), which is
//   cheaper than receiving the same bytes once per task. Publishing the whole
//   record to every topic is a FILE writer's instinct -- one artifact, complete
//   in itself -- and this is a bus.
//
//   Every routed message therefore states its `task`. Without it, "no objects
//   key" would mean BOTH "this frame detected nothing" and "this topic does not
//   carry objects", and the usual defensive read -- data.get("objects", []) --
//   cannot tell those apart: it returns an empty list for both and the consumer
//   silently sees nothing. With `task` stated, absence is never inferred.
//
//   The UNROUTED path is untouched: one topic, the whole frame_json record, no
//   `task` key. That is what every single-task pipeline still emits.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <glib.h>
#include <mosquitto.h>

#include "nvcustomprocess.h"
#include "nvdsmeta.h"
#include "frame_json.h"

namespace {

struct mosquitto* g_mosq = nullptr;
bool   g_init_done = false, g_up = false;
std::string g_topic = "ds/detections";
int    g_qos = 0, g_interval = 1;
bool   g_skip_empty = false, g_per_object = false;
double g_min_conf = 0.0, g_status_every = 1.0;

uint64_t g_frames = 0, g_objects = 0, g_published = 0, g_failed = 0;
double   g_t0 = 0.0, g_t_status = 0.0;
uint64_t g_pub_at_status = 0;

// -- per-task topics ---------------------------------------------------------
//
// A route is "when this part of the payload is present on a frame, publish that
// frame to this topic". Empty means the pipeline declared one task (or the user
// pinned a topic), and every path below then reduces to publishing g_topic --
// literally the same call this file has always made.
enum { STREAM_OBJECTS = 0, STREAM_SHAPES = 1, STREAM_SEG = 2 };

// `task` is carried as well as substituted into `topic`, because it is stated
// IN the message too -- see build_split and the PER-TASK TOPICS note above.
struct Route { int stream; std::string topic, task; };
std::vector<Route> g_routes;
// Whether any route needs the corresponding presence test. Both tests walk a
// metadata list, so a pipeline with no shapes route must not pay for the shapes
// walk on every frame of a long run.
bool g_want_shapes = false, g_want_seg = false;

int    env_i(const char* k, int v) { const char* s = getenv(k); return (s && *s) ? atoi(s) : v; }
double env_d(const char* k, double v) { const char* s = getenv(k); return (s && *s) ? atof(s) : v; }

// MQTT_TOPIC_TEMPLATE + MQTT_TOPIC_TASKS -> g_routes. Anything malformed is
// SKIPPED rather than guessed at: a route built from a half-parsed spec would
// publish to a topic nobody subscribed to, which is indistinguishable from the
// pipeline not running. Dropping it instead falls back to g_topic, where the
// data still arrives.
void build_routes() {
    const char* tmpl = getenv("MQTT_TOPIC_TEMPLATE");
    const char* spec = getenv("MQTT_TOPIC_TASKS");
    if (!tmpl || !*tmpl || !spec || !*spec) return;

    const std::string t(tmpl);
    const std::string hole = "{task}";
    const size_t at = t.find(hole);
    if (at == std::string::npos) {
        fprintf(stderr, "[mqtt] MQTT_TOPIC_TEMPLATE has no {task} -- "
                        "per-task topics disabled\n");
        return;
    }

    const std::string s(spec);
    // Bounded: three streams exist, so more than a handful of pairs means the
    // variable is not what this code thinks it is.
    for (size_t i = 0; i < s.size() && g_routes.size() < 8;) {
        size_t comma = s.find(',', i);
        if (comma == std::string::npos) comma = s.size();
        const std::string one = s.substr(i, comma - i);
        i = comma + 1;

        const size_t colon = one.find(':');
        if (colon == std::string::npos) continue;
        const std::string kind = one.substr(0, colon);
        const std::string name = one.substr(colon + 1);
        if (name.empty()) continue;

        Route r;
        if      (kind == "objects")      r.stream = STREAM_OBJECTS;
        else if (kind == "shapes")       r.stream = STREAM_SHAPES;
        else if (kind == "segmentation") r.stream = STREAM_SEG;
        else continue;
        r.topic = t.substr(0, at) + name + t.substr(at + hole.size());
        r.task = name;
        g_routes.push_back(r);

        if (r.stream == STREAM_SHAPES) g_want_shapes = true;
        if (r.stream == STREAM_SEG)    g_want_seg = true;
    }
}

// One task's message: the frame header every message carries, the task that
// produced it, and THAT TASK'S PART OF THE PAYLOAD ONLY.
//
// Assembled here from frame_json.h's public writers rather than by teaching
// frame_json::frame() which keys to emit. That is deliberate and it is the
// important line in this file: `frame_json::frame` is what the JSON file writer
// calls, and its output is diffed byte for byte against a production log. A
// selectivity parameter on it would put this bus-only concern on the path of a
// file consumer that never asked for it. The pieces below -- object(),
// shapes(), segmentation() -- are the same code frame() itself calls, so the
// bytes of each key are identical either way; only which keys appear differs.
//
// The header is frame()'s own, minus `objects`, so a routed message and an
// unrouted one agree field for field on everything they share.
void build_split(std::string& out, NvDsFrameMeta* fm, int W, int H,
                 const std::vector<NvDsObjectMeta*>& objs,
                 const Route& route) {
    frame_json::appendf(out, "{\"frame\": %u, \"source_id\": %u, \"pts_ns\": %"
                             PRIu64 ", \"width\": %d, \"height\": %d, \"task\": ",
                        fm->frame_num, fm->source_id, (uint64_t) fm->buf_pts, W, H);
    // escape_n rather than escape, by this file's own rule for a NEW key: the
    // task reaches us through an environment variable, so its bytes are not
    // this compiler's to vouch for even though the value it WRITES is a slug of
    // alphanumerics and dashes. A hand-edited MQTT_TOPIC_TASKS carrying Latin-1
    // would otherwise put raw 0x80..0xFF on the wire and cost a subscriber the
    // whole line rather than one field. See the note above escape_n itself.
    frame_json::escape_n(out, route.task.c_str(), route.task.size());
    if (route.stream == STREAM_OBJECTS) {
        out += ", \"objects\": [";
        for (size_t i = 0; i < objs.size(); ++i) {
            if (i) out += ", ";
            frame_json::object(out, objs[i]);
        }
        out += ']';
    } else {
        // THE KEY IS ALWAYS EMITTED, empty when the frame carries nothing.
        //
        // shapes() and segmentation() append their own ", <key>: [...]" and
        // append NOTHING AT ALL when the metadata is absent -- which is right
        // for frame_json::frame(), whose contract is "omit an absent key", and
        // wrong here. On a routed topic the key is not absent-because-missing,
        // it is THE SUBJECT OF THE TOPIC, and a message that silently drops it
        // gives that topic two message shapes: `{... "task": "segmentation",
        // "segmentation": [...]}` on a frame with data and `{... "task":
        // "segmentation"}` on one without.
        //
        // Reachable, not theoretical: fallback_route() returns the first route
        // when a pipeline has no objects route at all (segmentation + depth,
        // say), so a due-but-empty frame lands here with nothing to write. An
        // earlier revision of this function asserted in a comment that it could
        // not -- it produced exactly that header-only message.
        const size_t before = out.size();
        if (route.stream == STREAM_SHAPES) frame_json::shapes(out, fm, W, H);
        else                               frame_json::segmentation(out, fm);
        if (out.size() == before)
            out += (route.stream == STREAM_SHAPES) ? ", \"shapes\": []"
                                                   : ", \"segmentation\": []";
    }
    // Frame analytics is shared context on every existing task topic. Readers
    // identify counters by source_id/frame/unique_id; no new topic is inferred.
    frame_json::analytics(out, fm);
    out += '}';
}

// The objects route, or the first route, or none at all. Used for the two
// decisions that need "the route a message belongs to when no stream test
// picked one": per-object messages, and a due frame that carried nothing.
// Objects is preferred because its writer emits `"objects": []` for an empty
// frame, which is a well-formed "nothing here" -- the other two streams emit no
// key at all and would leave a message with only a header.
const Route* fallback_route() {
    for (size_t i = 0; i < g_routes.size(); ++i)
        if (g_routes[i].stream == STREAM_OBJECTS) return &g_routes[i];
    return g_routes.empty() ? NULL : &g_routes[0];
}

// Per-object messages hang off the OBJECTS route when there is one, so
// `<topic>/car` stays under the same task level as the frame records that
// accompany it. Returns a reference into a namespace-scope global either way.
const std::string& object_base() {
    const Route* r = fallback_route();
    return r ? r->topic : g_topic;
}

// Did a render stage publish anything on this frame? render_meta.h reaches this
// file through frame_json.h's `__has_include`, so a parsers/ directory without
// it compiles to a constant false and a shapes route simply never fires -- the
// same soft failure frame_json's own shapes[] key makes.
bool has_shapes(NvDsFrameMeta* fm) {
#ifdef FRAME_JSON_WITH_SHAPES
    return render_meta::present(fm);
#else
    (void) fm;
    return false;
#endif
}

// Is a semantic-segmentation meta attached? The enum lives in nvdsmeta.h, so
// this needs no include gstnvdsinfer.h -- see the note at the top of
// frame_json.h for why that header cannot be pulled into a glib-only build.
// The walk is capped for the same reason every list walk in frame_json.h is: a
// corrupted or cyclic list must cost a wrong answer, not the streaming thread.
bool has_segmentation(NvDsFrameMeta* fm) {
    if (!fm) return false;
    size_t walked = 0;
    for (NvDsMetaList* l = fm->frame_user_meta_list; l && walked < 4096; l = l->next) {
        ++walked;
        const NvDsUserMeta* um = (const NvDsUserMeta*) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) NVDSINFER_SEGMENTATION_META)
            return true;
    }
    return false;
}

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

void on_log(struct mosquitto*, void*, int level, const char* str) {
    // MOSQ_LOG_DEBUG at video rates is a flood; only what a human needs.
    if (level & (MOSQ_LOG_ERR | MOSQ_LOG_WARNING))
        fprintf(stderr, "[mqtt] %s\n", str);
}

void on_disconnect(struct mosquitto*, void*, int rc) {
    if (rc != 0)
        fprintf(stderr, "[mqtt] connection lost (rc=%d) -- the client thread "
                        "retries with backoff; publishes fail meanwhile\n", rc);
}

void on_connect(struct mosquitto*, void*, int rc) {
    if (rc == 0) fprintf(stderr, "[mqtt] connected\n");
}

// Status on stderr, not stdout: stdout is where a caller may want the payloads.
void status(bool final_line) {
    const double t = now_s();
    const double elapsed = t - g_t0;
    const double since = t - g_t_status;
    const double rate = since > 0 ? (double) (g_published - g_pub_at_status) / since : 0.0;
    fprintf(stderr,
            "[mqtt] %s %6" PRIu64 " frames  %7" PRIu64 " objects  %6" PRIu64
            " published  %5.1f msg/s  %5.1fs%s\n",
            final_line ? "done " : "     ",
            g_frames, g_objects, g_published, rate, elapsed,
            g_failed ? "  FAILED: " : "");
    if (g_failed)
        fprintf(stderr, "[mqtt]       %" PRIu64 " publishes failed\n", g_failed);
    g_t_status = t;
    g_pub_at_status = g_published;
}

void shutdown_mqtt() {
    if (g_frames) status(true);
    if (!g_mosq) return;
    // Let queued messages reach the socket before tearing the client down,
    // otherwise the tail of a run is silently dropped.
    mosquitto_disconnect(g_mosq);
    mosquitto_loop_stop(g_mosq, false);
    mosquitto_destroy(g_mosq);
    mosquitto_lib_cleanup();
    g_mosq = nullptr;
}

// Runs at process exit. `gst-launch-1.0 -e` turns SIGINT into EOS and exits
// normally, so this covers ctrl-c too.
struct Closer { ~Closer() { shutdown_mqtt(); } };
Closer g_closer;

void lazy_init() {
    if (g_init_done) return;
    g_init_done = true;

    const char* host = getenv("MQTT_HOST");
    if (!host || !*host) host = "127.0.0.1";
    const int port = env_i("MQTT_PORT", 1883);
    const char* topic = getenv("MQTT_TOPIC");
    if (topic && *topic) g_topic = topic;
    g_qos = env_i("MQTT_QOS", 0);
    g_interval = env_i("MQTT_INTERVAL", 1);
    if (g_interval < 1) g_interval = 1;
    g_skip_empty = env_i("MQTT_SKIP_EMPTY", 0) != 0;
    const char* mode = getenv("MQTT_PAYLOAD");
    g_per_object = (mode && strcmp(mode, "object") == 0);
    g_min_conf = env_d("MQTT_MIN_CONF", 0.0);
    g_status_every = env_d("MQTT_STATUS", 0.5);
    build_routes();

    mosquitto_lib_init();
    const char* cid = getenv("MQTT_CLIENT_ID");
    g_mosq = mosquitto_new((cid && *cid) ? cid : "ds-mqtt-pipeline", true, nullptr);
    if (!g_mosq) {
        fprintf(stderr, "[mqtt] mosquitto_new failed -- not publishing\n");
        return;
    }
    mosquitto_log_callback_set(g_mosq, on_log);
    mosquitto_connect_callback_set(g_mosq, on_connect);
    mosquitto_disconnect_callback_set(g_mosq, on_disconnect);
    mosquitto_reconnect_delay_set(g_mosq, 1, 30, true);   // backoff, not a hammer

    const char* user = getenv("MQTT_USER");
    const char* pass = getenv("MQTT_PASSWORD");
    if (user && *user)
        mosquitto_username_pw_set(g_mosq, user, (pass && *pass) ? pass : nullptr);

    const int rc = mosquitto_connect(g_mosq, host, port, env_i("MQTT_KEEPALIVE", 60));
    if (rc != MOSQ_ERR_SUCCESS) {
        // /usr/local/sbin/mosquitto is substituted from MOSQUITTO_PATH when this template is
        // loaded, so this message, the compile log and run.sh's help block
        // cannot name different binaries. See that constant for why.
        fprintf(stderr, "[mqtt] cannot reach a broker at %s:%d -- %s\n"
                        "[mqtt] start the one shipped with DeepStream:  "
                        "/usr/local/sbin/mosquitto -p %d &\n"
                        "[mqtt] the video keeps running; nothing is published\n",
                host, port, mosquitto_strerror(rc), port);
        mosquitto_destroy(g_mosq);
        g_mosq = nullptr;
        return;
    }
    // loop_start runs the network traffic on its own thread. Without it nothing
    // reaches the socket -- mosquitto_publish only ENQUEUES -- so the run would
    // report successful publishes while a subscriber saw none.
    if (mosquitto_loop_start(g_mosq) != MOSQ_ERR_SUCCESS) {
        fprintf(stderr, "[mqtt] loop_start failed -- not publishing\n");
        mosquitto_destroy(g_mosq);
        g_mosq = nullptr;
        return;
    }
    g_up = true;
    g_t0 = g_t_status = now_s();
    fprintf(stderr, "[mqtt] publishing to %s:%d topic '%s' qos %d, %s payloads"
                    "%s\n", host, port, g_topic.c_str(), g_qos,
            g_per_object ? "per-object" : "per-frame",
            g_interval > 1 ? "" : "");
    if (g_interval > 1)
        fprintf(stderr, "[mqtt] every %dth frame\n", g_interval);
    // Printed, because a topic a subscriber cannot guess is a topic nobody
    // connects to. The whole point of per-task topics is that they are
    // discoverable, and the run's own log is the first place anyone looks.
    for (size_t i = 0; i < g_routes.size(); ++i)
        fprintf(stderr, "[mqtt] task topic '%s' (when the frame carries %s)\n",
                g_routes[i].topic.c_str(),
                g_routes[i].stream == STREAM_OBJECTS ? "objects" :
                g_routes[i].stream == STREAM_SHAPES  ? "published shapes"
                                                     : "a segmentation map");
}

void publish(const std::string& topic, const std::string& payload) {
    const int rc = mosquitto_publish(g_mosq, nullptr, topic.c_str(),
                                     (int) payload.size(), payload.data(),
                                     g_qos, false);
    if (rc == MOSQ_ERR_SUCCESS) {
        ++g_published;
    } else {
        // Counted, not fatal: a publisher must not take the video down. NO_CONN
        // here means the broker went away mid-run and the client is retrying.
        if (++g_failed == 1)
            fprintf(stderr, "[mqtt] publish failed -- %s (further failures are "
                            "counted, not printed)\n", mosquitto_strerror(rc));
    }
}

}  // namespace

// ===========================================================================
// Entry point: MqttPublish -- after the tracker, metadata only
// ===========================================================================
extern "C" bool MqttPublish(NvDsCustomPostProcessFrameInCpu* in)
{
    lazy_init();

    NvDsFrameMeta* fm = (NvDsFrameMeta*) in->frame_meta;
    // Returning false aborts the pipeline. With no broker the video should still
    // play, so every failure path here lets the frames through.
    if (!fm || !g_up) return true;

    std::vector<NvDsObjectMeta*> objs;
    objs.reserve(in->num_obj_metas);
    for (size_t i = 0; i < in->num_obj_metas; ++i) {
        NvDsObjectMeta* ob = (NvDsObjectMeta*) in->obj_metas[i];
        // MQTT_MIN_CONF is a DETECTOR gate, and an object whose payload is not
        // in its bbox has no detector confidence to gate on. nvinfer's
        // classifier path fabricates one whole-frame object per frame and
        // attach_metadata_classifier assigns its class_id, object_id and
        // rect_params but never its confidence, so it reads 0.0f -- the value a
        // released pool slot is zeroed to. Gating that object on confidence
        // alone therefore discards the ENTIRE classification result, and every
        // attribute hanging off it, for any MQTT_MIN_CONF above zero. Same for
        // an instance mask, whose answer is in mask_params -- and this sink
        // publishes both, because its payload IS frame_json's (attributes[] and
        // mask{} come from frame_json::object below).
        //
        // frame_json::has_payload() is exactly that test and this is the call
        // site its own comment prescribes; it is the identical expression the
        // JSON writer gates on. It reads two pointers on an object the gate
        // already rejected, so a detector -- where both are always NULL -- keeps
        // byte-identical output and pays nothing measurable.
        if (ob && (ob->confidence >= g_min_conf || frame_json::has_payload(ob)))
            objs.push_back(ob);
    }

    // Which parts of the payload this frame actually carries. The two walks are
    // gated on a route having ASKED, so a pipeline with one topic pays for
    // neither -- both flags are then plain false and everything below folds
    // back to the single publish this file has always done.
    const bool have_analytics = frame_json::has_analytics_frame(fm);
    const bool have_objects = !objs.empty();
    const bool have_shapes  = g_want_shapes && has_shapes(fm);
    const bool have_seg     = g_want_seg && has_segmentation(fm);

    // A frame carrying published shapes or a segmentation map is NOT empty,
    // even with no objects on it. Without this, MQTT_SKIP_EMPTY=1 on a
    // detection+pose pipeline would drop every frame in which the detector
    // happened to find nothing, taking that frame's skeletons with it -- the
    // exact failure that left the VMS event log of a pose-only pipeline empty
    // (see templates/vms/sink.py, which relaxed the same gate for the same
    // reason). MQTT_SKIP_EMPTY defaults to 0, so this changes nothing by
    // default and nothing at all without a shapes or segmentation route.
    const bool due = (g_frames % (uint64_t) g_interval) == 0
                     && !(g_skip_empty && !have_objects && !have_shapes && !have_seg
                          && !have_analytics);
    if (due) {
        if (!g_per_object) {
            size_t sent = 0;
            for (size_t i = 0; i < g_routes.size(); ++i) {
                const int s = g_routes[i].stream;
                if (!((s == STREAM_OBJECTS && have_objects)
                      || (s == STREAM_SHAPES && have_shapes)
                      || (s == STREAM_SEG && have_seg)))
                    continue;
                // Built per route: each topic carries its own task's payload,
                // so a wildcard subscriber receives each part once rather than
                // the whole record once per task.
                std::string one;
                one.reserve(128 + (s == STREAM_OBJECTS ? 160 * objs.size() : 256));
                build_split(one, fm, in->W, in->H, objs, g_routes[i]);
                publish(g_routes[i].topic, one);
                ++sent;
            }
            // Nothing matched. A due frame is still reported either way -- a
            // frame that got this far is one a subscriber asked for -- but WHAT
            // is reported depends on whether this pipeline has routes at all.
            if (!sent) {
                const Route* r = fallback_route();
                std::string payload;
                payload.reserve(128 + 160 * objs.size());
                if (r) {
                    // Routed, but this frame carried nothing for any task. It
                    // goes out in the SAME SHAPE as every other message on that
                    // topic -- an empty objects[] -- because a topic whose
                    // messages sometimes have a `task` key and sometimes do not
                    // is the ambiguity `task` exists to remove. (Only reachable
                    // with MQTT_SKIP_EMPTY=0, which is the default.)
                    build_split(payload, fm, in->W, in->H, objs, *r);
                    publish(r->topic, payload);
                } else {
                    // UNROUTED: the whole frame_json record on the base topic.
                    // The only path a single-task pipeline ever takes, and it is
                    // byte-for-byte what this file has always published.
                    frame_json::frame(payload, fm, in->W, in->H, objs);
                    publish(g_topic, payload);
                }
            }
        } else {
            // Topic per class, so a subscriber can filter with an MQTT wildcard
            // (ds/detections/car) instead of parsing everything.
            for (size_t i = 0; i < objs.size(); ++i) {
                std::string one;
                one.reserve(224);
                frame_json::appendf(one, "{\"frame\": %u, \"source_id\": %u,"
                                         " \"pts_ns\": %" PRIu64 ", \"width\": %d,"
                                         " \"height\": %d, \"object\": ",
                                    fm->frame_num, fm->source_id,
                                    (uint64_t) fm->buf_pts, in->W, in->H);
                frame_json::object(one, objs[i]);
                one += '}';
                std::string t = object_base() + "/";
                t += (objs[i]->obj_label[0] ? objs[i]->obj_label : "unknown");
                publish(t, one);
            }
            // Per-object topics cannot represent a frame with zero objects.
            // Keep native frame counters on the configured base topic as one
            // frame record; individual object messages retain object analytics.
            if (have_analytics) {
                std::string payload;
                frame_json::frame(payload, fm, in->W, in->H, objs);
                publish(g_topic, payload);
            }
        }
    }

    g_objects += objs.size();
    ++g_frames;

    if (g_status_every > 0.0 && now_s() - g_t_status >= g_status_every)
        status(false);
    return true;
}
