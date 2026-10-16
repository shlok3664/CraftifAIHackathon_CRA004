// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

static float iouCalc(float l1, float t1, float w1, float h1,
                     float l2, float t2, float w2, float h2)
{
    float ix1 = std::max<float>(l1, l2);
    float iy1 = std::max<float>(t1, t2);
    float ix2 = std::min<float>(l1 + w1, l2 + w2);
    float iy2 = std::min<float>(t1 + h1, t2 + h2);
    float iw = std::max<float>(ix2 - ix1, 0.0f);
    float ih = std::max<float>(iy2 - iy1, 0.0f);
    float inter = iw * ih;
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float unionA = area1 + area2 - inter;
    return (unionA > 0.0f) ? (inter / unionA) : 0.0f;
}

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[ParseConfigDetPgie] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // YOLO11s output shape (batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // Layout: 84 x N where 84 = 4 box coords + 80 class scores
    // Need to transpose: for proposal i, box = data[row * N + i] for row 0..3,
    // class scores = data[(4+c) * N + i].

    const unsigned int featureDim = layer0.inferDims.d[0]; // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        fprintf(stderr, "[ParseConfigDetPgie] WARNING: featureDim=%u != 4+numClasses=%u\n",
                featureDim, 4u + numClasses);
    }

    const float nmsThreshold = 0.45f;

    // Temporary storage for candidates before NMS
    struct Detection {
        float left, top, width, height;
        float confidence;
        unsigned classId;
    };
    std::vector<Detection> candidates;
    candidates.reserve(512);

    for (unsigned int i = 0; i < numProposals; ++i) {
        // Box coords (cx, cy, w, h) in network-input pixel space
        float cx = data[0 * numProposals + i];
        float cy = data[1 * numProposals + i];
        float bw = data[2 * numProposals + i];
        float bh = data[3 * numProposals + i];

        // Find best class score — scores are post-sigmoid probabilities
        float maxScore = 0.0f;
        unsigned bestClass = 0;
        for (unsigned c = 0; c < numClasses; ++c) {
            float s = data[(4 + c) * numProposals + i];
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        // Only keep class 0 (person)
        if (bestClass != 0) continue;

        float thr = (bestClass < static_cast<unsigned>(detectionParams.perClassPreclusterThreshold.size()))
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.35f;
        if (maxScore < thr) continue;

        // Convert cx/cy/w/h to left/top/width/height, clamp to network dims
        float left = std::max<float>(cx - bw * 0.5f, 0.0f);
        float top  = std::max<float>(cy - bh * 0.5f, 0.0f);
        float right  = std::min<float>(cx + bw * 0.5f, netW);
        float bottom = std::min<float>(cy + bh * 0.5f, netH);
        float w = right - left;
        float h = bottom - top;
        if (w <= 1.0f || h <= 1.0f) continue;

        Detection det;
        det.left = left;
        det.top = top;
        det.width = w;
        det.height = h;
        det.confidence = maxScore;
        det.classId = bestClass;
        candidates.push_back(det);
    }

    // Sort by confidence descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // Greedy NMS
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        const Detection& a = candidates[i];
        NvDsInferObjectDetectionInfo obj;
        obj.classId = a.classId;
        obj.left = a.left;
        obj.top = a.top;
        obj.width = a.width;
        obj.height = a.height;
        obj.detectionConfidence = a.confidence;
        objectList.push_back(obj);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            const Detection& b = candidates[j];
            if (b.classId != a.classId) continue;
            float iou = iouCalc(a.left, a.top, a.width, a.height,
                                b.left, b.top, b.width, b.height);
            if (iou > nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }
    // ==== END GENERATED DECODE ====
    return true;
}