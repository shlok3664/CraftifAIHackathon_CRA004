# Sentinel-Embedded

**Real-Time Hazard-Zone Monitoring Using Edge AI**

Sentinel-Embedded is a computer-vision pipeline for detecting and tracking people in video streams and identifying potential intrusions into configurable hazard zones. It combines YOLO11s person detection, persistent object tracking, rule-based zone monitoring, and MQTT-based alert delivery.

## Features

- **Person Detection:** YOLO11s object detection.
- **Object Tracking:** Persistent tracking using a Hungarian IoU-based tracker.
- **Hazard-Zone Monitoring:** Configurable zones with intrusion detection and alert conditions.
- **TensorRT Optimization:** FP16 engine compilation for NVIDIA GPU inference.
- **MQTT Alerts:** Publish confirmed intrusion events to an MQTT broker.
- **JSON Metadata:** Structured per-frame detection and zone information.
- **Video Visualization:** Annotated video with tracking IDs, zone boundaries, and intrusion counters.

## Pipeline Architecture

1. **Input:** Read video frames and timestamps.
2. **Preprocessing:** Prepare frames for model inference.
3. **Inference:** Run YOLO11s person detection using the compiled TensorRT engine.
4. **Postprocessing:** Decode detections and filter bounding boxes.
5. **Tracking:** Associate detections across frames using persistent object IDs.
6. **Zone Monitoring:** Evaluate tracked objects against configured hazard zones.
7. **Outputs:** Generate annotated video, JSON metadata, and MQTT alerts.

## Model and Hardware

- **Detection Model:** YOLO11s
- **Inference Precision:** TensorRT FP16
- **Test GPU:** NVIDIA GeForce RTX 4050
- **Input Resolution:** 1920 × 1080
- **Engine Size:** 26.9 MB
- **Reported Engine Build Time:** 15.6 seconds

Hardware and software configurations may affect compilation time and runtime performance.

## Preliminary Evaluation

The FP16 TensorRT engine was compared against the ONNX baseline using a 30-sample slice of the COCO 2017 validation dataset.

| Metric | Result |
|---|---:|
| mAP@0.5 | 0.4778 |
| mAP@0.5:0.95 | 0.3422 |
| Precision@0.5 | 0.3669 |
| Recall@0.5 | 0.6655 |
| mAP@0.5 difference vs. ONNX | -0.0003 |

These results represent a preliminary evaluation on a small sample and do not establish real-world workplace-safety accuracy.

The pipeline was executed with object tracking, hazard-zone monitoring, and MQTT publishing enabled. Runtime logs show tracked detections and structured alert messages. Reliable end-to-end latency and independently measured inference throughput have not yet been established.

## Output

The pipeline generates:

- **Annotated Video:** Displays detections, tracking IDs, zone boundaries, and intrusion counters.
- **JSON Metadata:** Contains frame information, detections, tracking IDs, and zone statistics.
- **MQTT Events:** Publishes confirmed intrusion alerts to the configured broker.

## Requirements

- NVIDIA GPU compatible with the selected TensorRT environment
- Compatible NVIDIA drivers, CUDA, and TensorRT
- Docker, if required by the exported runtime
- Input video and pipeline configuration
- MQTT broker for alert publishing

Refer to the exported project's configuration and setup instructions for the exact environment requirements.

## Running the Project

1. Configure the input video and hazard-zone parameters.
2. Configure the MQTT broker and output settings.
3. Compile the pipeline and TensorRT engine.
4. Run the exported pipeline using the documented execution commands.
5. Inspect the annotated video, JSON metadata, and MQTT messages.

See the exported project files for the exact installation and execution commands.

## Repository Contents

- `README.md` — Project overview and documentation
- `metadata.json` — Structured frame and detection metadata, if included in the export
- `evidence/` — Screenshots documenting the pipeline, compilation, evaluation, and runtime output

The exact files may vary depending on the exported project version.

## Limitations

- Evaluation currently uses a small COCO 2017 validation sample.
- Real-world hazard-zone detection accuracy has not been comprehensively evaluated.
- End-to-end latency and inference throughput require further benchmarking.
- Tracking performance may be affected by occlusion, crowded scenes, and camera placement.

## Future Improvements

- Benchmark end-to-end latency and throughput under repeatable conditions.
- Evaluate detection and tracking on additional video datasets.
- Measure false positives and false negatives for hazard-zone alerts.
- Improve alert reliability and document deployment requirements.

## Acknowledgements

Built using YOLO, NVIDIA TensorRT, GStreamer, and the associated project tooling and dependencies. See the relevant upstream projects for licensing and attribution details.

## License

This project is distributed under the license specified in the repository. Refer to `LICENSE` for details.
