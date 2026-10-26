# Sentinel-Embedded

**Edge-AI Pipeline for Person Tracking and Hazard-Zone Intrusion Monitoring**

Sentinel-Embedded is a computer-vision pipeline that detects and tracks people in video streams, monitors configurable hazard zones, and publishes intrusion alerts through MQTT. It combines YOLO11s person detection, persistent object tracking, rule-based zone analysis, and NVIDIA TensorRT FP16 inference in an integrated video-processing workflow.

The project explores how edge-AI inference can be integrated with tracking, event detection, and messaging components for industrial monitoring applications.

> **Full project export:** [Access Sentinel-Embedded Core on Google Drive](https://drive.google.com/drive/folders/1CGiPUQAo8Sez8bVkmlQQto03wD_R-3JY?usp=drive_link)  
> The complete export contains larger project files that are not hosted directly on GitHub. The `evidence/` folder below contains selected screenshots and reports. Please check that the Drive folder is accessible to reviewers.

## Key Features

- **Person Detection:** Detect people in video frames using YOLO11s.
- **Persistent Object Tracking:** Associate detections across frames using a Hungarian algorithm-based tracker with Intersection over Union (IoU).
- **Hazard-Zone Monitoring:** Evaluate tracked objects against configurable zones and identify potential intrusions.
- **GPU Inference Optimization:** Compile the detection model into a TensorRT FP16 engine.
- **MQTT Alert Publishing:** Publish configured intrusion events to an MQTT broker.
- **Structured Metadata:** Export frame-level detection, tracking, and zone information as JSON.
- **Visual Monitoring:** Generate annotated video with bounding boxes, tracking IDs, hazard-zone indicators, and intrusion counters.

## System Architecture

The pipeline follows this processing sequence:

1. **Video Input:** Read frames from the configured video source.
2. **Preprocessing:** Prepare frames for model inference.
3. **Object Detection:** Run YOLO11s using the compiled TensorRT FP16 engine.
4. **Postprocessing:** Decode model outputs and filter detections.
5. **Object Tracking:** Associate detections across frames and maintain track IDs.
6. **Zone Analysis:** Evaluate tracked objects against configured hazard-zone boundaries.
7. **Event and Output Generation:** Produce annotated video and structured JSON metadata, and publish intrusion events through MQTT.

## Model and Test Environment

| Component | Configuration |
|---|---|
| Detection model | YOLO11s |
| Inference engine | NVIDIA TensorRT |
| Inference precision | FP16 |
| Test GPU | NVIDIA GeForce RTX 4050 |
| Input resolution | 1920 × 1080 |
| Compiled engine size | 26.9 MB |
| Reported engine build time | 15.6 seconds |

Engine size and build time describe compilation results, not runtime throughput. Runtime performance depends on the hardware, software stack, input workload, and execution configuration.

## Preliminary Evaluation

The TensorRT FP16 model was compared with an ONNX baseline using a 30-sample subset of the COCO 2017 validation dataset.

| Metric | TensorRT FP16 result |
|---|---:|
| mAP@0.5 | 0.4778 |
| mAP@0.5:0.95 | 0.3422 |
| Precision@0.5 | 0.3669 |
| Recall@0.5 | 0.6655 |
| mAP@0.5 difference versus ONNX | −0.0003 |

The reported mAP@0.5 difference is the observed difference from the ONNX baseline on this evaluation subset. It should not be interpreted as a comprehensive assessment of model equivalence or deployment accuracy.

**Evaluation scope:** This is a preliminary experiment on a small sample, not a representative benchmark of industrial environments. COCO-based metrics do not directly measure hazard-zone intrusion detection accuracy.

The pipeline was also executed with object tracking, zone monitoring, and MQTT publishing enabled. Runtime logs and visual outputs provide evidence of component integration. End-to-end latency, independently measured inference throughput, and real-world alert accuracy have not yet been established through repeatable benchmarks. The latency-tracing screenshots are diagnostic evidence only; anomalous tracer values mean they should not be treated as reliable end-to-end latency measurements.

## Outputs

Depending on the selected runtime configuration, the pipeline is designed to produce:

- **Annotated video:** Visualizes detections, track IDs, zone boundaries, and intrusion counters.
- **JSON metadata:** Records frame information, detections, tracking IDs, and zone-related information.
- **MQTT events:** Publishes configured intrusion alerts to the selected broker.

Output filenames and destinations depend on the project configuration. Refer to the exported configuration and runtime instructions for the exact paths.

## Requirements

- NVIDIA GPU supported by the selected TensorRT environment
- Compatible NVIDIA drivers, CUDA, and TensorRT versions
- GStreamer and the required pipeline dependencies
- Docker, if required by the exported runtime
- Input video and pipeline configuration
- MQTT broker for alert delivery

Exact dependency versions, environment variables, and execution commands should be taken from the included project configuration and setup files.

## Running the Project

The full exported project and its configuration files are available in the [Google Drive export](https://drive.google.com/drive/folders/1CGiPUQAo8Sez8bVkmlQQto03wD_R-3JY?usp=drive_link).

General workflow:

1. Set the input video source and hazard-zone parameters.
2. Configure MQTT connectivity and output destinations.
3. Prepare the runtime environment and compile the TensorRT engine, if it is not already available.
4. Launch the pipeline using the execution commands provided with the export.
5. Inspect the generated video, JSON metadata, runtime logs, and MQTT events.

Exact launch commands are not reproduced here because they depend on the exported runtime configuration. Use the commands supplied with the export.

## Evidence

The following files are currently present in the repository's `evidence/` directory:

- [Pipeline IR graph](evidence/IR_graph.png.png)
- [TensorRT engine build log](evidence/TensorRT%20engine_build.png.png)
- [FP16 benchmark report](evidence/Benchmark%20F16.png.png)
- [Accuracy evaluation](evidence/accuracy.png.png)
- [ONNX validation/check](evidence/onnx_check.png.png)
- [Live annotated run](evidence/live_run.png.png)
- [MQTT listener output](evidence/mqtt_listener.png.png)
- [Latency tracing capture 1](evidence/latency_tracing.png.png)
- [Latency tracing capture 2](evidence/latency_tracing.2png.png)
- [INT8 comparison](evidence/INT8%20Comparsion.png.png)

The INT8 comparison is a separate experiment; the evaluation table above reports the FP16 result. Tracing captures are included for diagnostic transparency and should not be read as validated end-to-end latency benchmarks.

## Limitations

- Evaluation is limited to a small subset of COCO 2017 validation data.
- Real-world performance in industrial environments has not been comprehensively evaluated.
- Hazard-zone alert precision, recall, and false-alarm rates require dedicated testing.
- End-to-end latency and inference throughput require repeatable measurements.
- Tracking reliability can degrade with occlusion, crowded scenes, and challenging camera placement.
- Deployment requirements depend on the target GPU, software environment, and messaging configuration.

## Future Work

- Establish repeatable end-to-end latency and throughput benchmarks.
- Evaluate detection and tracking on additional datasets and representative video scenarios.
- Measure hazard-zone alert precision, recall, false positives, and false negatives.
- Test robustness under occlusion, variable lighting, and crowded conditions.
- Document dependency versions, deployment steps, and runtime configuration.
- Improve event reliability and monitoring for deployment-oriented use cases.

## Acknowledgements

Built with YOLO, NVIDIA TensorRT, GStreamer, and the associated project tooling and dependencies. Refer to the relevant upstream projects for their licenses and attribution requirements.

## License

This repository is distributed under the [MIT License](LICENSE). Third-party models, libraries, and dependencies may have separate license terms; review their respective licenses before redistribution or deployment..

*Thank you*
