# Sentinel: Workplace Safety Vision System

**Embedded SF: CraftifAI Physical AI Hackathon | Track A (PipeGen - Perception)**

Sentinel is a workplace-safety vision system built on CraftifAI PipeGen. It processes video input, detects and tracks people using a YOLO11s model compiled into a TensorRT FP16 engine, monitors foot positions relative to a defined hazard zone, and emits real-time alerts.

## Key Features & Architecture
- **Model & Compiler:** YOLO11s compiled to TensorRT FP16 (26.9 MB, 237 layers).
- **Tracking:** Persistent Hungarian IoU tracker with track IDs.
- **Rule Engine:** Hazard-zone entry alert (0.5s hold condition, 30s repeat pause).
- **Outputs:** Real-time MQTT stream (`sentinel/alerts` @ ~60 msg/s), per-frame JSON metadata (`metadata.json`), derived incident log (`incidents.json`), and live annotated video overlay.
- **Hardware:** Intel/NVIDIA RTX 4050 GPU running via WSL2 Ubuntu and Docker.

## Performance
- **Accuracy:** FP16 matched ONNX baseline within -0.0003 mAP@0.5 on a COCO 2017 validation slice.
- **Throughput:** ~60 FPS real-time execution with 0 dropped frames (median latency ~16.58 ms).

## Repository Contents
- `pipeline/deepstream/`: Generated DeepStream application source and build scripts.
- `model_compile_output/`: TensorRT engine build artifacts and config files.
- `output/`: Output JSON metadata (`metadata.json`) and processed incident log (`incidents.json`).
- `evidence/`: IR graph, speed benchmarks, and live run screenshots.
