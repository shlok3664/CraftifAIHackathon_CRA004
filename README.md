# Sentinel: Workplace Safety Vision System

**Embedded SF: CraftifAI Physical AI Hackathon | Track A (PipeGen, Perception)**

## Problem
Workers enter hazard zones and nobody notices until an incident happens. Sentinel watches a camera feed, tracks every person, and raises an alert when someone's feet enter a defined hazard zone.

## What it does
- Reads a video file (a pedestrian crossing, 60 fps).
- Detects people with a **YOLO11s** COCO detector and tracks them with persistent IDs (Hungarian IoU tracker).
- Flags a **hazard-zone entry** when a tracked person's foot point (bottom-center of the box) stays inside the zone for 0.5 s, with a 30 s pause between repeat alerts.
- Produces three outputs: live **MQTT** messages (`sentinel/alerts`, Mosquitto), a per-frame **JSON export** (`metadata.json`), and an annotated **live display**.

## Pipeline (built with PipeGen's five stages)
| Stage | Choice |
|---|---|
| Input | Video file, 1080x1920, 60 fps, fixed camera |
| Tasks | Person detection, person tracking, hazard-zone entry alert |
| Model | YOLO11s (COCO) ONNX from the PipeGen catalog |
| Precision | TensorRT **FP16** engine (26.9 MB, 237 TRT layers, built in 15.6 s) |
| Outputs | MQTT, JSON metadata, live display |

The Pipeline IR, ONNX check, TensorRT build, accuracy check, DeepStream app and `run.sh` were all generated through PipeGen. The requirement, scene settings, zone definition and evaluation were mine.

## Results
- **ONNX check:** 321 of 323 layers supported by TensorRT, 0 unresolved.
- **Accuracy:** the FP16 engine matched the ONNX baseline within 0.0003 mAP@0.5 (mAP@0.5 = 0.4778) on a **30-image COCO 2017 validation slice**. The slice is small, so treat this as a compile-fidelity check, not a benchmark score.
- **Real-time:** the pipeline kept pace with the 60 fps source and published about 60 MQTT messages per second. Latency: <FILL IN from the PipeGen benchmark report in evidence/>.
- **Demo run:** 711 frames processed; 678 contained at least one zone intrusion and up to 11 people were in view at once. The zone covers the busiest part of the crosswalk, so intrusions are frequent.

## Output format
`output/metadata.json` has one record per frame: the zone polygon, `people_count`, `intrusion_count`, `zone_name`, and each person's track ID, confidence and bounding box. The same data is published frame by frame over MQTT.

## Repository contents
- `pipeline/deepstream/`: generated DeepStream application and run script
- `model_compile_output/`: TensorRT build artifacts and reports
- `output/`: `metadata.json` from the demo run
- `evidence/`: IR graph, speed and accuracy reports, live run screenshots

## Full PipeGen export
The complete export is too large for GitHub:
<https://drive.google.com/drive/folders/1CGiPUQAo8Sez8bVkmlQQto03wD_R-3JY?usp=drive_link>

## How to run
Needs an NVIDIA GPU and Docker. Start an MQTT broker on port 1883, then run `run.sh` from the export folder. Listen for alerts with `mosquitto_sub -t "sentinel/#" -v`.

## Setup notes
Built and tested on an MSI laptop (NVIDIA RTX 4050 Laptop GPU, 6 GB) using Windows, WSL2 Ubuntu and Docker.
