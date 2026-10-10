# Data sources, models and licences

Every dataset, model weight and third-party component used by SignCoach (CLAUDE.md hard rule 5).
Weights and datasets are not in this repository; they are downloaded into `$SIGNCOACH_DATA`.

## Datasets

| Name | Link | Used for | Licence | Attribution |
|---|---|---|---|---|
| Dognellaf / russian-road-signs | https://huggingface.co/datasets/Dognellaf/russian-road-signs | Training and validating the sign detector (38,918 labelled frames, 89 classes, split by source video) | CC BY 4.0 (LICENSE file in the dataset repo, checked 2026-10-10) | "Russian Road Signs" dataset by Dognellaf, CC BY 4.0 |
| Team recordings | — | Demo lessons and evaluation | Recorded/provided by the team; published only blurred, without audio | SignCoach team |

## Model weights

| Name | Link | Used for | Licence | sha256 |
|---|---|---|---|---|
| YOLO11s (COCO) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt | Starting point for fine-tuning the sign detector | AGPL-3.0 (Ultralytics) | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| YOLO11n (COCO) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt | Vehicles and persons (plate/head blur, masks for speed) | AGPL-3.0 (Ultralytics) | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| StreetSignSense YOLO12s | https://huggingface.co/AlessandroFerrante/StreetSignSenseY12s | Placeholder sign detector until the fine-tuned model is ready | CC BY 4.0 (model card); trained with Ultralytics (AGPL-3.0) | recorded by `scripts/download_models.py` (M1) |
| SignCoach sign detector (fine-tuned YOLO11s) | trained by us (`training/`) | Sign detection | AGPL-3.0 (derived from Ultralytics weights) | recorded after training (M6) |
| YuNet face detector | OpenCV Zoo — to be verified in M1 | Face blur | to be verified | — |
| Licence-plate detector | to be decided in M1 | Plate blur (fallback: blur band inside vehicle boxes) | to be verified | — |
| Text-to-speech model for English voice prompts | to be decided in M4 | Pre-generated voice prompt MP3s | to be verified | — |

## Software

| Component | Licence |
|---|---|
| Ultralytics (YOLO training, inference, ByteTrack) | AGPL-3.0 — the reason this repository is AGPL-3.0 |
| PyTorch, torchvision | BSD-3-Clause |
| OpenCV (opencv-python-headless) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| FastAPI, Pydantic, python-multipart | MIT / Apache-2.0 |
| Uvicorn | BSD-3-Clause |
| PyYAML, python-dotenv, gpxpy, huggingface_hub, lap | MIT / BSD / Apache-2.0 / Apache-2.0 / BSD-2-Clause |
| FFmpeg (system package) | LGPL/GPL |
| React, Vite, Tailwind CSS, TypeScript | MIT / MIT / MIT / Apache-2.0 |

## Related work (cited only — no code used)

- Aillustrate/traffic-sign-detection — no licence, so nothing from it is used.
