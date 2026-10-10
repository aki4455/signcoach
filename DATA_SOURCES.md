# Data sources, models and licences

Every dataset, model weight and third-party component used by SignCoach (CLAUDE.md hard rule 5).
Weights and datasets are not in this repository; they are downloaded into `$SIGNCOACH_DATA`.

## Datasets

| Name | Link | Used for | Licence | Attribution |
|---|---|---|---|---|
| Dognellaf / russian-road-signs | https://huggingface.co/datasets/Dognellaf/russian-road-signs | Training and validating the sign detector (38,918 labelled frames, 89 classes, split by source video) | CC BY 4.0 (LICENSE file in the dataset repo, checked 2026-10-10) | "Russian Road Signs" dataset by Dognellaf, CC BY 4.0 |
| Moscow dashcam clips `ru_01`…`ru_04` (3 min each) | provided by the team (shared via Telegram) | Sample lessons, demo uploads, evaluation (speed: `ru_01` calibration, `ru_02` test) | Free to use (confirmed by the team, 2026-10-11); only blurred, soundless outputs are published; raw clips are never in git | SignCoach team |

## Model weights

| Name | Link | Used for | Licence | sha256 |
|---|---|---|---|---|
| YOLO11s (COCO) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt | Starting point for fine-tuning the sign detector | AGPL-3.0 (Ultralytics) | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| YOLO11n (COCO) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt | Vehicles and persons (plate/head blur, masks for speed) | AGPL-3.0 (Ultralytics) | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| StreetSignSense YOLO12s | https://huggingface.co/AlessandroFerrante/StreetSignSenseY12s | Placeholder sign detector until the fine-tuned model is ready | CC BY 4.0 (model card; README also shows an MIT badge); trained with Ultralytics (AGPL-3.0) | `ea716000d066b3c48de91583210e8934de1865cb1956857c9c0fb89f7ec9ef9e` |
| SignCoach sign detector (fine-tuned YOLO11s) | trained by us (`training/`) | Sign detection | AGPL-3.0 (derived from Ultralytics weights) | recorded after training (M6) |
| YuNet face detector (`face_detection_yunet_2023mar.onnx`) | https://huggingface.co/opencv/face_detection_yunet (OpenCV Zoo) | Face blur | MIT (LICENSE file checked 2026-10-10), © 2020 Shiqi Yu | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Licence-plate detector YOLO11s (`license-plate-finetune-v1s.pt`) | https://huggingface.co/morsetechlab/yolov11-license-plate-detection | Plate blur, together with a fallback area on every vehicle box (`privacy.plate_band`) | AGPL-3.0 (model card); trained on Roboflow "license-plate-recognition-rxg4e" — the card notes train/test leakage, so we judged it on our own clips | `95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381` |
| Piper TTS (`piper-tts` 1.8, offline tool, optional `voice` dependency group — not installed on the server) | https://github.com/OHF-Voice/piper1-gpl | Generating the English voice prompt MP3s (`scripts/generate_voice.py`) | GPL-3.0-or-later | — |
| Piper voice `en_US-ljspeech-high` | https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ljspeech/high | Voice of the prompts in `assets/voice/en/` | Trained on the LJ Speech dataset (public domain, per the model card) | `5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a` |

## Software

| Component | Licence |
|---|---|
| Ultralytics (YOLO training, inference, ByteTrack) | AGPL-3.0 — the reason this repository is AGPL-3.0 |
| PyTorch, torchvision | BSD-3-Clause |
| OpenCV (opencv-python-headless) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| FastAPI, Pydantic, python-multipart | MIT / Apache-2.0 |
| Uvicorn | BSD-3-Clause |
| PyYAML, python-dotenv, huggingface_hub, lap | MIT / BSD / Apache-2.0 / BSD-2-Clause |
| FFmpeg (system package) | LGPL/GPL |
| React, Vite, Tailwind CSS, TypeScript | MIT / MIT / MIT / Apache-2.0 |

## Related work (cited only — no code used)

- Aillustrate/traffic-sign-detection — no licence, so nothing from it is used.
