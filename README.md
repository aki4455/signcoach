# SignCoach — AI co-instructor for driving schools

A driving-school instructor uploads the **dashcam video** of a student's lesson and gets a report of
every road sign the car passed, what it means (official text of the traffic rules), which rule was in
force, and where the student may have broken it — with the video moment as evidence, voice prompts at
the moments they would have sounded in the car (when the sign is recognised, a median 2.1 s before
passing it), and faces and number plates blurred.

- **Live site:** https://signcoach.taild1ce46.ts.net
- **Built for:** WIUT Hackathon 2026 final, Computer Vision track. Product target: driving schools in
  Uzbekistan; the demo uses Russian dashcam footage and the Russian Road Traffic Rules (ПДД РФ).
- **Results:** every number is produced by a script in `eval/` and stored in [`eval/results/`](eval/results/).

## How it works

```
 upload (chunked, resumable) ──> job queue (SQLite, one worker, models stay loaded)
                          │
                          ▼
  ingest ──> every n-th frame ──┬──> sign detector (our fine-tuned YOLO11s) + our sign tracker ──> track voting ──> relevance
  (probe, limits, normalize:    ├──> vehicles & people (YOLO11n) + licence plates (YOLO11s) ──> blur, speed masks
   fps, HDR, no audio)          ├──> ego-motion from the video: speed (road seen from above), turns, stops
                                └──> faces (YuNet) in the render pass
                                                          │
                       rules engine (Russian traffic rules, deterministic, pure) ──> lesson events + rules timeline
                                                          │
                       render: blurred 720p video with sign boxes, limit/speed display, subtitles; thumbnails
                                                          │
                                         report.json ──> report page (timeline, voice prompts, events, official texts)
```

- **No LLM in the pipeline.** Everything is our own deterministic code plus our detectors.
- **Sign detector:** YOLO11s fine-tuned on 17,066 frames of the Dognellaf Russian road-sign dataset
  (89 classes), split **by source video** so validation frames come from videos never seen in training.
- **Sign tracker (ours):** links one sign's boxes from frame to frame by predicting where it moves
  (direction, speed and growth on screen). Ultralytics' ByteTrack needs boxes to overlap, which fails for
  close signs at city/highway speed — see [`eval/results/tracker_comparison.md`](eval/results/tracker_comparison.md).
- **Track voting:** a sign counts only after several frames agree (sum of confidences per class).
- **Relevance:** side of the road (by how the sign drifts as we approach), whether it faces us, whether
  we approach it — every decision is written down in words in the report.
- **Speed without GPS:** the road is projected to a bird's-eye view and its forward slide between frames
  is measured by phase correlation, ignoring things fixed to the camera (dashboard, on-screen text,
  reflections) and other vehicles. Turns come from the sideways drift of distant scenery.
- **Rules:** the official texts of the signs, the zones in which they apply and the default speed limits
  were copied verbatim from ПДД РФ (Government Decree No. 1090, ред. от 28.08.2026) by
  `scripts/extract_pdd.py` and checked by a Russian-speaking team member (`docs/RULES_CHECKLIST.md`).
  The engine reports speeding, possible speeding, stop not made / made, turns against a sign, possible
  no-entry (an overhead sign, or a side sign we turned toward), and explains each new sign once.
- **Honesty by design:** a violation is reported only when the evidence is strong; otherwise it is
  "possible" (speed within its error, an assumed default limit, a sign with a conditions plate, or a
  camera whose speed was never calibrated). Unmeasurable speed is "not measured", never a guess.

More detail per module: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md). Build spec: [`CLAUDE.md`](CLAUDE.md).

## Results

Every number below is written into `eval/results/` by a script (`eval/*.py`, or `training/compare_models.py` for the model comparison); none is typed by hand.

| What we measured | Result | Source |
|---|---|---|
| Sign detector on 3,027 frames from 119 videos never seen in training (all 89 classes) | precision 0.71, recall 0.65, mAP50 0.70, mAP50-95 0.57 | [`detector_eval.md`](eval/results/detector_eval.md) |
| Our detector vs a pretrained European sign model, on the signs both know (same frames) | mean AP50 **0.81 vs 0.65**; 17 vs 26 ms per image on an RTX 3070 Ti | [`model_comparison.md`](eval/results/model_comparison.md) |
| Our sign tracker vs ByteTrack on 12 demo clips (8 Tashkent, 4 Moscow) | **254 vs 166** signs kept after voting (170 vs 115 judged to apply to us) | [`tracker_comparison.md`](eval/results/tracker_comparison.md) |
| Speed from video vs the dashcam's on-screen GPS speed, test clip ru_02 (highway, 64–89 km/h) | estimate for 28% of moments, mean error 45.6 km/h — **fails at highway speed** | [`speed_eval.md`](eval/results/speed_eval.md) |
| Processing time and energy (6 clips: 4 Tashkent 720p, 2 Moscow 1080p; RTX 3070 Ti) | **95 s per minute of video** (1.45–1.74× real time — slower than our real-time target); GPU board power 56 W on average → 0.089 kWh per lesson-hour | [`benchmark.md`](eval/results/benchmark.md) |
| Failure cases on the held-out frames | the largest misses and most confident false alarms, as images | [`failure_gallery/`](eval/results/failure_gallery/) |

## Run it locally

System packages: `ffmpeg` (with `ffprobe`), Node.js LTS + npm, `git`, [`uv`](https://docs.astral.sh/uv/). An NVIDIA GPU is
optional (`SIGNCOACH_DEVICE=cpu` works, slower).

```bash
uv sync                                   # Python 3.12 environment (PyTorch, Ultralytics, FastAPI, ...)
(cd web && npm ci && npm run build)       # frontend -> web/dist
cp .env.example .env                      # set SIGNCOACH_DATA, SIGNCOACH_DEVICE (cpu | cuda:0)
uv run python scripts/download_models.py  # pretrained weights with pinned sha256 -> $SIGNCOACH_DATA/models
uv run uvicorn app.main:app --port 8001   # http://127.0.0.1:8001
uv run pytest -q                          # tests (add -m "not slow" to skip the end-to-end run)
```

The fine-tuned sign detector is not in git (weights never are); train it as below or ask the team for
`dognellaf_y11s_1024.pt` and put it in `$SIGNCOACH_DATA/models/`.

## Train the sign detector

```bash
cd ~/datasets && hf download Dognellaf/russian-road-signs --repo-type dataset --local-dir russian-road-signs --exclude "unlabeled/*"
uv run python -I training/prepare_dognellaf.py --images ~/datasets/russian-road-signs/images \
    --labels ~/datasets/russian-road-signs/labels --classes ~/datasets/russian-road-signs/classes.txt \
    --out ~/signcoach-data/datasets/dognellaf_yolo      # completeness check, inventory, split by video
training/smoke_test.sh                                   # 1 epoch on 5%
training/start_full_training.sh 40                       # YOLO11s, 1024 px, 40 epochs, fliplr=0 (never mirror signs)
```

## Evaluate

```bash
uv run python eval/eval_detector.py --weights <best.pt> --imgsz 1024          # held-out videos
uv run python training/compare_models.py --model streetsignsense_y12s:1280 --model dognellaf_y11s:1024:<best.pt>
uv run python eval/benchmark.py ~/signcoach-data/datasets/own/ru_0*.mp4       # processing time and cost
uv run python eval/failure_gallery.py --weights <best.pt>                      # failure cases
uv run python scripts/calibrate_speed.py --video <ru_01> --truth eval/ground_truth/speed_ru_01.csv --write
uv run python eval/eval_speed.py --clip ru_02 <ru_02> eval/ground_truth/speed_ru_02.csv
```

## Deployment

A push to `main` triggers `.github/workflows/deploy.yml` on the team machine's self-hosted runner: it
checks out the commit in a separate deploy folder, installs, builds the frontend, runs the fast tests,
restarts the `signcoach` systemd service and waits until `/health` reports the new commit. Tailscale
Funnel publishes it over HTTPS. Setup steps: [`deploy/FUNNEL.md`](deploy/FUNNEL.md).

## Privacy

- Faces (YuNet), people's head areas, licence plates (plate detector) and a fallback area on every
  vehicle are pixelated in **every** published frame and thumbnail; plates found in neighbouring frames
  too. Sound is removed. The original upload and the unblurred working copy are deleted after processing.
- Uploads require the uploader's confirmation that they may upload the video.
- There is no public list of uploads: lesson links are long random ids; the lessons page shows the
  sample lessons and the lessons uploaded from the same browser.

## Limitations and known failure cases

- **No map.** "Until the next intersection" is approximated by our next turn, 300 m or 60 s. A no-entry sign
  at the side of the road usually closes a side road, so it is reported only if we turned toward it
  (overhead ones always): driving straight into a one-way street past a side sign is missed.
- **Speed-limit digits** are sometimes misread (an 80 read as 60 in a winter video, a 30 read as 50 on a
  Tashkent clip); voting over many frames fixes some, not all ([`failure_gallery/`](eval/results/failure_gallery/)).
  The confidence of a reading does not tell right from wrong (misreads 0.54–0.73, correct readings 0.55–0.81
  on the demo clips), so no threshold can filter them.
- **Round signs the detector was never trained on** are read as the closest class it knows: "no trucks"
  (3.4) as a 100 km/h limit (end of the Islam Karimov street sample), "no pedestrians" (3.10) as 70.
- **The back of a sign** can be read as a sign: the grey back of a round sign became "right turn only"
  (4.1.2) at 0:44 in the Islam Karimov street sample. A colour check (blue signs must contain blue,
  red-ringed signs red) could reject such boxes; we did not build it in time.
- **Signs missing from the training data:** no-left-turn (3.18.2), no-U-turn (3.19), start/end of a
  built-up area (5.23/5.24); end-of-speed-limit (3.25) has only 5 examples. Without settlement signs the
  built-up-area default limit is assumed, so speeding against it is only "possible".
- **Speed from video is an estimate.** On the highway test clip it was measurable only 28% of the time with
  a 45.6 km/h mean error ([`speed_eval.md`](eval/results/speed_eval.md)); it is sensitive to the horizon
  estimate, can read 0 km/h on dark featureless asphalt, and misses some short stops. It is calibrated on
  one dashcam only, so speeding is reported as "possible" for all uploads.
- **Relevance heuristics** can misjudge signs on wide roads, slip roads and sharp bends.
- Processing is designed for uploads up to 3 minutes and takes about 1.6× the video's length on our RTX 3070 Ti
  (analysis and rendering the blurred video take about half each); the target was real time.

## Data sources and licences

See [`DATA_SOURCES.md`](DATA_SOURCES.md). The repository is **AGPL-3.0** (it builds on Ultralytics,
AGPL-3.0). A commercial launch would need an Ultralytics Enterprise licence or a permissively licensed
detector.

## How this was built

The code was written with Claude Code (Anthropic) following the build spec in `CLAUDE.md`, milestone by
milestone; every milestone was reviewed, tested and merged by the team, who also chose the data, checked
the official rule texts and recorded the evaluation ground truth. Elimination-stage code reused: none.
