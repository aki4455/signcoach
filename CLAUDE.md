# SignCoach — AI co-instructor for driving schools

> Build spec for Claude Code. Read the whole file before writing any code. It is the source of truth for scope, data contracts and milestones.
> If a human instruction in chat conflicts with this file, the human wins — then update this file in the same commit so it stays true.
> "SignCoach" is a working name; the team may rename it.

---

## Start here (status at Saturday 22:40)

**Decisions already made:**
- We **fine-tune** a sign detector on the Dognellaf dataset (§14.2). **Training (T5) is running** detached in tmux session `signcoach-train` (started 22:29, 40 epochs, ~6.5 min/epoch, expected end ≈ 02:50). Check progress in `~/signcoach-data/train_runs/dognellaf_y11s_1024_*/results.csv`.
- **Deadline: Sunday 11 October 2026, 13:00 Asia/Tashkent.**
- **Demo footage is Russian dashcam video** that the team gives the jury to upload. Sign codes are the **Russian GOST numbers** used by the training data (e.g. 3.24 = speed limit). No Uzbek numbering layer.
- **English only.** No UZ/RU localisation; voice prompts are **English, generated with a text-to-speech model** (not recorded by the team).
- `MODEL_COVERAGE_TEST.md` is **dropped** (it only decided whether to fine-tune). M6 uses the comparison method defined in §14.2 T6; "core signs" = the §11.1 list.
- The review fixes approved on Saturday 22:00 are folded into this file (§9 `Report.signs`, §10.1 fps/HDR, §10.2 two passes, §11.4 semantics, §12 API/worker, §16 service).
- Until the fine-tuned model exists, everything is built **model-independent**: the sign detector sits behind an adapter (§10.3) with a replay implementation (detections from JSON) and an Ultralytics implementation (placeholder now, fine-tuned later). Swapping models is a config change.

**Order of work from now:**
1. **M0** skeleton + deploy (code done; waiting for the humans' GitHub login and root setup, §16).
2. **M1** the whole pipeline without depending on the final model (ingest, frames, detector adapter, voting, relevance, vehicles, blur, render, report, API, worker). Runs end to end with the replay detector and the placeholder model.
3. **M2** rules engine and **M4** frontend only need the §9 contracts and the fixture (both exist after M0) — a second Claude Code session can do them in parallel (M2 first, then M4), working in its own `git worktree` on its own branch. If you are that second session, leave M0, M1 and the training track to the first session, and don't edit this file — ask the first session to.
4. **Training track T** (§14.2): T1–T4 done, T5 running. Report training progress in every review.
5. While training runs the GPU is busy: run the app and tests with `SIGNCOACH_DEVICE=cpu`.

---

## 0. Situation

- **Event:** WIUT Hackathon 2026 final, Computer Vision track (partners: Toyota Royal Motors, BYD Royal Motors, Ducati).
- **Time:** under 24 hours in total. **Deadline: Sunday 11 October 2026, 13:00 Asia/Tashkent.** At the deadline the repo is archived and the deployed site is monitored; after it, only the submitted commit may be redeployed.
- **Roles:** you (Claude Code) write the code. Humans review every milestone, test, deploy, record data, label ground truth, write official sign texts and make the deck. Voice prompts are generated with a TTS model (§5).
- **Deliverables:** (1) a public, deployed website that runs end to end, including on a video the jury gives us; (2) a public GitHub repo with code, a README to run it, and a list of data sources with licences; (3) a deck (humans).
- **Judging:** technical complexity 25%, innovation 25%, impact 25%, presentation 15%, business 10%. The technical criteria reward: a real system rather than one call to someone else's model, quality measured on **our own** test set, speed and cost per request, and honest failure cases. The jury may question any team member about code written with an LLM, so code must be readable and documented well enough for the team to explain it.

## 1. Product

**One sentence:** a driving-school instructor uploads the dashcam or phone video of a student's lesson and gets a report of every road sign the car passed, what it means, which rule was in force, and where the student broke it — with video evidence.

- **Users:** driving schools in Uzbekistan (instructors and students). Startup mode.
- **Demo:** the jury uploads Russian dashcam recordings that the team provides. The interface and voice prompts are in English.
- **Flow:** record the lesson with a phone camera on the windshield or any dashcam → (optional) record a GPX track with a GPS logger app → upload on the website → processing → lesson report.
- **Report (see §13 for layout):** annotated, privacy-blurred video with a speed/limit overlay; voice prompts played at the moments they would have sounded in the car; event list (sign explained, speeding, possible speeding, stop not made, turn-ban violation/compliance) that jumps the video; summary score; signs not yet met.
- **No live mode.** Upload only. Live in-car prompts are a roadmap item for the deck, not something to build.

## 2. Hard rules (never break these)

1. **No faked results.** Never hard-code outputs for any video, never hand-edit report JSON, never show a metric that was not produced by a script in `eval/`. Sample lessons are produced by the real pipeline (`scripts/seed_samples.py`).
2. **Never invent official texts or legal values.** Sign meanings, default speed limits and how far a sign's effect extends must come from official road traffic rules, and every entry records its source document. Humans choose the source (the demo footage and sign numbering are Russian; the product targets Uzbekistan). Create the schema and placeholders marked `"verified": false` and `TODO`; humans fill them in. English working drafts are allowed but must be marked as drafts.
3. **Never tell the driver an action is permitted** ("you may turn left"). The system only warns and explains.
4. **Privacy:** no unblurred faces or licence plates in anything we publish or serve (annotated video, thumbnails, sample lessons, README images). Strip audio from all outputs.
5. **Licences:** every dataset, model weight and third-party code goes into `DATA_SOURCES.md` with its licence. Do not copy code from repositories without a licence (e.g. `Aillustrate/traffic-sign-detection` may be cited as related work only).
6. **No LLM calls in the pipeline.** All logic is our own deterministic code plus our detector.
7. **Never enable horizontal-flip augmentation** when training (`fliplr=0.0`): mirroring turns "no left turn" into "no right turn" and mirrors arrows and digits.
8. **Never commit** datasets, videos, model weights, uploads, `.env` or secrets. Keep them in the data directory (§4).

## 3. Working agreement

- Work **milestone by milestone** (§17). Each milestone is done on a branch `m<N>-<slug>`. When it is done, **stop** and post a review summary, then wait for the human to say "go":
  ```
  ## M<N> review
  Built: <bullets>
  Files: <list of files added/changed>
  How to test: <exact commands / URL steps>
  Results: <numbers from running it, if any>
  Known issues / shortcuts: <bullets>
  Next: <next milestone and what you need from humans>
  ```
- Humans merge to `main`; a push to `main` deploys (§16).
- Ask before: adding heavy dependencies, changing a data contract in `pipeline/types.py`, deleting files, or changing a decision in this file.
- Small commits, conventional messages (`feat:`, `fix:`, `chore:`, `test:`, `docs:`).
- Every module gets a top docstring explaining what it does and why, in plain English. Thresholds live in config, not in code.
- Two Claude Code sessions may run in parallel (e.g. pipeline vs. frontend/rules). The contracts in §9 and the sample fixture `tests/fixtures/report.sample.json` make that possible — keep them stable.

## 4. Environment

- **Machine:** Windows host with WSL2, Ubuntu 24.04. GPU: NVIDIA RTX 3070 Ti, 8 GB VRAM (about 1.4 GB already used by the host). CPU: AMD Ryzen 9 5900X, 24 threads. RAM: 32 GB. The same machine trains the model and serves the website.
- **No Docker.** Python via `uv`; the app runs as a `systemd` service; deploy through a GitHub Actions self-hosted runner; public HTTPS via Tailscale Funnel (§16).
- **GPU sharing:** while training runs (it uses most of the VRAM), run the app and tests with `SIGNCOACH_DEVICE=cpu` on short clips. The app must support `cpu` and `cuda:0` via that variable.
- **Data directory** (outside the repo, path in `.env` as `SIGNCOACH_DATA`, default `~/signcoach-data`). Shared data sits at the top; each app instance (`SIGNCOACH_INSTANCE` = `prod` for the deployed site, `dev` for development, `test` for tests) has its own state folder, so a dev server never takes production jobs:
  ```
  ~/signcoach-data/
    datasets/
      dognellaf -> ~/datasets/russian-road-signs            # images (hf download, see §14.2)
      dognellaf_labels -> ~/datasets/dognellaf-labels-git/repo/labels
      dognellaf_yolo/        # generated training set (symlinked images + cleaned labels + data*.yaml)
    models/        # weights (+ sha256 files)
    train_runs/  train_logs/
    instances/<instance>/
      uploads/       # chunk staging
      jobs/<job_id>/ # normalized video, intermediates
      media/<lesson_id>/  # annotated.mp4, thumbs/, report.json
      db/signcoach.sqlite3
  ```
- Keep datasets on the Linux filesystem, not under `/mnt/*` (slow). Disk is fine: the WSL virtual disk lives on `D:\WSL` (1.3 TB free) even though C: and E: are full.
- Installed: `ffmpeg` 6.1 (with zscale/tonemap, NVENC, CUVID), `uv`, `git`, `tmux`; Node.js 24 LTS and the GitHub CLI `gh` live in `~/.local` (no root needed). `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`. `sudo` needs a password, so root steps are scripted for the humans (`deploy/setup_host.sh`).

## 5. Tech stack

| Area | Choice |
|---|---|
| Python | 3.12, managed with `uv` |
| Detection & tracking | Ultralytics (YOLO11 / YOLO12), built-in ByteTrack via `model.track(..., persist=True, tracker="bytetrack.yaml")` |
| Deep learning | PyTorch with CUDA |
| Vision | OpenCV (`opencv-python-headless`), NumPy |
| Video | `ffmpeg` / `ffprobe` via `subprocess` (normalize, rotate, encode); render by piping raw frames to `ffmpeg` stdin rather than `cv2.VideoWriter` |
| GPS | `gpxpy` |
| Backend | FastAPI, Uvicorn, Pydantic v2, stdlib `sqlite3`, one background worker thread |
| Frontend | React + Vite + TypeScript + Tailwind, built to `web/dist` and served by FastAPI; HTML5 `<video>`; Leaflet only when a GPX track exists |
| Voice | English MP3s pre-generated offline with a local text-to-speech model (licence recorded in `DATA_SOURCES.md`; not an LLM, not called by the pipeline); browser `speechSynthesis` fallback |
| Tests | `pytest` |
| Faces | OpenCV Zoo YuNet face detector (verify licence) |
| Plates | Vehicles from a COCO YOLO11n model + a licence-plate detector with a verified licence; fallback: blur the lower-centre band of each vehicle box |

## 6. Architecture

```
 upload (chunked) ──> job queue (SQLite, 1 worker)
                          │
                          ▼
  ingest ──> frames @ analysis_fps ──┬──> sign detector + ByteTrack ──> track voting ──> relevance ──┐
  (ffprobe, normalize,               ├──> vehicle detector (COCO) ──> masks for speed, plate blur     │
   rotation, strip audio)            ├──> ego-motion: speed (road-plane BEV), yaw/turns, stops ───────┤
                                     └──> faces (YuNet) ──> blur                                       │
  optional GPX ──> speed/course, time alignment ───────────────────────────────────────────────────────┤
                                                                                                       ▼
                                          rules engine (rules.json + signs.json, deterministic) ──> lesson events
                                                                                                       │
                                     render annotated 720p video + thumbnails  <──────────────────────┤
                                                                                                       ▼
                                                                            report.json ──> report page
```

## 7. Repository layout

```
signcoach/
  CLAUDE.md  README.md  DATA_SOURCES.md  LICENSE (AGPL-3.0)  pyproject.toml  uv.lock  .env.example  .gitignore
  config/
    settings.yaml               # all thresholds and paths (see §8)
    class_maps/
      streetsignsense_y12s.yaml # model class name -> sign code (+ value)
      dognellaf_y11s.yaml
    rules/
      signs.json                # sign catalogue: names, meanings, voice keys (humans fill official texts)
      rules.json                # effects, scopes, defaults (humans verify)
    i18n/en.json                # UI strings (English only; keys keep texts out of code)
  app/
    main.py                     # FastAPI app, static files, startup (worker)
    api/uploads.py jobs.py lessons.py health.py
    worker.py                   # single FIFO worker thread
    db.py                       # sqlite schema + helpers
    settings.py                 # loads .env + settings.yaml
  pipeline/
    types.py                    # Pydantic data contracts (§9) — the shared language
    run.py                      # process_job(job) -> report.json + media
    ingest.py                   # ffprobe, limits, normalize, quality checks
    frames.py                   # frame iterator at analysis_fps with timestamps
    detect.py                   # sign detector + tracking wrapper, class mapping
    vehicles.py                 # COCO vehicle detector
    signs.py                    # track voting -> SignEvent
    relevance.py                # does this sign apply to us?
    egomotion.py                # BEV speed, yaw/turns, stops, quality
    gps.py                      # GPX parse, speed/course, alignment
    rules_engine.py             # pure, deterministic state machine
    privacy.py                  # face/plate blur
    render.py                   # annotated video, thumbnails
    report.py                   # report.json assembly, score, signs-not-met
  training/
    prepare_dognellaf.py  class_crops.py  train.py  smoke_test.sh  start_full_training.sh  compare_models.py
  eval/
    eval_detector.py  eval_events.py  eval_speed.py  eval_limit_timeline.py  benchmark.py  failure_gallery.py
    ground_truth/README.md      # CSV formats (§15)
    results/                    # generated JSON + markdown tables (committed), e.g. dataset_inventory.md
  scripts/
    download_models.py  dump_detections.py  seed_samples.py  calibrate_speed.py  generate_voice.py
  web/                          # Vite app
  assets/voice/en/<voice_key>.mp3   # generated by scripts/generate_voice.py
  docs/HOW_IT_WORKS.md          # one page per module, for the team's Q&A prep (M8)
  deploy/signcoach.service  deploy/sudoers.example  deploy/setup_host.sh  deploy/FUNNEL.md
  .github/workflows/deploy.yml
  tests/
    test_contracts.py  test_rules_engine.py  test_signs_voting.py  test_relevance.py  test_egomotion_synthetic.py  test_gps_align.py  test_api_smoke.py
    fixtures/report.sample.json  fixtures/make_report_sample.py  fixtures/rule_scenarios/*.json
```

## 8. Configuration (`config/settings.yaml`) — starting defaults

The table lists the main keys. The file also holds the relevance thresholds (`relevance.*`, incl. `conditional_plate_codes`), model files (`models.*`), privacy options (`privacy.*`), GPS alignment (`gps.*`), the job timeout (`job.timeout_s`), `turn_bans_enabled`, the voting baseline flag (`vote.single_frame_baseline`), the render debug flag and the cost inputs (`cost.*`).

| Key | Default | Note |
|---|---|---|
| `device` | from `SIGNCOACH_DEVICE`, else `cuda:0` | `cpu` while training |
| `limits.max_video_seconds` | 180 | longer → clear error |
| `limits.max_upload_bytes` | 1.5 GB | |
| `upload.chunk_bytes` | 8 MB | |
| `ingest.max_width` | 1920 | normalize keeps detail for small signs |
| `ingest.fps_cap` | 30 | |
| `analysis_fps` | 15 | frames analysed per second |
| `detect.backend` | `ultralytics` | `replay` for tests and offline iteration |
| `detect.model` | `streetsignsense_y12s` | switched at the M6 decision |
| `detect.imgsz` | 1280 | |
| `detect.conf` | 0.25 | low; voting filters |
| `vote.min_frames` | 3 | |
| `vote.min_conf` | 0.5 | mean confidence of the winning class |
| `vote.dedupe_window_s` | 2.0 | merge same code split by ID switches |
| `camera.height_m` | 1.3 | overridable per upload |
| `camera.hfov_deg` | 70 | overridable |
| `bev.x_range_m` | [-3, 3] | |
| `bev.z_range_m` | [6, 26] | |
| `bev.m_per_px` | 0.05 | |
| `speed.smooth_s` | 1.0 | |
| `speed.scale_k` | 1.0 | set by `scripts/calibrate_speed.py` |
| `speed.err_model` | `{a: 3.0, b: 0.10}` | km/h, err = a + b·v; replaced by calibration results |
| `stop.speed_kmh` | 2.0 | |
| `stop.min_duration_s` | 1.0 | |
| `stop.window_s` | [-2, 6] | around passing a stop sign |
| `speeding.tolerance_kmh` | 0 | humans may change |
| `speeding.min_duration_s` | 3 | |
| `turn.min_angle_deg` | 60 | |
| `turn.uturn_angle_deg` | 150 | |
| `turn.max_window_s` | 10 | |
| `ban_scope` | `{until_turn: true, max_distance_m: 300, max_time_s: 60}` | approximation of "until the next junction" (no map) |
| `score.weights` | `{violation: 10, possible: 3}` | score = max(0, 100 − Σ); label as indicative |
| `quality.low_light_luma` | 60 | mean luma below → warning |
| `quality.blur_laplacian_var` | 50 | below → warning |
| `output.height` | 720 | |
| `output.crf` | 23 | libx264 `-preset veryfast`, `+faststart` |

## 9. Data contracts (`pipeline/types.py`)

Pydantic models; all times are seconds from the start of the normalized video; speeds in km/h.

```python
class Detection(BaseModel):          # one box in one analysed frame
    frame_idx: int; t: float; track_id: int | None
    model_class: str; sign_code: str | None; value: int | None  # e.g. 3.24 + 60
    conf: float; bbox: tuple[float, float, float, float]        # x1,y1,x2,y2 in normalized-video pixels

class SignEvent(BaseModel):          # one physical sign, after voting
    id: str; sign_code: str; value: int | None
    t_first: float; t_last: float; t_pass: float   # t_pass = last time seen (sign leaves view)
    conf: float; n_frames: int
    side: Literal["right", "left", "overhead"]
    relevant: bool; relevance_reason: str
    conditional: bool                # supplementary plate detected under it
    best_frame_idx: int; best_bbox: tuple[float, float, float, float]

class MotionSample(BaseModel):
    t: float; speed_kmh: float | None; speed_err_kmh: float | None
    source: Literal["gps", "video", "none"]; quality: float      # 0..1
    yaw_rate_dps: float | None; heading_deg: float | None; stopped: bool

class TurnEvent(BaseModel):
    t_start: float; t_end: float; direction: Literal["left", "right", "uturn"]; angle_deg: float

class RuleState(BaseModel):          # snapshot whenever the state changes
    t: float; speed_limit_kmh: int | None; limit_source: Literal["sign", "default", "unknown"]
    limit_sign_event_id: str | None; active_bans: list[str]; mandatory: list[str]

class LessonEvent(BaseModel):
    id: str; t: float
    type: Literal["sign_explained", "speeding", "possible_speeding", "stop_not_made", "stop_ok",
                  "turn_ban_violation", "turn_ban_complied", "no_entry_possible", "info"]
    severity: Literal["info", "violation", "possible", "complied"]
    sign_code: str | None; value: int | None
    title_key: str; detail: dict      # i18n key + parameters, e.g. {"min_speed": 8}
    clip: tuple[float, float]          # suggested playback window
    voice_key: str | None; thumbnail: str | None
    sign_event_id: str | None

class Report(BaseModel):
    lesson: dict                       # id, student, label, created_at, duration_s, is_sample
    media: dict                        # annotated_video_url, poster_url
    speed_source: Literal["gps", "video", "none"]; calibration: dict
    summary: dict                      # score, signs_passed, violations, possible, complied
    events: list[LessonEvent]; timeline: list[RuleState]
    motion: list[MotionSample]         # downsampled to 2 Hz for the UI overlay/graph
    signs: list[SignEvent]             # every sign after voting, relevant or not ("every sign the car passed")
    signs_not_met: list[str]           # catalogue codes never seen in this student's lessons
    quality_warnings: list[str]        # i18n keys
    processing: dict                   # model name+version, device, seconds per stage, fps, peak VRAM, git sha
```

All models forbid unknown fields. `tests/fixtures/report.sample.json` is generated by `tests/fixtures/make_report_sample.py` (synthetic, never served as a lesson), validates against `Report` and covers every event type and severity; `tests/test_contracts.py` checks it.

## 10. Pipeline spec

### 10.1 Ingest (`ingest.py`)
- `ffprobe` → duration, fps, width/height, rotation, codec, `creation_time`. Reject over limits with an i18n error key.
- Normalize with ffmpeg: decode anything (HEVC/MOV from iPhones, MP4, MKV, AVI), apply rotation (ffmpeg autorotates by default — verify with a portrait iPhone clip), drop audio, H.264, width ≤ 1920, constant frame rate, `yuv420p`, `+faststart`.
  - **fps = min(source fps, `ingest.fps_cap`)** — never raise the frame rate: duplicated frames look like "car not moving" to the speed estimate.
  - **HDR** (ffprobe `color_transfer` = `arib-std-b67` or `smpte2084`, e.g. iPhone HLG/Dolby Vision): tone-map to SDR BT.709 with `zscale`+`tonemap`, otherwise colours wash out.
  Example (SDR, 60 fps source): `ffmpeg -y -i in -map 0:v:0 -an -vf "scale='min(1920,iw)':-2,fps=30" -fps_mode cfr -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart normalized.mp4`
- Quality checks on sampled frames → warnings: portrait orientation, low light (mean luma), blur (Laplacian variance), very short clip (< 10 s).

### 10.2 Frames (`frames.py`)
- Iterate the normalized video at `analysis_fps` with exact timestamps. **Two passes:** the analysis pass feeds detection, vehicles, ego-motion and faces (boxes are stored per frame); the render pass decodes again (cheap) once the rules engine has run, and reuses the stored face/vehicle boxes. Rendering cannot share the first pass: it needs the rules results, and buffering 3 minutes of 720p frames would take ~7.5 GB of RAM.

### 10.3 Sign detection and tracking (`detect.py`) — behind an adapter
The final model does not exist yet, so nothing outside `detect.py` may depend on a specific model.
- `SignDetector` protocol: `reset()` (new video) and `detect(frame, t, frame_idx) -> list[Detection]` (with track ids).
- `ReplayDetector`: reads detections from a JSON file. Used by tests, by the frontend fixtures, and to develop every later stage without a GPU.
- `UltralyticsDetector(model_name)`: loads weights and class map from config and runs `model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=detect.imgsz, conf=detect.conf, device=...)` frame by frame. Used with the placeholder now and the fine-tuned model later.
- Selected by `detect.backend` (`ultralytics` | `replay`) and `detect.model`.
- Map `model.names[class_id]` → `sign_code` (+ `value` for speed limits) using `config/class_maps/<model>.yaml`. Classes not in the map are kept with `sign_code=None` (shown only in debug). Swapping the model must only require a config change and a class map.
- `scripts/dump_detections.py` runs a detector over a video and saves the JSON that `ReplayDetector` reads, so later stages can be iterated without re-running detection.

### 10.4 Track voting (`signs.py`)
- Per track: winning class = argmax of summed confidence; keep if `n_frames ≥ vote.min_frames` and mean winning conf ≥ `vote.min_conf`.
- `t_pass` = last time seen. Merge tracks with the same code whose times overlap or are within `vote.dedupe_window_s` and whose boxes are near (ByteTrack ID switches).
- Keep a single-frame baseline path (no voting) behind a flag so `eval/` can measure what voting adds.

### 10.5 Relevance (`relevance.py`)
Heuristics, each recorded in `relevance_reason`:
- **Side:** box centre right of the frame centre (with margin) → `right`; above the horizon and wide/central → `overhead`; else `left`. Right and overhead are relevant; left is irrelevant unless overhead.
- **Facing:** for round/square signs, box width/height < 0.6 → likely facing a side road → irrelevant.
- **Approaching:** box area must grow over the track; shrinking or static → irrelevant.
- **Conditional:** if the model has supplementary-plate classes (8.x), a plate box directly under the sign (within 1.5× sign width) → `conditional=true`; the rules engine then downgrades violations to `possible`.

### 10.6 Ego-motion (`egomotion.py`)
**Speed from video (road-plane method):**
1. Intrinsics: `f = (W/2) / tan(hfov/2)`, principal point = image centre.
2. Horizon row `y_h`: estimate from lane-line vanishing point over the first seconds (Hough lines, robust median); fallback `0.45·H`; allow a per-upload override.
3. Bird's-eye view: for a ground point at image row `y > y_h`, distance `Z ≈ h·f/(y − y_h)`, lateral `X ≈ (x − cx)·Z/f`. Build a BEV grid over `bev.x_range_m × bev.z_range_m` at `bev.m_per_px` with `cv2.remap`.
4. Mask BEV pixels covered by vehicle boxes (bottom edges projected) from `vehicles.py`.
5. Forward shift between consecutive BEV frames with `cv2.phaseCorrelate` (Hanning window); shift (px) × `m_per_px` × `analysis_fps` × `speed.scale_k` × 3.6 = km/h. The phase-correlation response and texture energy give `quality`.
6. Robust smoothing over `speed.smooth_s` (median + simple Kalman). Low-quality samples → `speed_kmh=None`.
7. `speed_err_kmh` from `speed.err_model` (calibrated).
- Speed is proportional to camera height: a per-upload `height_m` scales the result linearly.
**Yaw and turns:** sparse optical flow (`goodFeaturesToTrack` + LK) above the horizon (distant scenery); median horizontal flow → yaw rate `atan(dx/f)·analysis_fps`; integrate to heading. A turn is a heading change ≥ `turn.min_angle_deg` within `turn.max_window_s`; U-turn ≥ `turn.uturn_angle_deg`. With GPX, use GPS course instead.
**Stops:** speed < `stop.speed_kmh` for ≥ `stop.min_duration_s` (BEV shift near zero is the most reliable output of this method).

### 10.7 GPS (`gps.py`) — optional input
- Parse GPX (`gpxpy`), compute speed (haversine/Δt, smoothed) and course.
- Align to video time: initial offset from `creation_time` vs. GPX start; refine by cross-correlating video speed with GPS speed over ±60 s; allow a manual offset in the upload form.
- When GPS is present it is the speed source and the course source; the video estimate is still computed and stored (used by `eval_speed.py`).
- Map/junction lookup (OpenStreetMap) is **out of scope** unless everything else is done.

### 10.8 Rules engine (`rules_engine.py`) — see §11
Pure function: `run(sign_events, motion, turns, rules, catalogue, settings) -> (events, timeline)`. No I/O. Fully covered by `pytest` scenarios.

### 10.9 Privacy (`privacy.py`)
- Faces: YuNet on every rendered frame, plus the head area of every COCO `person` box from the vehicle detector (YuNet misses small and side-on faces). Plates: plate detector on vehicle crops, or the fallback band. Strong pixelation/Gaussian blur. Applied to every output frame and thumbnail. Unblurred frames never leave `jobs/<job_id>/`; the raw upload and the normalized (unblurred) video are deleted after processing.

### 10.10 Render (`render.py`) and report (`report.py`)
- Output video at `output.height` (720p), `analysis_fps`, H.264 libx264 via an ffmpeg stdin pipe. Overlay: boxes and labels for relevant signs; HUD (limit in force, speed with "est." when from video, red when over); subtitle line for voice prompts at event times. A debug flag also draws irrelevant signs in grey.
- Thumbnails: one blurred crop per lesson event.
- `report.json` per §9, including processing stats (seconds per stage, frames per second, peak VRAM, model version, git SHA).

## 11. Rules and sign catalogue

### 11.1 Required signs (semantic keys; codes are the Russian GOST numbers of the training data, checked against `classes.txt` on Saturday 22:25)

Dataset status (`eval/results/dataset_inventory.md`): **missing** 3.18.2 (no left turn), 3.19 (no U-turn), 5.23.x/5.24.x (settlement start/end); **effectively missing** 3.25 (5 examples from 1 video); **thin** 3.31 (133 examples), 4.1.3 (6 videos), 4.1.6 (104 examples). Rules that depend on a missing sign are `"enabled": false`; without settlement signs `assume_in_settlement` stays true. Real conditional plates (8.5.x time, 8.4.x vehicle type) have 0–2 examples, so conditional handling is effectively inactive with the fine-tuned model.

| Key | Code | Use |
|---|---|---|
| speed_limit_<v> | 3.24 (+ value) | set limit |
| end_speed_limit | 3.25 | end limit |
| end_all_restrictions | 3.31 | end limits and bans |
| no_entry | 3.1 | possible violation if passed and relevant |
| no_right_turn / no_left_turn | 3.18.1 / 3.18.2 | turn ban |
| no_u_turn | 3.19 | turn ban |
| no_overtaking / end_no_overtaking | 3.20 / 3.21 | explain only |
| no_stopping / no_parking | 3.27 / 3.28 | explain only |
| stop | 2.5 | must stop |
| give_way | 2.4 | explain |
| main_road / end_main_road | 2.1 / 2.2 | explain |
| mandatory_straight / right / left / straight_or_right / straight_or_left / right_or_left | 4.1.1 … 4.1.6 | explain; turn against → violation |
| roundabout | 4.3 | explain |
| pedestrian_crossing | 5.19.1 / 5.19.2 | explain |
| pedestrian_crossing_ahead | 1.22 | explain |
| children | 1.23 | explain |
| speed_bump / speed_bump_ahead | 5.20 / 1.17 | explain |
| settlement_start / settlement_end | 5.23.x / 5.24.x | switch default limit |

If a required class is missing from the detector, report it in the M6 review and mark the related rule `"enabled": false`.

### 11.2 `signs.json` (catalogue entry)
```json
{
  "code": "3.24", "key": "speed_limit", "category": "prohibitory", "shape": "round",
  "name": "Maximum speed limit (draft)",
  "meaning": "Draft: do not exceed the speed shown. TODO: official text",
  "voice_key": "speed_limit_{value}",
  "detectable": true,
  "source": "TODO: official rules document + annex/section",
  "verified": false
}
```
`shape` feeds the facing heuristic (§10.5); `detectable` = the current detector's class map contains the code (signs that cannot be detected are left out of "signs not met yet").

### 11.3 `rules.json`
```json
{
  "defaults": {
    "speed_limit_kmh": {"in_settlement": null, "outside_settlement": null},
    "assume_in_settlement": true,
    "_todo": "Humans fill default limits from the official rules (record the source). While null, limit_source='unknown' and no speeding is reported without a sign."
  },
  "signs": {
    "3.24": {"effects": [{"type": "set_speed_limit", "value_from": "detection"}],
             "scope": {"ends_on": ["sign:3.25", "sign:3.31", "sign:5.24", "turn"], "max_distance_m": null, "_todo": "verify scope in the official rules"}},
    "3.18.2": {"effects": [{"type": "add_ban", "ban": "left"}], "scope": {"ends_on": ["turn"], "use_settings_ban_scope": true}},
    "2.5": {"effects": [{"type": "require_stop"}]}
  }
}
```

### 11.4 Engine semantics
- Process a time-ordered stream of relevant `SignEvent`s (at `t_pass`), `MotionSample`s and `TurnEvent`s. Maintain `RuleState`; append a snapshot at every change.
- **Severity is the single source of truth** for colour and score; `type` says what happened.
- **sign_explained:** the first time each sign **code + value** appears in a lesson (voice prompt + explanation) — so a new 40 limit after a 60 limit gets its own prompt.
- **speeding:** with a known limit, if `speed − err > limit + tolerance` for ≥ `speeding.min_duration_s` → `speeding` / `violation`; if `speed > limit + tolerance` but `speed − err ≤ limit + tolerance` for that duration → `possible_speeding` / `possible`. Merge contiguous seconds into one event with min/max speed and duration. If speed is unavailable → no speeding events, and a `speed_not_checked` warning.
- **stop:** after a relevant stop sign, if the minimum speed in `stop.window_s` around `t_pass` > `stop.speed_kmh` → `stop_not_made` (`violation`) with the minimum speed; else `stop_ok` (`complied`). No valid speed sample in the window → `info` "stop not checked" — never a violation.
- **turn bans:** while a ban is active, a matching `TurnEvent` → `turn_ban_violation` (`detail.kind = "ban"`); a ban that expires without a matching turn → `turn_ban_complied`. Mandatory-direction signs (4.1.x) work the same way: a turn outside the allowed directions → `turn_ban_violation` with `detail.kind = "mandatory"`. Scope per `ban_scope` (approximation, documented in the README).
- **no entry:** a relevant no-entry sign passed → `no_entry_possible` (`possible`), never a hard violation.
- **conditional signs:** any violation from a sign with `conditional=true` is downgraded: speeding becomes `possible_speeding`; other types keep their type with severity `possible` and `detail.reason = "conditional"`.
- **score:** `max(0, 100 − Σ weights)`; shown as "indicative".
- Write at least 15 JSON scenarios in `tests/fixtures/rule_scenarios/` (input streams + expected events), covering every event type, limit replacement, end signs, unknown speed, conditional downgrade and ban expiry.

## 12. Backend API and worker

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/uploads/init` | `{filename, size_bytes}` → `{upload_id, chunk_bytes}` |
| PUT | `/api/uploads/{id}/chunk?index=n` | raw bytes of chunk `n` |
| GET | `/api/uploads/{id}` | `{received: [chunk indices], size_bytes, chunk_bytes}` — lets the browser resume an interrupted upload |
| POST | `/api/uploads/{id}/gpx` | optional small GPX file |
| POST | `/api/uploads/{id}/complete` | `{student, lesson_label, camera_height_m?, hfov_deg?, gps_offset_s?, consent: true}` → `{job_id}` |
| GET | `/api/jobs/{id}` | `{status: queued|processing|done|failed, stage, progress 0..1, queue_position, error_key, lesson_id}` |
| GET | `/api/lessons?ids=a,b` | sample lessons + the lessons whose ids the browser asks for (the ones it uploaded); never a public list of everyone's uploads |
| GET | `/api/lessons/{id}/report` | `report.json` |
| GET | `/media/{lesson_id}/...` | annotated video, poster, thumbnails — **must support HTTP Range** (verify seeking works; add a range handler if needed) |
| GET | `/api/students/{student}/progress` | stretch |
| GET | `/health` | `{ok, git_sha, model, device, queue_length}` — the git SHA is our deployment evidence |

- SQLite tables: `jobs(id, status, stage, progress, error_key, params_json, upload_path, lesson_id, created_at, started_at, finished_at)`, `lessons(id, student, label, is_sample, duration_s, report_path, video_path, created_at)`. Lesson and job ids are long random tokens (unguessable links).
- One worker thread, FIFO. Models load once and stay in memory. On startup, jobs stuck in `processing` go back to `queued`. Per-job timeout 15 min (`job.timeout_s`), enforced cooperatively: stages check the deadline between frames (a Python thread cannot be killed). Every failure maps to a user-friendly `error_key` (i18n), never a raw stack trace. At most `limits.max_queued_jobs` waiting jobs.
- Uploads require `consent: true` ("I may upload this video; faces and plates will be blurred").
- Target: processing time ≤ video duration for a 1080p clip on the RTX 3070 Ti. Measure in `benchmark.py`.

## 13. Frontend (`web/`)

Pages:
1. **Upload:** drag-and-drop video, optional GPX, student name, lesson label, optional camera height, consent checkbox; chunked upload with progress; then a job view (stage name, progress bar, queue position) that redirects to the report when done. Clear messages for every `error_key`.
2. **Report** (layout agreed with the team):
   - Header: lesson name, student, duration, route label, speed source. English only — no language toggle.
   - Annotated video player (HUD is burned into the video).
   - Timeline under the player with event markers coloured by severity (grey info, red violation, amber possible, green complied); click → seek.
   - Voice prompts: when playback crosses an event with a `voice_key`, play `assets/voice/en/<voice_key>.mp3` (TTS-generated; fallback: `speechSynthesis` with the English text). A toggle turns voice on/off. iPhones block audio that is not started by a tap, so the first tap on Play unlocks one shared audio element that all prompts reuse.
   - Summary cards: score (labelled indicative), signs passed, violations, possible.
   - Event list: time, sign icon/thumbnail, title, detail, severity pill; click → seek to `clip[0]`.
   - "Signs not met yet" chips.
   - Quality-warning banner when present.
   - "How this was computed" section: speed source, calibration, model version, processing time.
3. **Lessons:** sample lessons plus the lessons uploaded from this browser (ids remembered in `localStorage`).
4. **Student progress:** stretch.
- Mobile-friendly (judges will open it on phones). Sentence case, no jargon. All strings through `config/i18n/en.json`.
- Build the report page against `tests/fixtures/report.sample.json` first.

## 14. Models and training

### 14.1 Placeholder model (interim — used until the fine-tuned model is ready)
- `AlessandroFerrante/StreetSignSenseY12s` on Hugging Face, file `streetsignsense-yolo12s.pt`; 63 classes of European/International signs; licence CC BY 4.0 / MIT (verify the LICENSE file and record it). Trained at 640 px, so it may miss small distant signs.
- `scripts/download_models.py` downloads it to `models/`, writes a sha256, prints `model.names`; build `config/class_maps/streetsignsense_y12s.yaml` from those names (map only signs that match our catalogue).

### 14.2 Fine-tune (runs in the background; never blocks pipeline work)
- **Data:** `Dognellaf/russian-road-signs` (Hugging Face dataset, CC BY 4.0): 43,851 video frames, 89 classes (signs by Russian GOST number + traffic-light states), YOLO format.
  - **Images** are in `~/datasets/russian-road-signs` (symlinked as `{DATA}/datasets/dognellaf`), downloaded by the humans with
    `cd ~/datasets && hf download Dognellaf/russian-road-signs --repo-type dataset --local-dir russian-road-signs --exclude "unlabeled/*" --max-workers 32`
    — re-run exactly that command to resume. All 43,851 images in `images/part_00…04` are complete. The repo also contains 10,000 duplicate images and 10,000 duplicate labels directly in `images/` and `labels/`; they are ignored. That download was still fetching labels at ~350 files/min on Saturday night; it is no longer needed.
  - **Labels** came from a shallow git clone (labels are plain git files, not LFS): `~/datasets/dognellaf-labels-git/repo/labels` (symlinked as `{DATA}/datasets/dognellaf_labels`).
- **T1 – completeness (done 22:25):** 38,918 complete image+label pairs; 4,933 images have no label file and are not used; 0 labels without images; 0 duplicate names; 195 boxes clipped to the image.
- **T2 – inventory (done):** `training/prepare_dognellaf.py` → `eval/results/dataset_inventory.{md,json}` (730 source videos). Speed-limit values verified from crops (`training/class_crops.py`): 3.24.1…3.24.11 = 10…110 km/h. Class map: `config/class_maps/dognellaf_y11s.yaml`. Missing/thin signs: see §11.1.
- **T3 – split (done):** frames grouped by **source video** (video id derived from file names: PXL/VID recording ids without segment suffix, dashcam clips grouped per drive hour, `asmr-*` segments in blocks of 25); 119 of 730 videos held out (3,027 val frames, 15.1%), chosen so every class's val share is close to 15%. Every 2nd frame kept, plus every frame with a rare class (< 300 instances). Train 17,066 frames. Output `{DATA}/datasets/dognellaf_yolo/` with `data.yaml` (full val), `data_train.yaml` (1,000-frame val subset used during training to save time) and `data_smoke.yaml`.
- **T4 – smoke test (passed 22:28):** `training/smoke_test.sh` — YOLO11s, imgsz 1024, 1 epoch on 5%: no errors, mAP50 0.003 > 0, 4.1 GB VRAM at batch 6.
- **T5 – full run (running since 22:29):** `training/start_full_training.sh 40` → `YOLO11s.train(imgsz=1024, epochs=40, batch=6, amp=True, cos_lr=True, close_mosaic=3, fliplr=0.0, degrees=0.0, save_period=1, workers=8, patience=10)` in tmux session `signcoach-train`; ~8.5 it/s ≈ 6.5 min/epoch incl. validation → ends ≈ 02:50. Logs in `{DATA}/train_logs/`, run in `{DATA}/train_runs/dognellaf_y11s_1024_*`. The app runs with `SIGNCOACH_DEVICE=cpu` meanwhile. Report progress (epoch, mAP, ETA) at each review.
- **T6 – compare (M6, after training):** `training/compare_models.py` runs both models (placeholder at 1280; fine-tuned at 1024 and 1280) on the **full validation split** (held-out videos) and reports, for the §11.1 classes both models can output: per-class precision, recall and AP50 after mapping to catalogue codes, plus inference speed (ms/frame) on the RTX 3070 Ti; plus event-level results via `eval_events.py` on the team's demo clips when labelled. Output: `eval/results/model_comparison.md`. The humans choose; switching = set `detect.model` and its class map, then re-run `seed_samples.py`.
- Record final weights with sha256 in `models/` and in `DATA_SOURCES.md`; keep weights out of git (offer them as a GitHub Release asset or a Hugging Face upload if the humans want).

### 14.3 Other models
- COCO `yolo11n.pt` for vehicles (Ultralytics, AGPL-3.0).
- YuNet face detector (OpenCV Zoo; verify licence).
- Licence-plate detector: search Hugging Face for a YOLO plate detector with a clear, compatible licence; if none is clear within 20 minutes, use the fallback band and say so in the README.

## 15. Evaluation (`eval/`) — every number in the deck comes from here

Ground truth from the team's own clips — the demo dashcam recordings for signs and limits, and a drive with a GPX track for speed (humans create these; formats documented in `eval/ground_truth/README.md`):
- `signs_<route>.csv`: `t_seconds, sign_code, value, side, relevant(y/n), notes` — one row per physical sign passed.
- `limit_<route>.csv`: `t_start, t_end, limit_kmh` — the limit in force over time.
- `stops_<route>.csv`: `t_seconds, full_stop(y/n)`.
- `gps_<route>.gpx` recorded at the same time as the video.

Scripts (each writes `eval/results/<name>.json` and a markdown table):
- `eval_detector.py`: Ultralytics val mAP on the held-out Dognellaf videos.
- `eval_events.py`: event-level precision/recall (match = same code within ±1.5 s), relevant signs only; with and without voting (ablation); day vs night.
- `eval_speed.py`: video speed vs GPS — MAE, p90 absolute error, % of time with a valid estimate; day vs night; separately for the calibration and test routes.
- `scripts/calibrate_speed.py`: fit `speed.scale_k` (median of v_gps/v_video for v ≥ 10 km/h) and the error model `a + b·v` on the **calibration route only**.
- `eval_limit_timeline.py`: % of driving time with the correct limit in force.
- `benchmark.py`: processing seconds per minute of video, per stage; frames per second; peak VRAM; GPU-seconds per minute. Cost per lesson-hour = formula with inputs in config (`electricity_price_per_kwh`, `gpu_power_w`, or `cloud_gpu_price_per_hour`) — humans fill the prices.
- `failure_gallery.py`: exports the worst misses and false positives (blurred) as images for the deck.

## 16. Deployment

- **Service:** `deploy/signcoach.service` (user `akmal`): runs `/home/akmal/signcoach-deploy/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000` directly (not `uv run`, which could try to sync or download packages on every restart), with `EnvironmentFile=/home/akmal/signcoach-deploy/.env` (`SIGNCOACH_INSTANCE=prod`), `Restart=always`.
- **Root steps** are in `deploy/setup_host.sh`, run by a human with `sudo bash deploy/setup_host.sh`: installs Tailscale, the sudoers rule, the service and the runner service. Step-by-step guide: `deploy/FUNNEL.md`.
- **Deploy directory** is a separate git clone (`~/signcoach-deploy`), not the dev checkout.
- **Workflow** `.github/workflows/deploy.yml`: triggers on push to `main` and `workflow_dispatch` (input: `ref`, for redeploying the submitted tag). Runs on `[self-hosted, signcoach]`. Steps: fetch + `git checkout --force <sha>` in the deploy dir → `uv sync --frozen` → `npm ci && npm run build` in `web/` → `uv run pytest -q tests -m "not slow"` → `sudo systemctl restart signcoach` (allowed by a sudoers rule for exactly that command; see `deploy/sudoers.example`) → poll `/health` until it returns the new git SHA, else fail.
- **Public URL:** Tailscale Funnel to port 8000: `sudo tailscale funnel --bg 8000` (Funnel enabled for the tailnet in the admin console; machine named `signcoach` so the URL is `https://signcoach.<tailnet>.ts.net`). It can take ~10 minutes the first time. Document steps in `deploy/FUNNEL.md`; test from a phone on mobile data.
- **Runner security:** the repo is public and the runner executes code on this machine. The workflow triggers only on push to `main` and `workflow_dispatch`, never on pull requests; repo setting *Fork pull request workflows → require approval for all outside collaborators*; no fork PRs are merged during the event.
- **Uptime:** the site is monitored after the deadline. Windows must not sleep or install updates, and WSL can stop when no WSL terminal is open — keep one open (see `deploy/FUNNEL.md`).
- **Freeze:** at the deadline, tag `final-submission`; after that, only `workflow_dispatch` with that tag.

## 17. Milestones (re-planned Saturday 22:40; deadline Sunday 13:00; targets are guides)

| # | Milestone | Target | Acceptance |
|---|---|---|---|
| M0 | Skeleton + deploy | 23:15 (code done 22:45; deploy needs the humans' `gh` login + `sudo bash deploy/setup_host.sh` + Tailscale login) | Repo layout, `uv` project, `settings.yaml`, `types.py`, `report.sample.json`, FastAPI `/health` + static placeholder page, service file, workflow, Funnel live. A push to `main` redeploys and `/health` shows the new SHA; page opens on a phone over mobile data. |
| T1–T4 | Completeness, inventory, split, smoke test | **done 22:28** | Completeness verified; class inventory reported (core signs present/missing); split by video; smoke test passed (§14.2). |
| T5 | Full training (background) | 22:29 → ≈ 02:50 | Running detached; checkpoint every epoch; progress in every review; app on `cpu` meanwhile. |
| M1 | Pipeline end to end, model-independent | 01:30 | Chunked upload → job → ingest → frames → detector adapter (replay + placeholder) → tracking, voting, relevance → vehicle detection + face/plate blur → render → `report.json` → bare report page. A clip with no signs returns a valid report with a "no signs" note. Works on `cpu` and `cuda:0`. |
| M2 | Rules engine + catalogue | 01:30 if a parallel session runs, else after M1 | `signs.json`/`rules.json` with all §11.1 entries (texts TODO), engine, ≥ 15 passing scenarios. Pure Python — a parallel session can start right after M0 is merged. |
| M4 | Report UI | 03:00 | §13 complete against the fixture, then wired to real reports; voice sync (TTS MP3s); English; works on a phone. |
| M6 | Model comparison + swap | after T5 (≈ 03:00–03:30) | `eval/results/model_comparison.md` posted; humans pick; class map switched; samples re-seeded. |
| M3 | Ego-motion + GPS | 05:00 (calibration needs a daylight drive → after ~06:30) | BEV speed, yaw/turns, stops, GPX alignment; synthetic BEV-shift test passes; calibration script runs on the team's calibration route. Model-independent — start as soon as M1 is merged. |
| M5 | Privacy hardening + quality warnings | 07:00 | Blur verified on every output and thumbnail (spot-check gallery); audio stripped; quality warnings shown; temp frames deleted. |
| M7 | Evaluation | 10:00 | All §15 scripts run on the team's ground truth; results committed; failure gallery exported. |
| M8 | Hardening + docs | 12:30 | §18 edge cases pass; `seed_samples.py` produces 2–3 sample lessons (day, night, with violations); README + DATA_SOURCES complete; `docs/HOW_IT_WORKS.md` (one page per module, for the team's Q&A prep); tag `final-submission` before 13:00. |

If time runs short, cut in this order: student progress page → Leaflet map → conditional-plate handling → turn bans (keep as explanations only, flag `turn_bans_enabled: false`). Never cut privacy, evaluation or the deploy pipeline.

## 18. Edge cases (the jury's clip)

- iPhone HEVC `.mov`, portrait video, 60 fps, 4K, very short (< 10 s), over 3 minutes (clear rejection), no signs at all, night, rain/dirty windshield, camera mounted at an unknown height (speed marked "estimate, uncalibrated"), no GPX, GPX that doesn't overlap the video (ignore with warning), upload interrupted (resume chunks), same file uploaded twice, a non-video file.
- Every case ends in either a valid report with honest warnings or a clear error message — never a crash or a spinner that never ends.

## 19. What you need from the humans (ask for it at the right milestone)

- M0: `gh auth login` (as the account whose SSH key is on this machine, `aki4455`) so Claude can create the repo, open PRs and register the runner; then `sudo bash deploy/setup_host.sh`, `sudo tailscale up --hostname=signcoach`, Funnel enabled, `sudo tailscale funnel --bg 8000`.
- M2: official texts for `signs.json`, default speed limits and scope rules for `rules.json`, and which rules document is the source (until then: TODO placeholders).
- M3: calibration route video + GPX (daylight), measured camera height and the phone model (for field of view).
- M4: nothing — voice prompts are generated with a TTS model (Claude provides the list of keys and texts for review).
- M6: the decision on which model to use.
- M7: ground-truth CSVs for the demo/test clips (day and night).
- M8: approval of sample lessons for the public site, and confirmation that the demo recordings may be published (blurred).

## 20. README.md and DATA_SOURCES.md

- **README:** what it is (one paragraph), architecture diagram, how to run locally (system packages, `uv sync`, `npm ci && npm run build`, `.env`, `uv run uvicorn ...`), how to download models, train, evaluate and seed samples, deployment, limitations and known failure cases (no map, ban scope approximation, speed estimate error, sign classes missing), privacy approach, a declaration that the code was written with Claude Code and reviewed by the team, and the elimination-stage code reused (if any — humans fill).
- **DATA_SOURCES.md** table: name, link, what we use it for, licence, attribution text. At least: Dognellaf/russian-road-signs (CC BY 4.0), StreetSignSense YOLO12s weights (verify), Ultralytics + YOLO11 COCO weights (AGPL-3.0), YuNet (verify), plate detector (verify) or "not used", OpenCV, FastAPI and other libraries, the demo/team recordings (blurred, no audio; source and permission noted) and the text-to-speech model used for the voice prompts. Related work, cited only: Aillustrate/traffic-sign-detection (no licence; no code used).

## 21. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Fine-tune underperforms or fails | Keep the placeholder model (class map switch only). |
| Dataset download stalls | Train on complete pairs only; top up scarce core signs with targeted downloads (§14.2). |
| Training still running near the demo | Use the best checkpoint so far (`best.pt` is updated every epoch); stop training before seeding samples and benchmarking. |
| GPU out of memory while training | App and tests on `SIGNCOACH_DEVICE=cpu`; training `batch=4`. |
| Video speed poor on a clip | `quality` gate → `speed_kmh=None`, "speed not checked" warning; GPX path when available. |
| Turn detection unreliable | `turn_bans_enabled: false` → bans shown as explanations only. |
| Plate detector licence unclear | Fallback blur band; documented. |
| Processing slower than real time | Lower `detect.imgsz` to 1024, `analysis_fps` to 10; report the change. |
| Funnel or network down at the defence | The team's recorded backup video (humans). |
| Site goes down after the deadline (Windows sleep/updates, WSL stops) | Sleep off, updates paused, a WSL terminal kept open; tested with all other terminals closed (§16). |
| A sign the rules need is not in the training data (3.18.2, 3.19, 5.23/5.24, ~3.25) | Rule `"enabled": false`; listed as a known limitation in the README and the deck. |
| Fork pull request runs code on the self-hosted runner | No `pull_request` triggers; approval required for outside contributors; no fork PRs merged. |
| Commercial use of Ultralytics (AGPL-3.0) | Repo is AGPL-3.0; for a commercial launch: Ultralytics Enterprise licence or a permissively licensed detector (business slide). |
