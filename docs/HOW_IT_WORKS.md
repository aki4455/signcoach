# How SignCoach works — the team's guide

This guide is for every team member, including those who did not write the code. Read sections 1 and 2
for the big picture. Read section 3 for any part the jury may ask about. Before the defence, read the
numbers cheat sheet (section 4) and the questions (section 5).

Every number here comes from a file in the repo, and its source is named. Settings are the values in
`config/settings.yaml` today. File paths are relative to the repository root.

---

## 1. What SignCoach is

**The problem.** In a driving lesson the instructor watches the road, the student and the traffic at the
same time. Afterwards there is no record of which signs the car passed, what they meant, or where the
student broke a rule.

**The product.** A driving-school instructor uploads the dashcam video of a lesson. SignCoach returns a
report:
- every road sign the car passed, what it means (the official text of the traffic rules) and whether it
  applied to our car;
- the rules in force over time (speed limit, turn bans, mandatory directions);
- events: possible speeding, a stop not made, a turn against a sign, a possible no-entry;
- the video with sign boxes and a speed/limit display drawn in, faces and number plates pixelated;
- voice prompts that play at the moments an in-car system would have spoken.

**Users.** Driving schools in Uzbekistan: instructors and students. The video file is the only input.
There is no phone app, no live mode and no GPS logger (CLAUDE.md "Start here").

**What is ours and what is third-party.**

| Ours (written or trained for this project) | Third-party (used as-is, licences in `DATA_SOURCES.md`) |
|---|---|
| The sign detector's weights: YOLO11s fine-tuned by us on the Dognellaf dataset (`training/`) | Ultralytics library and the YOLO11s COCO starting weights (AGPL-3.0) |
| Dataset preparation: completeness check, inventory, split by source video (`training/prepare_dognellaf.py`) | Dognellaf "russian-road-signs" dataset (CC BY 4.0) |
| The class map, with speed-limit values checked on example crops (`config/class_maps/dognellaf_y11s.yaml`, `training/class_crops.py`) | YOLO11n COCO for vehicles and people (AGPL-3.0) |
| Our sign tracker (`pipeline/tracking.py`) | YuNet face detector, OpenCV Zoo (MIT) |
| Track voting with size weighting and id-switch merging (`pipeline/signs.py`) | Licence-plate detector by morsetechlab (AGPL-3.0) |
| Relevance: does a sign apply to us? (`pipeline/relevance.py`) | Piper text-to-speech with the LJ Speech voice, run offline once to make the MP3s (GPL-3.0-or-later; voice data public domain) |
| Speed, turns and stops from the video alone (`pipeline/egomotion.py`) and the speed calibration (`scripts/calibrate_speed.py`) | FFmpeg, OpenCV, NumPy, PyTorch, FastAPI, React, Vite, Tailwind |
| Rules engine (`pipeline/rules_engine.py`) and the script that copies the official rule texts (`scripts/extract_pdd.py`) | Tailscale Funnel (public HTTPS address) |
| Privacy pipeline: how faces, heads, plates and the fallback area are combined (`pipeline/privacy.py`) | |
| Rendering, report, API, job queue, website, voice timing (`pipeline/render.py`, `app/`, `web/`) | |
| Evaluation scripts and the deploy pipeline (`eval/`, `.github/workflows/deploy.yml`) | |

**No LLM anywhere in the pipeline** (CLAUDE.md hard rule 6). Every decision is our own deterministic code
plus our detectors. The code was written with Claude Code following the build spec in `CLAUDE.md`. The
team reviewed, tested and merged every milestone (README "How this was built").

---

## 2. The journey of one video

```
 browser ── 8 MB chunks ──> /api/uploads ──> job in SQLite queue ──> one worker thread (models stay loaded)
                                                                         │
 1 ingest: ffprobe, limits, re-encode (H.264, ≤1920 px wide, ≤30 fps, no audio, HDR -> SDR)
                                                                         │
 2 analysis pass: every n-th frame (≈15 per second) ─┬─ sign detector (our YOLO11s, 1280 px) ─> our tracker
                                                     ├─ vehicles + people (YOLO11n) and plates (plate detector)
                                                     └─ ego-motion: road seen from above -> speed; distant scenery -> turns
                                                                         │
 3 voting (one decision per physical sign) -> relevance (does it apply to us?)
                                                                         │
 4 ego-motion finish: speed per frame, stops, turns
                                                                         │
 5 rules engine (pure function): signs + speed + turns -> events + rules timeline
                                                                         │
 6 render pass (decode again at 720p): pixelate faces/heads/plates FIRST, then draw sign boxes,
   limit/speed display and subtitles -> annotated.mp4, thumbnails, poster
                                                                         │
 7 report.json -> report page: video, timeline, voice prompts, events, official texts
   the raw upload and the unblurred working copy are deleted
```

| Step | What happens | Output | Code |
|---|---|---|---|
| Upload | The browser sends the file in 8 MiB chunks, then "complete" with student name, lesson label, consent | A queued job | `app/api/uploads.py`, `web/src/api.ts` |
| Queue | One worker thread takes the oldest job | Progress (stage, 0–1) in the database | `app/worker.py`, `app/db.py` |
| Ingest | Check the file, reject over 3 minutes, re-encode to one standard format | `normalized.mp4` (private job folder), quality warnings | `pipeline/ingest.py` |
| Frames | Keep every n-th frame so frames are evenly spaced | Frames with exact timestamps | `pipeline/frames.py` |
| Detection + tracking | Find sign boxes in each frame; link boxes of the same sign | `Detection` list with track ids | `pipeline/detect.py`, `pipeline/tracking.py` |
| Vehicles, people, plates | Find cars, buses, trucks, motorcycles, people and plates | Boxes per frame (for blur and for the speed mask) | `pipeline/vehicles.py` |
| Voting | One class per physical sign; drop short or unsure tracks | Voted signs | `pipeline/signs.py` |
| Relevance | Side of the road, facing, approaching, plate below | `SignEvent` with a reason in words | `pipeline/relevance.py` |
| Ego-motion | Speed, heading, turns, stops from the pixels | `MotionSample` list, `TurnEvent` list | `pipeline/egomotion.py` |
| Rules engine | Rules in force over time; judge speed, stops, turns, no entry | `LessonEvent` list + `RuleState` timeline | `pipeline/rules_engine.py` |
| Render | Second decode at 720p: blur first, then overlays | `annotated.mp4`, `thumbs/*.jpg`, `poster.jpg` | `pipeline/render.py`, `pipeline/privacy.py` |
| Report | Summary, score, signs not met, timings | `report.json` | `pipeline/report.py`, `pipeline/run.py` |
| Website | Report page with voice prompts | — | `web/src/pages/ReportPage.tsx`, `web/src/voice.ts` |

The data passed between steps is defined once, as strict Pydantic models in `pipeline/types.py`
(unknown fields are rejected). `tests/fixtures/report.sample.json` is a synthetic report that covers every
event type; the website was built against it first.

---

## 3. Module by module

### 3.1 Upload, queue and worker — `app/api/uploads.py`, `app/worker.py`, `app/db.py`

**What.** The browser sends the video in 8 MiB chunks (`PUT /api/uploads/{id}/chunk`). The server keeps a
chunk only once it has fully arrived, and `GET /api/uploads/{id}` lists the chunks it has, so an upload
can be resumed. The web page tries each chunk up to 4 times before it gives up. `complete` checks the consent box,
joins the chunks into a private job folder and adds a job to an SQLite table. One background thread takes
the oldest job, runs the pipeline and writes its progress back; the page polls `GET /api/jobs/{id}` every
1.5 s and opens the report when it is done.

**Why this way.**
- One GPU, so one worker. The models are loaded once and stay in memory (loading takes 1.3 s,
  `eval/results/benchmark.md`).
- SQLite needs no database server and survives restarts. Jobs that were running during a restart go back
  to the queue.
- A Python thread cannot be killed from outside, so every stage checks a deadline between frames.
- Client file names are never used in paths; only a cleaned extension is kept.

**Settings.** Max video length 180 s · max upload 1.5 GiB · chunk 8 MiB · at most 20 waiting jobs · job
timeout 900 s · unfinished uploads deleted after 24 h.

**Errors.** Every failure becomes a readable message key (for example `error.too_long`,
`error.not_a_video`, `error.queue_full`, `error.timeout`), never a stack trace.

**Who can see a lesson.** There is no public list of uploads. Lesson ids are 22-character random
tokens. The lessons page shows the sample lessons plus the ids this browser remembers (`localStorage`).

### 3.2 Ingest — `pipeline/ingest.py`

**What.** `ffprobe` reads duration, frame rate, size, rotation, codec and colour format. Not a video, or
over 3 minutes, gives a clear error. FFmpeg then re-encodes to H.264, at most 1920 px wide, constant frame
rate, no audio, rotation applied, HDR (iPhone HLG / Dolby Vision) tone-mapped to normal colours. Quality
warnings come from 20 sampled frames: portrait video, very short clip (< 10 s), low light (typical frame
brightness below 60 of 255), blurry (sharpness score below 50).

**Why.**
- Every later step reads one predictable file.
- The frame rate is never raised (`fps = min(source, 30)`): duplicated frames would look like a car
  standing still to the speed estimate.
- HDR is tone-mapped because otherwise colours wash out and red sign rings fade.

**Tested** with synthetic files: iPhone-style portrait HEVC `.mov`, HDR HLG, 4K at 60 fps, a video over 3
minutes, an audio-only file, a non-video file (`tests/test_edge_cases.py`, `tests/test_ingest.py`).

### 3.3 Frames — `pipeline/frames.py`

**What.** Decodes every n-th frame through an FFmpeg pipe, with n = round(frame rate / 15). A 30 fps video
gives 15 analysed frames per second; a 25 fps video gives 12.5.

**Why.**
- Keeping whole frames keeps the time between analysed frames exactly constant. Resampling 25 fps to
  15 fps would alternate gaps of 0.04 s and 0.08 s and break the speed estimate.
- **Two passes.** The analysis pass runs at full resolution. The render pass decodes again at 720p,
  because rendering needs the rules results, which need the whole video. Holding 3 minutes of 720p
  frames in memory instead would take about 7.5 GB (CLAUDE.md §10.2). Both passes use the same frame
  numbers.

### 3.4 Sign detector and its training — `pipeline/detect.py`, `training/`, `config/class_maps/`

**What.** A YOLO11s object detector that we fine-tuned on Russian road signs. It finds boxes in each frame
and gives each box a class (89 classes: signs by their GOST number, plus traffic-light states) and a
confidence. A class map turns model classes into catalogue codes and km/h values: the classes 3.24.1 …
3.24.11 are speed limits of 10 … 110 km/h, read off example crops (`training/class_crops.py`). The app
runs the detector at 1280 px with FP16 on the GPU and a low confidence threshold (0.25), because voting
filters later.

**Training data.** Dognellaf "russian-road-signs" (Hugging Face, CC BY 4.0): 43,851 video frames, of
which 38,918 have a label file and are used; 730 source videos (`eval/results/dataset_inventory.md`).

**Training choices** (CLAUDE.md §14.2, `training/train.py`):
- **Split by source video.** 119 of the 730 videos (3,027 frames) are held out for validation. Frames of
  one video are near-identical, so a random frame split would leak and inflate the score.
- Every 2nd frame kept (neighbours are near-duplicates), plus every frame with a rare class: 17,066
  training frames.
- **No horizontal flip** (`fliplr=0.0`): mirroring turns "no right turn" into "no left turn" and mirrors
  digits and arrows. No rotation either (signs are upright).
- YOLO11s from COCO weights, image size 1024, batch 6, mixed precision, cosine learning rate, mosaic
  off for the last 3 epochs, a checkpoint every epoch, early stop after 10 epochs without improvement.
- Result: training stopped early at epoch 35; the best was epoch 25 (mAP50 0.756 on the 1,000-frame
  validation subset used during training; CLAUDE.md §17 T5).

**Accuracy on the 3,027 held-out frames** (all 89 classes, at 1024 px): precision 0.714, recall 0.652,
mAP50 0.699, mAP50-95 0.57 (`eval/results/detector_eval.md`).

**Model comparison** (`training/compare_models.py`, `eval/results/model_comparison.md`). Same held-out
frames; every prediction and label translated into the same catalogue codes:

| Model | ms per image | Mean AP50 on signs both models know |
|---|---:|---:|
| StreetSignSense YOLO12s (pretrained, European signs) at 1280 px | 26.1 | 0.65 |
| Ours at 1024 px | 18.0 | 0.793 |
| **Ours at 1280 px (used)** | **17.1** | **0.81** |

We use 1280 px although we trained at 1024, because it scored higher on the held-out videos (0.81 vs
0.793; `config/settings.yaml` note).

**Why an adapter.** Our model did not exist while the rest was built. Every other module only sees
`Detection` objects, so swapping the model is a settings change plus a class map. A `ReplayDetector`
replays saved detections, so later stages can be tested without a GPU.

**Weak spots.**
- Speed-limit digits are among the weakest classes: AP50 at 1280 px is 0.27 for 10 km/h, 0.37 for 60 and
  0.17 for 80 (`model_comparison.md`). An 80 was read as 60 in a winter video (README).
- Signs missing from the training data cannot be found: no left turn (3.18.2), no U-turn (3.19), start/end
  of a built-up area (5.23/5.24). End of speed limit (3.25) has only 5 examples.
- Signs the model never saw are read as the closest class it knows: "no trucks" (3.4) as a 100 km/h limit,
  "no pedestrians" (3.10) as 70 (README).
- The back of a round sign was once read as "right turn only" (`eval/results/sample_signs.md`).

### 3.5 Our sign tracker — `pipeline/tracking.py`

**What.** The detector works one frame at a time. The tracker decides which boxes in consecutive frames
belong to the same physical sign and gives them one track id.

**How (plain English).** Signs move smoothly on screen: outwards from the point we drive towards, faster
and bigger as we approach. Each track remembers how its sign moved and grew, and predicts where it will
be in the next frame. A new box joins the track whose prediction is closest, measured in **sign sizes**
(30 px is far for a 20 px sign but close for a 120 px one). Pairs are linked closest first, each track and
each box at most once per frame, so two signs on one pole keep their own tracks. A track ends when its
sign has not been seen for 0.5 s. Detection uses **class-agnostic NMS** (one box per object even if the
model is unsure between "60" and "80"), so voting decides the class later.

**Why not ByteTrack** (the tracker built into Ultralytics, our first version)? ByteTrack links boxes only
when they overlap on screen. At 12.5–15 analysed frames per second, a sign close to the car moves more than
its own width between two frames, so its boxes never overlap and the sign is lost. On the Moscow highway
clip ru_02, ByteTrack kept no sign at all although the detector saw a keep-right sign at 0.95 confidence in
9 frames in a row (CLAUDE.md §10.3).

**Result** on the 12 demo clips (8 Tashkent, 4 Moscow; `eval/results/tracker_comparison.md`): 254 signs
kept after voting with our tracker vs 166 with ByteTrack; 170 vs 115 judged to apply to us. There is no
ground truth for these clips, so this counts signs kept; the signs only one tracker kept are listed there
for a person to check.

**Settings** (`track.*`): match within 1.0 sign sizes (2.5 for a track seen once, whose motion is still
unknown) · a sign may not grow or shrink more than 1.8× between two analysed frames · a class change costs
0.3 extra · newest motion estimate weighted 0.5 · a track ends after 0.5 s unseen.

**Tested:** a fast sign whose boxes never overlap stays one track; two signs on one pole keep their tracks;
a box far from the prediction starts a new track; a long gap ends a track (`tests/test_tracking.py`).

### 3.6 Track voting — `pipeline/signs.py`

**What.** One decision per physical sign. For each track, every frame votes for its class; the class with
the highest total wins. The sign is kept only if the winning class was seen in at least 3 frames with a
mean confidence of at least 0.5. Two tracks of the same sign split by a tracker id switch are merged (same
class, at most 2 s apart, centres within 15% of the frame width).

**Near views count more** (`vote.weight: size`). Each frame's vote is its confidence × the square root of
the box area. Far away, a speed limit's digits are a few blurry pixels and flicker between 40/60/80; close
up they are sharp. On the 12 demo clips this changed 2 of 265 decisions, both a "60" that really shows 50
(checked on crops; CLAUDE.md §10.4).

**Why.** A single frame can be wrong (a "60" read as "80", a reflection). A sign is usually seen in many
consecutive frames, so voting removes most one-frame mistakes. A switch (`vote.single_frame_baseline`)
turns voting off for comparison.

**Weak spot.** The confidence of a reading does not tell right from wrong: misreads had 0.54–0.73,
correct readings 0.55–0.81 on the demo clips, so no threshold can filter them (README).

### 3.7 Relevance: does the sign apply to us? — `pipeline/relevance.py`

**What.** For each voted sign, four simple, explainable checks. The decision and the reason in words are
written into the report ("on the right side, facing us, getting closer").
- **Side.** High and central (box bottom in the top 35% of the frame, centre in the middle 60%) →
  overhead. Otherwise the sign's sideways drift while it grows decides: driving forward, signs on the right
  drift right and signs on the left drift left (drift of at least 10% of the frame width). Small drift →
  the position of the largest view (right of centre by 3% of the width → right). Right and overhead signs
  apply to us; left ones do not (right-hand traffic).
- **Facing.** A round or square sign seen at an angle looks narrow. Width/height below 0.6 of its normal
  shape → it faces a side road, not us.
- **Approaching.** The box must grow (area × 1.3 or more from the start to the end of the track). A sign
  that does not grow belongs to a parallel road or one we drive away from.
- **Conditional.** A supplementary plate listed in `relevance.conditional_plate_codes` (8.1.1, 8.2.1, 8.5.4,
  8.13) directly below the sign (within 1.5 sign widths) → the sign may not apply right now; its violations
  are only "possible".

**Why.** No map in scope (24 hours). Every heuristic is visible in the report, so a person can judge it.

**Weak spots.** Wide roads, slip roads and sharp bends can fool the side and approach checks (README).
Real conditional plates (time of day, vehicle type) have 0–2 examples in the training data, so the
conditional check rarely fires (CLAUDE.md §11.1).

### 3.8 Vehicles, people and plates — `pipeline/vehicles.py`

**What.** Two more detectors run on every analysed frame:
- YOLO11n trained on COCO (Ultralytics): cars, buses, trucks, motorcycles and people, at 640 px,
  confidence ≥ 0.2;
- a licence-plate detector (YOLO11s by morsetechlab), at 1280 px, confidence ≥ 0.15.

**Used for** privacy (section 3.11) and for the speed estimate: other vehicles and people are hidden from
the road picture, because they move on their own.

**Why low thresholds.** An extra pixelated box is harmless; a missed plate is not (`config/settings.yaml`).

### 3.9 Speed, turns and stops from the video — `pipeline/egomotion.py`, `scripts/calibrate_speed.py`

There is no GPS: everything comes from the pixels.

**Speed (road-plane method).**
1. **Camera model.** Focal length from the field of view (default 70°), camera height (default 1.3 m),
   horizon line from where lane lines meet (their **vanishing point**, measured on 30 frames spread over
   the video; fallback 45% of the frame height). Both defaults can be changed per upload.
2. **Bird's-eye view (BEV).** The road patch from 3 m left to 3 m right and from 6 m to 26 m ahead is
   re-drawn as if seen from above, at 5 cm per pixel. In this view, driving forward makes the road slide
   down by (distance driven) / 5 cm pixels.
3. **Phase correlation** measures that slide between two consecutive analysed frames (a fast Fourier-based
   way to find how far one picture is shifted against another). Speed = shift × 0.05 m × analysed frames
   per second × 3.6 × `scale_k` → km/h. At 15 analysed frames/s, one pixel of shift is 0.05 × 15 × 3.6 =
   2.7 km/h.
4. **What is hidden first:** other vehicles and people; and things fixed to the camera — dashboard,
   on-screen date/speed text, logos — found as pixels that hardly change over the whole video.
5. **"Nothing moved" is double-checked.** Reflections or a car ahead at our speed create a strong match at
   zero shift. We prefer the best forward shift when it is at least half as strong; and when the answer
   is still zero, we subtract a running average of the road picture and look again.
6. **Smoothing and honesty.** A rolling median over 1 s. Moments with a weak match, a dark/featureless
   road or too little visible road become "not measured", never a guess.

**Calibration** (`scripts/calibrate_speed.py`, `eval/results/speed_calibration.json`). A person read the
dashcam's on-screen GPS speed every 5 s on two Moscow clips (36 readings each). ru_01 is the calibration
clip: `scale_k` = 1.014 (median true/estimated speed) and an error model err = 1.0 + 0.529 × speed km/h
(e.g. ≈ 33 km/h at 60 km/h). ru_02 is the test clip, never used for fitting.

**Accuracy on the test clip ru_02** (highway, true speed 64–89 km/h; `eval/results/speed_eval.md`): an
estimate for 28% of moments; mean absolute error 45.6 km/h; 90% of errors below 70.4 km/h. **It fails at
highway speed.** The team decided to report this as it is (CLAUDE.md §17 M6). The ru_01 numbers are
in-sample (it was used for calibration), so we do not quote them as accuracy.

**Why speeding is only ever "possible".** One dashcam's calibration does not cover every uploaded camera,
so `speed.calibrated` stays false. The display says "est." and the rules engine never reports a hard
speeding violation from video speed.

**Turns.** Distant scenery above the horizon shifts sideways when the car turns. The median sideways
**optical flow** (Lucas–Kanade: following small corner points from one frame to the next) gives the turning
rate; adding it up gives the heading. A turn = at least 60° within 10 s (turning rate above 3°/s, pauses
under 1 s bridged); 150° or more = U-turn. Angles need no calibration.

**Stops.** Speed under 2 km/h for at least 1 s. "Not moving" needs no scale, so it is the most reliable
output of this method.

**Tested** on synthetic road videos: known speeds 20, 50 and 90 km/h are recovered through a static
on-screen box; standing still is a stop; a stop in the middle of a drive; a right turn; no turns when
driving straight (`tests/test_egomotion_synthetic.py`).

**Weak spots** (README): highway speed; dark featureless asphalt can read 0 km/h; a wrong horizon or
camera height scales every speed; some short stops are missed.

### 3.10 Rules engine and the rule texts — `pipeline/rules_engine.py`, `config/rules/`, `scripts/extract_pdd.py`

**What.** A pure function: relevant signs + speed + turns → lesson events and a timeline of the rules in
force. No files, no network, no randomness, so 28 JSON scenarios test every behaviour
(`tests/fixtures/rule_scenarios/`).

**Where the rules come from.** `scripts/extract_pdd.py` copies, word for word, from the official Russian
Road Traffic Rules (ПДД РФ, Government Decree No. 1090, ред. от 28.08.2026, ConsultantPlus): each sign's
official name and meaning, the zone in which each sign applies, and the default limits (60 km/h in
built-up areas, 90 outside). Each entry records its source. A Russian-speaking team member checked every
line (`docs/RULES_CHECKLIST.md`); all 36 catalogue entries are marked verified. English texts are drafts
and labelled as such. Nothing is paraphrased and presented as official (CLAUDE.md hard rule 2).

**What it tracks and judges.**

| Rule | How it works | Event (severity) |
|---|---|---|
| Sign explained | The first time each sign code + value appears in the lesson | `sign_explained` (info) + voice prompt |
| Speed limit | Set by 3.24 when we pass it; ended by another 3.24, 3.25, 3.31, settlement signs, or our next turn; otherwise the default (60 in built-up areas, assumed) | Shown in the timeline |
| Speeding | Over the limit for at least 3 s; "speeding" only if over even after subtracting the speed error, with a limit from a sign and a calibrated camera — in practice always `possible_speeding` (reason: uncalibrated / default limit / conditional) | `possible_speeding` (possible) |
| Stop sign (2.5) | Slowest speed from 2 s before to 6 s after the sign: above 2 km/h → not made; no speed data → "stop not checked" (never a violation) | `stop_not_made` (violation) / `stop_ok` (complied) / info |
| Turn bans (3.18.x, 3.19) and mandatory directions (4.1.x) | Active until our next turn, or 300 m driven, or 60 s — an approximation of "the next intersection" (no map). A left-turn arrow also allows a U-turn (from the official text) | `turn_ban_violation` (violation) / `turn_ban_complied` (complied) |
| No entry (3.1) | Overhead sign → counts. Side sign → counts only if we turned toward its side between 3 s before and 8 s after passing it (team decision: a side sign usually closes the side road it stands at) | `no_entry_possible` (possible), never a hard violation |
| Conditional sign | A violation caused by a sign with a supplementary plate | downgraded to possible |

**Score.** 100 − 10 per violation − 3 per possible violation (never below 0), shown as "indicative".

**Why "possible" so often.** We never accuse without strong evidence: an assumed default limit, a sign
with a conditions plate, an uncalibrated speed and no-entry without a map are all "possible".

**Why ПДД РФ, not the Uzbek rules.** The training data and sign numbers are Russian, and the team chose the
Russian rules for the official texts (CLAUDE.md, Saturday 23:10; kept on Sunday 03:45). Uzbek signs use the
same numbering. On the Tashkent clips the default limit is still right: the Uzbek rules (lex.uz,
docs/5953887) allow 60 km/h in Tashkent — the same 60 as ПДД РФ §10.2 (CLAUDE.md "Start here").

**Weak spots.** No map, so "until the next intersection" is approximated, and driving straight into a
one-way street past a side no-entry sign is missed. The speed limit also ends at our next turn rather than
at the next intersection.

### 3.11 Privacy — `pipeline/privacy.py`

**What.** On every rendered frame and every thumbnail, before anything is drawn or saved:
1. faces found by YuNet (score ≥ 0.6);
2. the head area (top 28%) of every person box — YuNet misses small and side-on faces, a person box does
   not;
3. plates found by the plate detector, also those found one frame before or after (so one missed frame
   does not flash a readable plate);
4. a fallback area over the lower 65% (and middle 80% of the width) of every vehicle box, for plates the
   detector misses.

All regions are pixelated to about 5 blocks across, which cannot be undone. Sound is removed. The raw
upload and the unblurred working video are deleted after processing; only blurred media are served.

**Why both plate methods.** On a Moscow clip the first fallback area alone left plates of close cars
readable; the plate detector found most plates but missed a dirty van's (CLAUDE.md §14.3).

**Our own bonnet.** On many phone videos the camera sees the car's own bonnet; the vehicle detector calls
it a car, and the fallback area pixelated the bottom of every frame. A vehicle box that is at least half
the frame wide, starts in the lower 40% and touches the bottom edge is treated as our bonnet: no fallback
area. Detected plates are always hidden (`tests/test_privacy.py`).

**Checked by eye.** `eval/privacy_gallery.py` makes contact sheets of frames from every served video and
all thumbnails; the samples were checked before going public (CLAUDE.md §17 M5).

**Note.** Faces printed on posters and billboards are pictures, not people, and they are not always
pixelated (seen on the Alisher Navoi street sample's contact sheet).

### 3.12 Render and report — `pipeline/render.py`, `pipeline/report.py`, `pipeline/run.py`

**What.** The render pass decodes the video again at 720p. For each frame: blur first; then green boxes
and names for relevant signs; a display with the limit in force and our speed ("est.", red when over); a
subtitle line while an event is "spoken" (3 s per event; overlapping subtitles share one line, "Main road ·
Pedestrian crossing"). Frames are piped as raw pixels into FFmpeg (H.264, no audio). Thumbnails are cut
around each sign's largest view, from blurred frames only.

`report.json` contains the lesson info, events, the rules timeline, speed at 2 Hz, every sign (relevant or
not, with the reason), signs not met in this lesson, quality warnings, and processing statistics: seconds
per stage, frames per second, peak GPU memory, model version and git commit.

### 3.13 Website and voice — `web/`, `web/src/voice.ts`

**Pages.** Upload (drag and drop, student name, lesson label, optional camera height and field of view,
consent box, progress, queue position); Lessons (samples + this browser's lessons); Report.

**The report page.** The video; a timeline under it with the speed limit band and coloured event markers
(red violation, amber possible, green done right, grey info; tap to jump); a speed chart against the
limit; summary cards (indicative score, signs passed, violations, possible, done right); the event list
with thumbnails, the English explanation and the official Russian text with its source link; "signs not
met yet"; every sign detected with its reason; "How this was computed". English only; mobile-friendly.

**Voice prompts.** English MP3s made once, offline, with Piper text-to-speech (`scripts/generate_voice.py`,
46 entries in `assets/voice/en/manifest.json`, including the connector "and"). Not recorded by the team,
and nothing is synthesised while the report plays. If an MP3 cannot play, the browser's own speech says
the same text.
- **When.** A sign is announced when it is **confirmed** — its 3rd sighting as its final class — which is
  when a system in the car could speak. Its rule still starts when we pass it. Across the 12 demo clips that
  is a median of 2.1 s before passing; 85% at least 1 s ahead, 54% at least 2 s
  (`eval/results/tracker_comparison.md`).
- **Never cut off.** A prompt that comes due while another is speaking waits and follows it joined by
  "and" ("Main road… and pedestrian crossing"). A prompt more than 4 s late is dropped, so the voice never
  lags behind the video. Jumping in the video clears the queue.
- **iPhones** only play audio started by a tap: the first tap on Play unlocks one shared audio element
  that every prompt reuses.

### 3.14 Deployment — `.github/workflows/deploy.yml`, `deploy/`

A push to `main` triggers GitHub Actions on our own machine (a **self-hosted runner**). It checks out the
commit in a separate deploy folder, installs exact package versions, builds the website, runs the fast
tests, restarts the `signcoach` systemd service and waits until `/health` reports the new git commit — our
proof that the pushed code is live. Tailscale Funnel gives the public HTTPS address
https://signcoach.taild1ce46.ts.net.

**Security.** The repo is public and the runner executes code on our machine, so the workflow never runs
on pull requests: only on pushes to `main` and manual runs (CLAUDE.md §16).

**Machine.** One Windows PC with WSL2 Ubuntu 24.04, an RTX 3070 Ti (8 GB), a Ryzen 9 5900X and 32 GB RAM
(CLAUDE.md §4). It trained the model and serves the site.

### 3.15 Evaluation and tests — `eval/`, `tests/`

Every number in the deck and README comes from a script (README "Results"):

| Script | Measures | Result file |
|---|---|---|
| `eval/eval_detector.py` | Detector accuracy on 3,027 held-out frames (119 unseen videos) | `detector_eval.md` |
| `training/compare_models.py` | Pretrained European model vs ours, per sign | `model_comparison.md` |
| `eval/compare_trackers.py` | ByteTrack vs our tracker; how early signs are announced | `tracker_comparison.md` |
| `eval/eval_sample_signs.py` | Signs shown in the 4 sample lessons vs the team's check | `sample_signs.md` |
| `scripts/calibrate_speed.py`, `eval/eval_speed.py` | Speed calibration (ru_01) and accuracy (ru_02) | `speed_calibration.json`, `speed_eval.md` |
| `eval/benchmark.py` | Processing time, GPU memory, GPU power, energy | `benchmark.md` |
| `eval/failure_gallery.py` | Worst misses and most confident false alarms | `failure_gallery/` |
| `eval/privacy_gallery.py` | Contact sheets for the privacy check by eye | outside git |

**Not measured** (we say so): how many signs we miss in full lessons (recall), how often the right limit
is in force, how often stop judgements are right (CLAUDE.md §15).

**Tests.** 83 pytest tests (`uv run pytest`): 82 fast ones run on every deploy — including the 28 rule
scenarios, the tracker, voting, relevance, synthetic speed/turn/stop videos, privacy, ingest edge cases and
the upload API — plus 1 slow end-to-end run of the whole pipeline on a synthetic clip.

---

## 4. Numbers cheat sheet

| What | Number | Source |
|---|---|---|
| Training frames / held-out frames | 17,066 / 3,027 (from 119 of 730 videos) | CLAUDE.md §14.2 T3 |
| Labelled frames in the dataset | 38,918 of 43,851; 89 classes | `eval/results/dataset_inventory.md` |
| Detector on held-out frames (all classes, 1024 px) | precision 0.714, recall 0.652, mAP50 0.699, mAP50-95 0.57 | `eval/results/detector_eval.md` |
| Ours vs pretrained European model (mean AP50, signs both know) | 0.81 (1280 px) vs 0.65; 0.793 at 1024 px | `eval/results/model_comparison.md` |
| Inference time per image (RTX 3070 Ti) | 17.1 ms (ours, 1280 px) vs 26.1 ms | `eval/results/model_comparison.md` |
| Training run | early stop at epoch 35, best epoch 25 (mAP50 0.756 on the 1,000-frame subset) | CLAUDE.md §17 T5 |
| Signs kept: our tracker vs ByteTrack (12 clips) | 254 vs 166 (170 vs 115 applying to us) | `eval/results/tracker_comparison.md` |
| ByteTrack on the highway clip ru_02 | 0 signs | `eval/results/tracker_comparison.md` |
| Voice announcement before passing the sign | median 2.1 s; 85% ≥ 1 s; 54% ≥ 2 s (ByteTrack: 1.9 s) | `eval/results/tracker_comparison.md` |
| Size-weighted voting | changed 2 of 265 decisions, both 60 → 50 (correct) | CLAUDE.md §10.4 |
| Signs shown in the 4 Tashkent samples | 66 of 68 right (97%) | `eval/results/sample_signs.md` |
| Speed on the test clip ru_02 (64–89 km/h) | estimate for 28% of moments, mean error 45.6 km/h, p90 70.4 km/h | `eval/results/speed_eval.md` |
| Speed calibration (ru_01) | scale_k 1.014; error 1.0 + 0.529 × v km/h | `eval/results/speed_calibration.json` |
| Processing time | 95.0 s per minute of video (1.45–1.74× real time); target was real time | `eval/results/benchmark.md` |
| GPU time | 1.584 GPU-hours per lesson-hour | `eval/results/benchmark.md` |
| GPU power and energy | 56.2 W average; 0.089 kWh per lesson-hour | `eval/results/benchmark.md` |
| PyTorch GPU memory peak | 122 MB (weights + activations; CUDA runtime not included) | `eval/results/benchmark.md` |
| Model loading (once per server start) | 1.3 s | `eval/results/benchmark.md` |
| Detector failure gallery (held-out frames) | 797 misses or misreads, 1,026 false alarms at confidence ≥ 0.25 | `eval/results/failure_gallery/README.md` |
| Rule scenarios / tests | 28 / 83 (82 fast + 1 end-to-end) | `tests/` |
| Catalogue entries checked against ПДД | 36 of 36 | `config/rules/signs.json` |
| Default limits | 60 km/h built-up, 90 outside (ПДД РФ §10.2/10.3) | `config/rules/rules.json` |
| Upload limits | 3 minutes, 1.5 GiB, 20 waiting jobs | `config/settings.yaml` |

Cost per lesson-hour in money needs the electricity price, which is not in the repo
(`cost.electricity_price_per_kwh` is empty): cost = 0.089 kWh × price.

---

## 5. Questions the jury may ask

**About the AI and the technology**

1. **What does the AI actually do?** Our detector finds signs in each frame. Our tracker and voting turn
   many noisy boxes into one decision per physical sign. Geometry decides which signs apply to us. The
   speed, turns and stops come from the video itself. A deterministic rules engine then turns all of this
   into explained events.
2. **Did you train your own model?** Yes. YOLO11s fine-tuned on 17,066 frames of the Dognellaf Russian
   road-sign dataset, at 1024 px, on our RTX 3070 Ti. Training stopped early at epoch 35; we use the best
   epoch, 25.
3. **How do we know the accuracy is not inflated?** The validation frames come from 119 videos the model
   never saw; we split by video, not by frame. On them: precision 0.714, recall 0.652, mAP50 0.699. On our
   own Tashkent sample lessons, 66 of the 68 signs we show are right.
4. **Why not just use a ready-made sign model?** We tested one (StreetSignSense, European signs) on the
   same held-out frames: mean AP50 0.65 vs 0.81 for ours, and ours is faster (17.1 vs 26.1 ms per image).
5. **Why YOLO11s?** A small detector that trained on our single 8 GB GPU in about 3 hours (22:29 → 01:43)
   and runs at 17.1 ms per image.
6. **Why 1280 px when you trained at 1024?** Small, distant signs. On the held-out videos 1280 px scored
   0.81 vs 0.793.
7. **Why did you write your own tracker?** ByteTrack links boxes only when they overlap. Close signs at
   city or highway speed move more than their own width between analysed frames, so ByteTrack lost them —
   on the highway clip it kept no sign at all. Ours predicts each sign's motion. On 12 clips it keeps 254
   signs vs 166.
8. **Why not just analyse more frames per second?** More frames cost GPU time, and processing is already
   slower than real time (95 s per video minute). Evenly spaced frames also matter for the speed estimate.
9. **What is voting?** A sign is seen in many frames. Every frame votes for a class, closer (bigger) views
   count more, and a sign counts only if its class was seen in at least 3 frames with mean confidence ≥ 0.5.
   It removes one-frame mistakes and fixed a 60/50 misreading on a sample.
10. **How do you know a sign applies to you?** Side of the road (from how the sign drifts as we approach),
    whether it faces us, whether we approach it, and whether a plate below adds conditions. The reason is
    written in the report for every sign.
11. **How do you measure speed without GPS?** We re-draw the road as seen from above and measure how far it
    slides between frames (phase correlation). The dashboard, on-screen text and other cars are hidden
    first.
12. **How accurate is the speed?** On the test clip ru_02 (highway, 64–89 km/h) it gave an estimate for only
    28% of moments, with a mean error of 45.6 km/h. It fails at highway speed. That is why it is labelled
    "est." and why speeding is only ever "possible".
13. **Why is the speed so wrong on the highway?** The method needs a clear match of road texture between
    frames. Moments without a clear match are dropped (hence 28% coverage). At 80 km/h and 12.5 analysed
    frames per second the road moves about 1.8 m between frames. The result also scales with the assumed
    camera height and horizon. The robust fix is a second speed source (the car's or a phone's GPS), which
    we left out of scope: the video is the only input.
14. **How do you detect turns and stops?** Turns: distant scenery shifts sideways when the car turns; 60° or
    more within 10 s is a turn, 150° a U-turn. Stops: speed under 2 km/h for at least 1 s. Neither needs
    calibration.
15. **What happens with a sign that is not in the training data?** The model reads it as the closest class
    it knows: "no trucks" became a 100 km/h limit on a sample. Missing entirely: no left turn, no U-turn,
    start/end of built-up areas. Fix: add examples of these signs and retrain.
16. **What are your biggest failure cases?** Highway speed; misread speed-limit digits; unknown signs read
    as known ones; the back of a sign read as a sign; no map. All are listed in the README and
    `eval/results/failure_gallery/`.
17. **Is there any LLM or external AI API in the system?** No. No calls to any outside service; all
    decisions are our code and our detectors on our machine.
18. **Why not send the video to GPT-4V or a cloud vision API?** The videos show people and plates, so they
    stay on our machine. A trained detector plus deterministic rules is explainable: every event has a
    sign, a time, a rule text and a video moment. And our own model is measured on held-out data.

**About rules and the product**

19. **Where do the rule texts come from?** Copied word for word from the official ПДД РФ by a script, with
    the source of each entry, and checked line by line by a Russian-speaking team member.
20. **Why Russian rules for Uzbekistan?** The training data and the sign numbers are Russian; Uzbek signs
    use the same numbering. The Tashkent default limit is 60 km/h in both rule books. The rules are data
    files (`config/rules/`), so the Uzbek texts can replace them.
21. **Why are most violations only "possible"?** We never accuse without strong evidence: assumed default
    limits, an uncalibrated speed estimate, signs with a conditions plate and no-entry without a map are
    all "possible".
22. **Does it tell the student what they may do?** No. It only warns and explains; it never says an action
    is permitted (CLAUDE.md hard rule 3).
23. **What is the score?** 100 minus 10 per violation and 3 per possible violation. It is labelled
    indicative.
24. **Why are signs announced only about 2 s before passing, not earlier?** An in-car system can only speak
    once it has recognised the sign; earlier would mean using future frames. We announce at confirmation
    (the 3rd sighting): median 2.1 s before passing.
25. **Can it work live in the car?** Not built; upload only. The voice timing already simulates the in-car
    moment. Processing is 1.45–1.74× slower than real time on our GPU, so live use needs a faster setup
    (CLAUDE.md suggests a smaller image size and fewer frames per second).
26. **What if the jury uploads a strange video?** Tested: iPhone HEVC `.mov`, portrait, HDR, 4K at 60 fps,
    audio-only and non-video files. Over 3 minutes gives a clear error. Odd but valid videos get a report
    with warnings (portrait, dark, blurry, short, speed not measured).

**About privacy, data and licences**

27. **How do you protect privacy?** Faces, people's heads and number plates are pixelated in every frame and
    thumbnail, with two plate methods. Sound is removed. The original upload is deleted. Lessons have no
    public list, and links are random.
28. **What if the face or plate detector misses something?** The fallback area on every vehicle, plates from
    neighbouring frames and head areas from person boxes cover the misses. We also checked every sample
    video by eye.
29. **What data did you use? Is it licensed?** Dognellaf (CC BY 4.0) for training. Moscow dashcam clips
    from the team for evaluation. Tashkent YouTube clips used with the owners' consent for the samples
    (`DATA_SOURCES.md`).
30. **Can you sell this?** The repo is AGPL-3.0 because Ultralytics is. A commercial launch needs an
    Ultralytics Enterprise licence or a permissively licensed detector (README).

**About cost, scale and engineering**

31. **How fast is it and what does it cost?** 95 s of processing per minute of video. The GPU draws 56.2 W
    on average: 0.089 kWh per lesson-hour (electricity cost = 0.089 kWh × the price). GPU time is the
    bigger resource: 1.584 GPU-hours per lesson-hour on one RTX 3070 Ti.
32. **How does it scale?** Today: one worker, one GPU, a queue of up to 20 jobs. More driving schools need
    more GPU workers or faster settings.
33. **How do you know the deployed site runs the submitted code?** `/health` reports the git commit; the
    deploy only finishes when that commit is live.
34. **What did Claude Code write, and how do you know it works?** It wrote the code to our spec
    (`CLAUDE.md`). Every milestone was reviewed, tested and merged by the team. 83 automated tests,
    measured results on held-out data, the team's checks (rule texts, speed readings, sample signs, privacy
    sheets). Every module starts with a plain-English explanation of what it does and why.
35. **What would you do next?** Add the missing signs to the training data; a colour check against sign
    backs; a map for intersections and no-entry; a real speed source; faster processing; live mode in the
    car.

---

## 6. Glossary

- **AP50 / mAP50.** Average precision with a box counted as found when it overlaps the true box by at least
  50% (IoU 0.5); mAP = the mean over classes. **mAP50-95** averages over stricter overlaps.
- **Precision / recall.** Precision = share of found signs that are real; recall = share of real signs we
  found.
- **IoU.** Overlap of two boxes: shared area / combined area.
- **NMS (non-maximum suppression).** Keeps one box where the detector proposed several overlapping ones.
  **Class-agnostic** NMS keeps one box even if the boxes have different classes.
- **YOLO / fine-tuning / epoch.** YOLO is a family of fast object detectors. Fine-tuning = continuing to
  train a model already trained on other images. An epoch = one pass over the training data.
- **Augmentation, `fliplr`.** Random changes to training images; horizontal flip is off because it mirrors
  signs.
- **Track / tracker.** A track is one physical sign followed across frames; the tracker builds tracks.
  **ByteTrack** is the tracker built into Ultralytics.
- **Voting.** Combining a track's per-frame classes into one decision.
- **BEV (bird's-eye view).** The road re-drawn as if seen from above.
- **Phase correlation.** Measuring how far one picture is shifted against another with Fourier transforms.
- **Optical flow (Lucas–Kanade).** Following small points from one frame to the next.
- **Vanishing point / horizon.** Where parallel lane lines meet in the image; it fixes the horizon height.
- **Yaw / heading.** Turning rate / the direction the car points.
- **FP16.** Half-precision numbers on the GPU: faster, less memory.
- **CFR / CRF.** Constant frame rate / the H.264 quality setting.
- **HDR, tone mapping.** High-dynamic-range video (iPhone); tone mapping converts it to normal colours.
- **GOST numbers.** Russian sign numbering (e.g. 3.24 = maximum speed limit), also used in Uzbekistan.
- **ПДД РФ.** The Russian Road Traffic Rules (Government Decree No. 1090).
- **SQLite.** A database stored in one file, no server.
- **systemd.** Linux service manager that keeps the website running and restarts it.
- **Self-hosted runner.** Our own machine running GitHub Actions jobs.
- **Tailscale Funnel.** Publishes a local port on a public HTTPS address.
- **TTS (text-to-speech).** Software that turns text into speech; we used Piper offline.
- **AGPL-3.0.** An open-source licence that requires sharing the source of network services built on it.

---

## 7. Where things are

```
app/            web server: API (uploads, jobs, lessons, health), job queue (SQLite), worker thread
pipeline/       the processing: ingest, frames, detect, tracking, signs (voting), relevance, vehicles,
                egomotion, rules_engine, privacy, render, report, run (ties it together), types (contracts)
config/         settings.yaml (every threshold), rules/ (signs.json, rules.json), class_maps/, i18n/en.json
training/       dataset preparation, training, model comparison
eval/           evaluation scripts; results/ (every number we quote); ground_truth/ (human readings)
scripts/        models download, official rule texts, voice prompts, speed calibration, sample lessons
web/            the website (React + TypeScript)
assets/voice/   the pre-generated English voice prompts
tests/          83 tests, incl. 28 rule scenarios in fixtures/rule_scenarios/
deploy/, .github/workflows/   systemd service, host setup, Tailscale Funnel guide, deploy workflow
docs/           this guide, LIMITATIONS.md, RULES_CHECKLIST.md
```
