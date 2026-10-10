# How SignCoach works — one section per module (for the team's Q&A)

Each section: what the module does, why it is built that way, the numbers that control it
(`config/settings.yaml`), and questions the jury may ask. Code paths are clickable on GitHub.

---

## 1. Upload, queue and worker — `app/api/uploads.py`, `app/worker.py`, `app/db.py`

**What.** The browser sends the video in 8 MB chunks (`PUT /api/uploads/{id}/chunk`); if the connection
drops it asks which chunks arrived (`GET /api/uploads/{id}`) and resends only the missing ones.
`complete` checks the consent box, joins the chunks into a job folder and puts a job in an SQLite table.
One background thread takes the oldest job, runs the pipeline and writes progress back.

**Why.** One GPU, models loaded once (loading per job would waste seconds); SQLite needs no server and
survives restarts (running jobs go back to the queue). A Python thread cannot be killed, so every stage
checks a deadline (15 min) between frames.

**Q: What if two people upload at once?** Both jobs queue; the page shows the queue position. Max 20 waiting.
**Q: Can someone see other people's lessons?** No public list; lesson ids are 22-character random tokens.

## 2. Ingest — `pipeline/ingest.py`

**What.** `ffprobe` reads duration, frame rate, size, rotation, codec, colour. Over 3 minutes or not a video →
a clear error. Then ffmpeg re-encodes to H.264, ≤1920 px wide, constant frame rate, **no audio**,
rotation applied, HDR (iPhone HLG/Dolby Vision) tone-mapped to SDR. Quality warnings: portrait, dark, blurry, short.

**Why.** Every later stage reads one predictable file. The frame rate is never raised (30 fps cap only
lowers it): duplicated frames would look like the car standing still to the speed estimate.

## 3. Frames — `pipeline/frames.py`

**What.** Decodes every n-th frame (n = round(fps / 15): 30 fps → 15 fps, 25 fps → 12.5 fps) through an
ffmpeg pipe. Two passes: analysis (full resolution) and render (720p), with identical frame indices.

**Why.** Keeping whole frames keeps the time between frames exact (resampling 25 → 15 fps would alternate
0.04 s and 0.08 s gaps and break speed). Rendering needs the rules results, which need the whole video,
so a second cheap decode is better than holding 7 GB of frames in memory.

## 4. Sign detection — `pipeline/detect.py`, `training/`

**What.** YOLO11s fine-tuned by us on the Dognellaf Russian road-sign dataset (89 classes: signs by GOST
number + traffic lights), at 1024 px, 40 epochs. A class map turns
model classes into sign codes and km/h values (3.24.1 … 3.24.11 = 10 … 110 km/h, read off example crops).

**Training choices.** Split **by source video** (730 videos; 119 held out) — a random frame split would
leak near-identical frames and inflate the score. Every 2nd frame kept (neighbours are near-duplicates),
plus every frame with a rare class. **No horizontal flip** augmentation: it would turn "no right turn"
into "no left turn" and mirror digits. Mosaic off in the last 3 epochs.

**Q: Why not just use a pretrained sign model?** We tried one (StreetSignSense, European signs): on our
held-out Russian frames it is much weaker — see `eval/results/model_comparison.md`.
**Q: What can't it see?** No-left-turn, no-U-turn and settlement signs are absent from the training data.

## 5. Tracking and voting — `pipeline/tracking.py`, `pipeline/signs.py`

**Tracking (our own).** The detector works frame by frame; the tracker decides which boxes belong to the
same physical sign. Each track predicts where its sign will be in the next frame from how it has been
moving and growing; a new box joins the track it is closest to, measured in sign sizes; closest pairs are
linked first, so two signs on one pole keep their own tracks. A track ends after 0.5 s unseen.

**Why not ByteTrack** (Ultralytics' built-in tracker, our first version): it links boxes only when they
overlap. At 12.5–15 analysed frames per second a sign close to the car moves more than its own width
between frames, so ByteTrack dropped it — on the Moscow highway clip it found no sign at all although the
detector saw a keep-right sign at 0.95 confidence in 9 frames in a row. Numbers: `eval/results/tracker_comparison.md`.

**Q: Why not just analyse more frames per second?** Even at 25 fps the boxes of a close sign often overlap by
less than ByteTrack needs, and it would double the GPU time.


**What.** For each tracked sign, the class with the highest **sum** of confidences wins; the sign is kept only
if seen in ≥3 frames with mean confidence ≥0.5. Tracks of the same sign split by a tracker id switch are
merged (same class, ≤2 s apart, close boxes).

**Why.** Single frames are noisy ("60" read as "80", reflections). A switch turns this off (`vote.single_frame_baseline`) to measure what voting adds.

## 6. Relevance — `pipeline/relevance.py`

**What.** Decides whether a sign applies to us, and writes the reason in words:
- **side:** high and central → overhead; otherwise the sign's sideways drift while it grows (driving forward,
  right-side signs stream right, left-side signs stream left), falling back to its position;
- **facing:** a round/square sign seen much narrower than its normal shape faces a side road;
- **approaching:** the box must grow over the track;
- **conditional:** a supplementary plate under the sign → violations become "possible".

**Q: Why not use a map?** No map in scope (24 h); every heuristic is visible in the report so a person can judge.

## 7. Speed, turns and stops from the video — `pipeline/egomotion.py`

**What.** Camera model: focal length from the field of view, horizon from where lane lines meet. Each frame's
road is re-sampled into a **bird's-eye view** (6 m × 20 m patch at 5 cm per pixel); the forward shift between
frames (phase correlation) × 5 cm × frames per second = speed. Things fixed to the camera (dashboard,
on-screen text, logos — pixels that hardly change over the whole video) and other vehicles are masked; a
"nothing moved" match is overruled when the background-subtracted road clearly moved. Turns: sideways
optical flow of distant scenery → yaw → heading; ≥60° within 10 s is a turn, ≥150° a U-turn.

**Honesty.** Low-quality moments are "not measured". The estimate scales with camera height and lens, so it
is calibrated on a dashcam whose screen shows its GPS speed (ru_01) and tested on another clip (ru_02).
Because uploads come from many cameras, speeding is only ever "possible".

**Q: How accurate?** See `eval/results/speed_eval.md` (ru_02, read from the dashcam's own speed display).
**Known failures:** dark featureless asphalt can read 0; a wrong horizon scales the speed.

## 8. Rules engine — `pipeline/rules_engine.py`, `config/rules/`

**What.** A pure function: signs + speed + turns → lesson events and the rules in force over time. It keeps
the speed limit (from 3.24, ended by 3.25 / 3.31 / a new limit / settlement signs / our next turn), turn bans
(3.18.x, 3.19) and mandatory directions (4.1.x) until our next turn or 300 m / 60 s, and judges:
speeding (violation only if over even after subtracting the speed error), stops at 2.5 (slowest speed in
[−2 s, +6 s]), turns against a sign, no entry (always "possible"; a no-entry sign at the side of the road
usually closes a side road, so it counts only if we turned that way around it — overhead signs always count).
27 JSON scenarios test every rule.

**Where the rules come from.** `scripts/extract_pdd.py` copies the sign definitions, zones of action and
default limits (60 km/h in built-up areas, 90 outside) verbatim from ПДД РФ; a Russian-speaking team member
checked every entry (`docs/RULES_CHECKLIST.md`). Examples taken from the text: a left-turn arrow also allows a
U-turn; 3.31 ends speed and overtaking zones, not turn bans.

**Q: Why "possible" so often?** We never accuse without strong evidence: assumed default limits, signs with a
conditions plate, uncalibrated speed and no-entry are all "possible".

## 9. Privacy — `pipeline/privacy.py`, `pipeline/vehicles.py`

**What.** In every rendered frame and thumbnail: faces (YuNet), people's head areas (COCO person boxes),
licence plates (plate detector, also plates found one frame before/after) and a fallback area over the lower
65% of every vehicle — all pixelated to ~5 blocks across. Sound removed; raw upload and unblurred working
video deleted after processing.

**Why both plate methods.** On our Moscow clip the first version (a band in the lower middle of each car) left
plates of close cars readable; the plate detector found most, but missed a dirty van's — so we use both.

## 10. Render and report — `pipeline/render.py`, `pipeline/report.py`

**What.** Draws sign boxes, the limit/speed display ("est." for video speed) and subtitles for voice prompts
onto the blurred frames, encodes 720p H.264 through an ffmpeg pipe, cuts thumbnails and a poster from blurred
frames. `report.json` holds events, the rules timeline, speed at 2 Hz, every sign with its relevance reason,
signs not met yet, warnings and processing statistics (seconds per stage, fps, GPU memory, model, git commit).

## 11. Website — `web/`

**What.** React + TypeScript + Tailwind. Upload with progress; the report page: video, timeline with the speed
limit band and coloured event markers, voice prompts played when playback passes an event (pre-generated
English MP3s; iPhones need one tap to unlock audio), summary, events with the official sign text, every sign
detected, how it was computed.

## 12. Deployment — `.github/workflows/deploy.yml`, `deploy/`

A push to `main` deploys to the team machine (self-hosted runner, systemd, Tailscale Funnel). The workflow
waits until `/health` reports the new git commit — that is our deployment evidence.

## 13. Evaluation — `eval/`, `training/compare_models.py`

Every number in the deck comes from these scripts (results in `eval/results/`): detector accuracy on 3,027
held-out frames from 119 unseen videos; placeholder vs our model per sign; processing time and cost per
lesson; failure gallery; speed accuracy on a clip with the dashcam's own GPS speed.
