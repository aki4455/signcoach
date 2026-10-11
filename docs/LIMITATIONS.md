# SignCoach: limitations and known failure cases

This is what SignCoach cannot do yet, where it fails and what we did not measure. Every number comes
from a file in this repository; the file is named in brackets. Status: Sunday 11 October 2026.

Terms used below:
- **AP50**: a standard score for a sign detector, from 0 to 1. It combines how many real signs it finds
  with how many of its boxes are right. Higher is better.
- **Precision**: the share of what we show that is right.
- **Recall**: the share of real signs that we find.
- **Held-out videos**: 119 videos (3,027 frames) that were never used in training, so the scores are not
  inflated by memorised scenes [eval/results/detector_eval.md].

## For the slide

- **Speed from video fails at highway speed.** On our test clip it gave a speed only 28% of the time,
  with a mean error of 45.6 km/h [eval/results/speed_eval.md]. Speeding is only ever reported as "possible"
  [config/settings.yaml].
- **Some signs are missing from the training data.** No-left-turn, no-U-turn and town start/end signs
  have 0 examples; end of speed limit has 5, none of them in training [eval/results/dataset_inventory.md,
  eval/results/detector_eval.md].
- **2 of the 68 signs in our sample lessons are wrong.** The other 66 are right. The wrong ones: a
  "no trucks" sign, never in our training data, read as a 100 km/h limit; and the back of a sign read
  as "right turn only" [eval/results/sample_signs.md].
- **Some speed-limit values are weak.** For 80 km/h signs on held-out videos, our detector scores AP50 0.17
  and the pretrained European model 0.66 [eval/results/model_comparison.md].
- **There is no map.** Sign zones that officially end at the next intersection end at our next turn
  instead (turn bans also after 300 m or 60 s). A no-entry sign at the side of the road counts only if we
  turn toward it [config/rules/rules.json, config/settings.yaml].
- **It is slower than real time.** Processing takes 95 s per minute of video (1.45–1.74× the video's
  length) on one RTX 3070 Ti, using 0.089 kWh per lesson-hour [eval/results/benchmark.md].
- **Not measured:**
  - how many signs we miss over a whole lesson;
  - whether the speed limit in force is right over time;
  - whether stop judgements are right [CLAUDE.md §15].

## 1. Speed from the video

### Speed is unreliable at highway speed, and it was tested on one clip only

- **What happens:** we have no GPS. Speed comes from how fast the road slides under the camera.
- **Evidence:** on the test clip, a Moscow highway at 64–89 km/h [README.md], a person read the
  dashcam's own GPS speed at 36 moments [eval/results/speed_eval.md]. Our estimate:
  - gave a speed at 28% of those moments;
  - had a mean error of 45.6 km/h;
  - read too high by 8.4 km/h on average;
  - was off by more than 70.4 km/h at 1 moment in 10 (of the moments with an estimate).
- **Why:** the estimate depends on where we find the horizon, and it scales with the camera's height and
  lens. Unless the uploader enters them, we assume a height of 1.3 m and a 70° field of view
  [config/settings.yaml]. Dark, featureless asphalt gives nothing to track, so it can read 0 km/h
  [README.md]. We have not fully diagnosed why it fails at highway speed.
- **Consequence:**
  - The calibration comes from one dashcam (ru_01), so it does not cover other cameras
    (`speed.calibrated: false` [config/settings.yaml]). Speeding is therefore never a definite violation,
    only "possible speeding" [CLAUDE.md, start-here section].
  - The measured error model is 1.0 + 0.529 × speed km/h [eval/results/speed_calibration.json]. At
    60 km/h that is 1.0 + 0.529 × 60 = 32.74 km/h.
  - We have no true speed for the Tashkent sample clips, so we cannot say whether their "possible
    speeding" events are right.
- **Next:**
  - take speed from the car, the phone's GPS or the dashcam's speed overlay;
  - calibrate per camera;
  - test at city speeds.

### Stops and turns

- **What happens:** speed from video "misses some short stops" [README.md]. Stop judgements were not
  measured [CLAUDE.md §15], and no script in `eval/` measures turn detection.
- **Evidence:**
  - none of the 4 sample lessons passes a stop sign [eval/ground_truth/sample_signs.csv];
  - the held-out videos contain only 9 labelled stop-sign boxes [eval/results/detector_eval.md].
  
  So the stop check is exercised only by the rule-engine tests [tests/fixtures/rule_scenarios/].
- **Next:** label stops and turns on full lessons and measure them.

## 2. Which signs we can recognise

### Signs missing from the training data

- **What happens:** the detector can only find signs it was trained on.
  - It cannot see:
    - no left turn (3.18.2);
    - no U-turn (3.19);
    - start and end of a built-up area (5.23.x, 5.24.x).

    All have 0 examples in the dataset [eval/results/dataset_inventory.md].
  - End of speed limit (3.25) has only 5 examples, all from one video. That video is one of our
    held-out videos, so the detector never saw this sign in training (AP50 0.0
    [eval/results/detector_eval.md]).
  - Other signs are thin [eval/results/dataset_inventory.md]:
    - end of all restrictions (3.31): 133 examples;
    - left-only arrow (4.1.3): 6 videos;
    - right-or-left arrow (4.1.6): 104 examples.
- **Consequence:**
  - We can never flag an illegal left turn or U-turn under a ban sign.
  - We cannot tell when we enter or leave a town (see §3).
  - In practice a speed limit is not ended by an "end of limit" sign. It lasts until:
    - another limit sign;
    - an end-of-all-restrictions sign;
    - our next turn [config/rules/rules.json].
- **Next:** record and label Uzbek dashcam footage that contains these signs.

### A sign the detector never saw is read as the nearest sign it knows

- **What happens:** the detector always answers with one of the classes it was trained on.
- **Evidence:**
  - a "no trucks" sign (3.4) became a 100 km/h limit in the Islam Karimov street sample at 1:54
    [eval/results/sample_signs.md];
  - "no pedestrians" (3.10) was read as 70 km/h [README.md].
- **Why:** most likely, a red-ringed round sign that the detector has never seen ends up as the
  closest red-ringed round sign it knows. Both examples above became speed limits.
- **Next:** more sign classes in training, and a way to answer "unknown sign".

### The back of a sign can be read as a sign

- **Evidence:** at 0:44 in the Islam Karimov street sample, the grey back of a round sign was shown as
  "right turn only" (4.1.2) [eval/results/sample_signs.md].
- **Why:** the grey back has the same round outline as a real sign, and nothing checks the colours
  inside the box.
- **Next:**
  - a colour check: blue signs must contain blue and red-ringed signs red. We designed it but did not
    build it in time [README.md];
  - backs of signs as negative training examples.

### Speed-limit digits are sometimes misread

- **Evidence:**
  - an 80 km/h sign in a winter video was read as 60 or 40 [CLAUDE.md §17, M6];
  - a 30 was read as 50 on a Tashkent clip [README.md];
  - on held-out videos, 80 km/h signs score AP50 0.17 and recall 0.20 [eval/results/model_comparison.md];
  - on held-out frames, only 9% of the detector's "60 km/h" boxes (before voting) are really 60 km/h
    signs [eval/results/model_comparison.md].
- **Where our model is worse than the pretrained European one:** on average our fine-tuned model is
  better (mean AP50 0.81 vs 0.65), but not everywhere [eval/results/model_comparison.md]:

  | Sign | Ours (AP50) | Pretrained (AP50) |
  |---|---:|---:|
  | 80 km/h | 0.17 | 0.66 |
  | Left-only arrow (4.1.3) | 0.64 | 0.86 |
  | Roundabout (4.3) | 0.61 | 0.79 |
  | 30 km/h | 0.52 | 0.72 |
  | 10 km/h | 0.27 | 0.45 |
- **What already helps:**
  - A sign must be seen in at least 3 frames [config/settings.yaml].
  - Close-up views count more in the vote than distant ones. This fixed the one digit misread in our
    samples: Alisher Navoi street at 1:35 was read as 60 and now shows the correct 50
    [eval/ground_truth/sample_signs.csv].
- **Next:**
  - more training examples of these values;
  - a second step that reads the digits from the cropped sign.

### Confidence does not tell right from wrong

- **Evidence:** on the demo clips, misread speed limits had confidence 0.54–0.73. Correct readings had
  0.55–0.81 [README.md]. The ranges overlap, so no threshold can filter the misreads out.

### How often the detector misses, per frame

- **Evidence:** on the 3,027 held-out frames, all 89 classes: precision 0.714, recall 0.652, mAP50 0.699
  [eval/results/detector_eval.md].
- At the detector's low first-stage threshold, before voting, those frames contain
  [eval/results/failure_gallery/README.md]:
  - 797 missed or misread signs;
  - 1,026 false alarms (some may be real signs that the dataset did not label).
- These are per-frame numbers. How many physical signs we miss over a whole lesson was not measured.

## 3. Rules without a map

### "Until the next intersection" is approximated

- **What happens:** officially, a speed limit applies until the nearest intersection after the sign, and
  a turn ban applies to the next intersection [config/rules/rules.json, official texts]. We do not
  know where intersections are, so:
  - a speed limit ends at our next turn;
  - a turn ban ends at our next turn, or after 300 m or 60 s [config/settings.yaml `ban_scope`].
- **Consequence:** when we drive straight through an intersection, a lower limit stays in force after it
  officially ended. That can produce a "possible speeding" event that is not real.
- **Next:** junctions from OpenStreetMap.

### No entry

- **What happens:**
  - A no-entry sign at the side of the road usually closes the side road it stands at. It is reported
    only if we turned toward it within −3…8 s of passing it. An overhead sign always counts
    [config/settings.yaml `no_entry`].
  - It is always "possible", never a violation.
- **Consequence:** driving straight into a one-way street past a side sign is missed [README.md].
- **Next:** one-way streets from a map.

### We assume a built-up area everywhere

- **What happens:** the detector cannot see town start/end signs, so we always assume a built-up area:
  60 km/h by default, not 90 [config/rules/rules.json]. Speeding against an assumed default limit is
  only "possible" [CLAUDE.md §11.4].
- **Uzbekistan:** the Uzbek rules allow 60 km/h in Tashkent, Nukus and regional and district centres,
  which matches us. Other settlements allow 70 km/h, so there we would be 10 km/h too strict [CLAUDE.md,
  start-here section].
- **Next:** town boundaries from a map, plus training data for the settlement signs.

### Plates that limit a sign are not seen

- **What happens:** a small plate under a sign can limit it, for example to trucks or to certain hours.
  These plates (8.4.x, 8.5.x) have 0–2 examples in the training data [CLAUDE.md §11.1], so in practice the
  detector does not find them. A sign that only applies to trucks or at certain hours is treated as applying to us.

### The rule texts are Russian

- **What happens:** the sign meanings, zones and default limits are copied word for word from the
  Russian traffic rules (ПДД РФ, edition of 28.08.2026). A Russian-speaking team member checked them
  [config/rules/rules.json, docs/RULES_CHECKLIST.md]. The English texts are draft translations.
- **Why:** our training data and its sign numbers are Russian. Uzbek signs use the same numbering
  [CLAUDE.md].
- **Next:** for a launch in Uzbekistan, the official Uzbek rules (lex.uz).

### The score is a simple formula

- **What happens:** score = 100 − 10 per violation − 3 per "possible" event [config/settings.yaml
  `score.weights`]. The app labels it "indicative". No instructor has validated it.

## 4. Which signs apply to us

- **What happens:** signs on poles beside other lanes or side roads must be ignored. We decide this with
  rules of thumb [config/settings.yaml `relevance`]:
  - the side of the road, from how the sign drifts sideways as we approach (at least 10% of the image
    width);
  - whether a round or square sign faces us: one much narrower than tall (width/height below 0.6)
    faces a side road;
  - whether we approach it: it must grow on screen.
- **Evidence:** these rules can misjudge signs on wide roads, slip roads and sharp bends [README.md].
  Their accuracy was not measured. Our sample check covers only the signs we judged to apply to us, so
  a sign wrongly dismissed as "not ours" is not counted [eval/results/sample_signs.md].
- **Next:** understanding the lane layout, plus a map.

## 5. Processing speed and cost

- **What happens:** processing takes 95.0 s per minute of video, on average over 6 clips
  [eval/results/benchmark.md]:
  - 1.45–1.74× the video's length;
  - on one RTX 3070 Ti;
  - our target was real time [CLAUDE.md §12].
- **Example:** the 2-minute Islam Karimov street clip took 198.39 s [eval/results/benchmark.json]:

  | Stage | Seconds |
  |---|---:|
  | Ingest | 10.15 |
  | Analysis | 92.04 |
  | Rendering the blurred video | 93.61 |
  | Signs and rules | 0.07 |
- **Energy:**
  - the GPU drew 56.2 W on average, which is 0.089 kWh per lesson-hour. The CPU's power is not included
    [eval/results/benchmark.md];
  - we have no electricity price, so there is no money figure (`cost.*` is empty
    [config/settings.yaml]).
- **Limits of the demo set-up** [config/settings.yaml]:
  - one worker handles one upload at a time, and at most 20 uploads wait in the queue;
  - uploads are limited to 3 minutes and 1.5 GiB, which is far shorter than a real lesson;
  - a job is stopped after 15 minutes.
- **Next:**
  - faster inference;
  - analysis and rendering in parallel;
  - fewer analysed frames per second;
  - more workers or GPUs.

## 6. Privacy

- **The automatic blur can miss.** For example, our face detector misses small and side-on faces
  [CLAUDE.md §10.9], and the plate detector missed a dirty van's plate on a Moscow clip [CLAUDE.md §14.3].
  We cover this in two ways:
  - every detected person's head area is blurred;
  - a fallback area over the lower part of every detected vehicle (except our own bonnet) is pixelated
    [config/settings.yaml `privacy`].

  A person or vehicle that the detectors do not find is not blurred.
- **Manual check:**
  - We check every sample lesson by eye, but as a spot check: frames spread over the video plus every
    thumbnail [eval/privacy_gallery.py]. It does not cover every frame.
  - Uploaded lessons are not checked by a person.
- **Faces on posters and billboards are not blurred.** You can see some in the Alisher Navoi street
  sample. They are printed images, not people in the street.
- **Our own car's bonnet is sometimes pixelated.** This is harmless. The rule that skips the bonnet only
  fires when the detected "vehicle" is wide, low and touches the bottom of the image
  [config/settings.yaml `privacy.own_hood`].

## 7. What we measured, and what we did not

| Measured | Result | Source |
|---|---|---|
| Sign detector on 3,027 held-out frames, all 89 classes | precision 0.714, recall 0.652, mAP50 0.699 | [eval/results/detector_eval.md] |
| Our detector vs a pretrained European model, on signs both know | mean AP50 0.81 vs 0.65 | [eval/results/model_comparison.md] |
| Signs shown in the 4 sample lessons | 66 of 68 right. Precision only. Claude Code did a first pass, and the team confirmed every verdict | [eval/results/sample_signs.md], [eval/ground_truth/README.md] |
| Our tracker vs ByteTrack, 12 clips | 254 vs 166 signs kept. Counts only, with no ground truth | [eval/results/tracker_comparison.md] |
| How early a sign is announced | median 2.1 s before passing; 85% at least 1 s ahead, 54% at least 2 s | [eval/results/tracker_comparison.md] |
| Speed from video | one test clip, 36 moments | [eval/results/speed_eval.md] |
| Processing time and energy | 6 clips | [eval/results/benchmark.md] |

Not measured:
- How many signs we miss over a whole lesson [CLAUDE.md §15].
- Whether the speed limit in force is right over time [CLAUDE.md §15].
- Whether stop judgements are right [CLAUDE.md §15].
- Which-sign-applies-to-us decisions, turn detection, and no-entry and turn-ban judgements.
- Speed at city speeds, and speed on the Tashkent clips, for which we have no true speed.
- Night, rain and dirty windscreens. The app warns when video is dark or blurry [config/settings.yaml
  `quality`], but the sample lessons are all daytime [CLAUDE.md §17, M8].
- Use by a real driving school: no instructor has tried it yet.

## 8. Product scope

- **Upload only.** It works after the lesson, with no live in-car mode [CLAUDE.md §1]. In our videos, 85%
  of the signs that apply to us were confirmed at least 1 s before we passed them, and 54% at least
  2 s before [eval/results/tracker_comparison.md]. A live system would warn late for the rest.
- **Video is the only input.** There is no GPS, no map and no data from the car [CLAUDE.md].
- **English only**, both the interface and the voice. Driving schools in Uzbekistan need Uzbek and
  Russian [CLAUDE.md].
- **Demo footage is not student lessons.** It is Tashkent driving-tour videos from YouTube, used with
  the owners' consent, plus Moscow dashcam clips [DATA_SOURCES.md].
- **One machine.** The site runs on one team computer (RTX 3070 Ti) published through Tailscale Funnel
  [CLAUDE.md §4, §16].
- **Licence.** The code is AGPL-3.0 because it builds on Ultralytics. A commercial launch needs an
  Ultralytics Enterprise licence or a permissively licensed detector [README.md].

## What we would build next (most impact first)

1. **Trustworthy speed.** Take speed from the car, the phone's GPS or the dashcam's own speed overlay,
   and calibrate per camera. Then speeding can be a definite verdict instead of "possible".
2. **Uzbek training data.** Label Tashkent dashcam footage that includes the missing signs (no left turn,
   no U-turn, town start/end, end of limit, no trucks) and the backs of signs. Add the colour check.
3. **Full-lesson ground truth.** Label every sign, limit and stop in a set of real lessons. Then measure
   what we miss (recall) and the limit in force over time.
4. **Map data** (OpenStreetMap): intersections, one-way streets and town boundaries, for exact sign
   zones and no-entry.
5. **Uzbek rules**, and an Uzbek and Russian interface and voice.
6. **Faster and cheaper processing**, to reach real time and handle full-length lessons.
7. **Live in-car prompts.**
