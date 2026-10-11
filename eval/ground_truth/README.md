# Ground truth formats (CLAUDE.md §15)

Ground truth is created by **people** from the team's collected dashcam clips — never by the
system and never by an AI model (one exception, stated in its section: `sample_signs.csv`). Clips used to tune or calibrate something are named as such and
are not used to report accuracy for that thing. One CSV per clip, named `<kind>_<clip>.csv`,
where `<clip>` is the short clip name (e.g. `ru_02`). Times are seconds from the start of the video.

## speed_<clip>.csv — on-screen GPS speed

Only for clips whose dashcam shows its GPS speed in the picture. Read it every 5 s
(`~/signcoach-data/eval_sheets/speed_sheet_<clip>.png` shows the display crops with their times).

```
t_seconds,speed_kmh
0,44
5,44
```
Calibration clip: `ru_01` (`scripts/calibrate_speed.py`). Test clip: `ru_02` (`eval/eval_speed.py`).

## signs_<clip>.csv — every road sign the car passes

One row per physical sign, at the moment the car passes it.

```
t_seconds,sign_code,value,side,relevant,notes
12.4,3.24,60,right,y,
31.0,5.19.1,,right,y,
40.2,3.27,,left,n,for the side road
```
`sign_code` uses the Russian GOST numbers of `config/rules/signs.json`; `value` only for speed limits;
`side` = right | left | overhead; `relevant` = y if the sign applies to our car.

## limit_<clip>.csv — the speed limit in force

```
t_start,t_end,limit_kmh
0,35.5,60
35.5,180,40
```

## stops_<clip>.csv — stop signs

```
t_seconds,full_stop
88.0,n
```

## sample_signs.csv — what each sign shown in a sample lesson really is

A check of what the 4 public sample lessons show, not a list of every sign in their videos. One row
per sign a sample shows as applying to our car, at the moment we pass it (`t_pass` in the report);
`sign_code`/`value` say what the sign really is, `none` when there is no sign (e.g. the back of a
sign). Made from crop sheets (one crop per shown sign, at its best frame): Claude Code did a first pass
and the team confirmed every verdict (Sunday 11 Oct 2026, 09:30). `eval/eval_sample_signs.py` compares
the current samples with it (precision only).

```
sample,t_seconds,sign_code,value,notes
"Tashkent, Islam Karimov street",44.6,none,,the grey back of a round sign; we show 4.1.2 (right turn only)
"Tashkent, Alisher Navoi street",95.5,3.24,50,
```
