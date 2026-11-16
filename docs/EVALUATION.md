# Evaluation

A smoke detector is judged by two questions an operator would ask:

1. **How many minutes after ignition does it alert?**
2. **How often does it wake me up for nothing?**

Frame accuracy answers neither, so it is reported only for comparison with prior work.

## 1. Inputs

For each sequence the harness has, in time order: `timestamps_s`, per-frame `scores`
from a scorer, per-frame `labels`, and (if known) the ignition time, defined as the
timestamp of the first frame with `label == 1`.

Scores become alerts through `AlertPolicy` (see `ARCHITECTURE.md` §8). Everything below
is computed from alert events, except the frame-level metrics.

## 2. Frame-level metrics (comparability only)

Binary predictions are `score >= threshold`. Accuracy, precision, recall and F1 use the
usual definitions; a metric with a zero denominator is reported as `nan`, never 0.

## 3. Time-to-detection (TTD)

For one fire sequence with ignition time `t0`:

- **Detected** if at least one alert fires at time `t >= t0`.
  `TTD = (t_first_alert_at_or_after_t0 - t0) / 60` minutes.
- **Missed** if no alert fires at or after `t0` before the sequence ends.
- An alert before `t0` is a **false alarm** (§4). It does not count as a detection, and
  because of the cooldown it can delay the real one. That cost is intended.

Reported per experiment:

| Name | Definition |
|---|---|
| `detection_rate` | detected fires / fires |
| `ttd_median_detected` | median TTD over detected fires |
| `ttd_mean_detected` | mean TTD over detected fires |
| `ttd_mean_penalised` | mean TTD over **all** fires, with a missed fire counted as the time from `t0` to the end of its sequence |
| `ttd_p90_penalised` | 90th percentile of the same penalised values |

`ttd_mean_detected` is the closest match to the number in the SmokeyNet paper, which
averages over fires and does not spell out how misses are handled. The penalised
figures are the honest headline: a model cannot improve them by staying silent.

Resolution caveat: FIgLib has roughly one frame per minute, so TTD on it moves in
steps of about a minute and differences under a minute are within sampling noise.

## 4. False alarms per camera-day

- A **false alarm** is an alert event fired on a frame with `label == 0` in footage
  where no smoke is visible yet (before `t0`, or anywhere in a sequence with no fire).
- **Negative exposure** is the total duration of that footage, in camera-days. For one
  sequence it is the sum of gaps between consecutive negative frames, with each gap
  capped at `max_gap_s` (default 180 s) so a hole in the recording is not counted as
  watched time.
- `false_alarms_per_camera_day = false_alarms / negative_exposure_days`.

Small-sample caveat: FIgLib gives about 40 minutes of pre-ignition footage per fire, so
the 62 test fires add up to under two camera-days. The rate is therefore reported with
an exact Poisson confidence interval, and every table states the exposure it rests on.
A real deployment claim needs continuous negative footage, which is future work unless
a suitable source turns up in Phase 0.

## 5. Operating curve

No single threshold is "the" result. `operating_curve` sweeps the threshold (and
optionally `k`, `n`) and returns one row per setting with false alarms per camera-day,
detection rate and the TTD statistics. Models are compared **at matched false-alarm
rate**: the README headline is TTD at a fixed false-alarm budget chosen on validation.

Thresholds and policy settings are always chosen on validation data and then frozen
before the test set is touched.

## 6. Uncertainty

Fires are the unit of resampling. `bootstrap_over_sequences` resamples sequences with
replacement and returns percentile intervals for any metric. Model-to-model
differences are reported as paired bootstrap differences over the same fires.

## 7. Protocols

| ID | Train | Test | Purpose |
|---|---|---|---|
| P1 | FIgLib official train fires (144) | FIgLib official test fires (62) | Comparable to published numbers |
| P2 | FIgLib, all stations but a held-out group | held-out stations | Unseen-tower generalisation |
| P3 | FIgLib (US) | Pyro-SDIS (France) | Unseen camera network and landscape |

P2 uses `leave_groups_out_folds` on `station_id`. Before any P1 result is quoted,
`leakage_report` is run on the official split to count cameras and stations shared
between train and test; that count goes in the README next to the P1 numbers.

## 8. Published reference numbers (FIgLib test, P1)

From Dewangan et al., *FIgLib & SmokeyNet* (Remote Sensing, 2022), averages of five runs:

| Model | Acc | F1 | Precision | Recall | TTD (min) |
|---|---|---|---|---|---|
| SmokeyNet (ResNet34 + LSTM + ViT, 2 frames) | 83.49 | 82.59 | 89.84 | 76.45 | 3.12 |
| ResNet34 + ViT (1 frame) | 82.53 | 81.30 | 88.58 | 75.19 | 2.95 |
| ResNet34 (1 frame) | 79.40 | 78.90 | 81.62 | 76.58 | 2.81 |
| Faster R-CNN (1 frame) | 71.56 | 66.92 | 81.34 | 56.88 | 5.01 |

Read as transcribed from the arXiv v3 text; re-check against the journal PDF in Phase 0
before quoting them in the README. Note that the paper reports no false-alarm rate per
unit of time and no unseen-camera test, which is the gap this project aims at.

## 9. Sanity baselines

Every results table includes three rows that cost nothing and catch harness bugs:

- **Never alert**: detection rate 0, zero false alarms.
- **Always alert**: a false alarm at the start of every sequence and again each time the
  cooldown expires; fires are "detected" only as fast as the cooldown allows.
- **Oracle**: score = label. TTD 0, zero false alarms.
