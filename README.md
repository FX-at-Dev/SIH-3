# Indian Road Perception + Driving Recommendation System

A perception and decision-support prototype for Indian road environments. It
analyses a live camera feed, a recorded video or a still image, detects road
users and hazards, tracks them across frames, estimates where they are and
where they are going, and emits a single driving recommendation:

```
CRUISE      NUDGE       YIELD       CREEP       STOP
```

> ### Safety notice
>
> **This is a research prototype for perception and decision support. Its
> output is advisory information for a human or a downstream planner. It is
> not connected to vehicle actuation, has not been validated for use on public
> roads, and must not be used to control a real vehicle.**
>
> Distances are **monocular estimates** derived from image geometry and assumed
> object sizes — not sensor measurements. They are displayed with a `~` suffix
> throughout. All thresholds are prototype defaults, not safety standards.

---

## Contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [The five recommendations](#the-five-recommendations)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [Testing](#testing)
- [Extending the system](#extending-the-system)

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py --source camera
```

The stock YOLO weights (`models/yolo11n.pt`) download automatically on first
run. Try it without a camera:

```bash
python main.py --source image --input data/samples/bus.jpg --save
python scripts/demo_scenarios.py
```

### Keyboard controls

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| `Q` / `Esc` | quit | `S` | save screenshot |
| `R` | reset tracking | `D` | toggle detection details |
| `Space` | pause / resume | `M` | toggle dashboard panel |
| `T` | toggle trajectories | `A` | toggle drivable-area overlay |
| `H` | print help | | |

---

## What it does

```
 frame
   │
   ├─► Detection            YOLO, or a custom Indian-road model
   ├─► Pothole stage        separate model or CV heuristic (clearly labelled)
   ├─► Tracking             Kalman + IoU, persistent IDs across frames
   ├─► Distance             monocular estimate, always marked as an estimate
   ├─► Road position        left / centre / right, in-path / crossing / beside
   ├─► Prediction           constant-velocity, 3 s horizon, path-entry time
   ├─► Risk                 per-object and whole-scene, with reasons
   └─► Decision  ──────────►  CRUISE / NUDGE / YIELD / CREEP / STOP
                              + target speed, risk level and a written reason
```

Everything downstream of detection depends on **tracking**: without persistent
IDs there is no velocity, no time-to-collision and no trajectory prediction.
That is why tracking is not optional in the default configuration.

### Object classes

| Group | Classes |
| --- | --- |
| Road users | `car` `truck` `bus` `motorcycle` `bicycle` `auto_rickshaw` `person` |
| Animals | `cow` `bull` `dog` `horse` `sheep` `other_animal` |
| Indian-specific | `auto_rickshaw` `bullock_cart` `pushcart` |
| Hazards | `pothole` `road_damage` `speed_bump` `debris` `construction_barrier` `traffic_cone` `road_blockage` |
| Signage | `traffic_light` `traffic_sign` |
| Fallback | `unknown` |

Adding a class requires two lines in `src/core/classes.py` — one alias, one
category — and nothing else in the pipeline changes.

**Important:** the stock COCO model detects only about half of these. See
[Known limitations](#known-limitations) and [`models/README.md`](models/README.md).

---

## The five recommendations

| Action | Meaning | Typical trigger |
| --- | --- | --- |
| **CRUISE** | Continue normally | No conflict with the predicted path, adequate confidence |
| **NUDGE** *(LEFT/RIGHT)* | Small lateral adjustment, keep moving | Partial obstruction with verified space beside it |
| **YIELD** | Slow and give priority | A road user entering or occupying the path |
| **CREEP** | Move very slowly | Dense traffic, poor confidence, constrained passage |
| **STOP** | Full stop | Critical collision risk, blocked path, no safe trajectory |

Two rules sit above the hierarchy:

1. **Safety dominates progress.** Where the evidence is ambiguous, the more
   conservative action wins.
2. **CREEP is never a substitute for STOP.** Crawling is the response to
   *uncertainty*; a critical collision risk always produces STOP.

### Direction is earned, not guessed

A NUDGE direction is only offered when that side is estimated to be genuinely
clearer *and* the vehicle physically fits through the resulting gap:

```
usable width = gap beside the obstacle inside the corridor
             + free space outside it (only if that side is clear,
               capped at half a lane)

required     = vehicle_width_m + nudge_clearance_m      # 1.8 + 0.6 = 2.4 m
```

If neither side qualifies, no nudge is offered and a more conservative action
takes over.

### Two different overlap measures

A distinction that matters, and that a naive implementation gets wrong:

| Measure | Question | Used for |
| --- | --- | --- |
| `ego_path_overlap` | *How much of the object is in my lane?* | Risk |
| `corridor_obstruction` | *How much of my lane does it block?* | Avoidance |

A narrow pothole squarely in a wide lane scores ~1.0 overlap but only ~0.25
obstruction: fully in the path, yet easily driven around. A bus in the same
position scores high on both. Conflating the two makes the system either stop
for potholes or drive into buses.

### Explanations are generated, never canned

Every recommendation carries a reason built from the actual world state:

```
STOP        Critical collision risk: pedestrian (ID 4) - TTC 1.2s.
            No safe avoidance path is available on either side.

NUDGE LEFT  Pothole (ID 1) at an estimated 4.6 m occupies the centre of the
            predicted path (31% blocked). Space is estimated to be available
            on the left; a full stop is not required.

YIELD       Pedestrian (ID 2) at an estimated 16.0 m is approaching the ego
            corridor and is predicted to enter it in 1.8 s. Reducing speed to
            give priority.

CREEP       Proceeding at very low speed: dense mixed traffic (12 objects:
            8 pedestrian(s), 4 vehicle(s), 0 animal(s)). The path is
            technically open but conditions are not confidently readable.
```

---

## Architecture

```
indian-road-ai/
├── main.py                     CLI entry point
├── configs/default.yaml        every threshold lives here
├── models/README.md            custom-model guide
│
├── src/
│   ├── core/
│   │   ├── types.py            shared contract: Detection, TrackedObject, ...
│   │   ├── config.py           Pydantic-validated YAML config
│   │   └── classes.py          label -> canonical class -> semantic category
│   │
│   ├── input/                  camera / video / image behind one interface
│   ├── perception/             detector, pothole, tracker, drivable area
│   ├── geometry/               ego path, distance, road position
│   ├── prediction/             constant-velocity trajectory prediction
│   ├── risk/                   TTC, collision analysis, risk engine
│   ├── decision/               the five-state decision engine
│   ├── visualization/          renderer, dashboard, overlays
│   ├── logs/                   event logging and session recording
│   └── app.py                  pipeline orchestration
│
├── scripts/
│   ├── demo_scenarios.py       ten built-in scenarios
│   ├── evaluate_dataset.py     batch evaluation over a dataset
│   └── export_results.py       session reports and charts
│
└── tests/                      145 unit tests
```

Each stage sits behind an interface — `BaseDetector`, `BaseTracker`,
`BaseDistanceEstimator`, `BasePredictor`, `DrivableAreaDetector`,
`BasePotholeDetector`, `InputSource` — so `src/app.py` never learns that the
detector happens to be YOLO or the tracker happens to use a Kalman filter.

> The package is `src/logs/` rather than `src/logging/`, to avoid any ambiguity
> with the standard library's `logging` module.

### Fault isolation

A single bad frame never stops a run. Detection, tracking, risk, decision and
rendering are each isolated; a failure degrades that frame and is logged.
A failure in the decision engine **fails safe**, emitting STOP with the reason
stated, rather than failing open.

---

## Usage

### Live camera

```bash
python main.py --source camera
python main.py --source camera --camera-id 1
```

### Video

```bash
python main.py --source video --input data/videos/road.mp4
python main.py --source video --input road.mp4 --save-results
python main.py --source video --input road.mp4 --loop
```

Supports MP4, AVI, MOV, MKV and anything else the local OpenCV build decodes.
Corrupt frames mid-file are skipped rather than ending playback.

### Image

```bash
python main.py --source image --input data/images/street.jpg
python main.py --source image --input street.jpg --save
python main.py --source image --input data/images/ --recursive   # a directory
```

Image mode prints a full analysis table and the recommendation with its
explanation. Tracking and velocity are unavailable from one still frame, so no
TTC is reported — the system says `--` rather than inventing a number.

### Scenario demos

```bash
python scripts/demo_scenarios.py                      # all ten
python scripts/demo_scenarios.py --scenario cow_blocking --show
python scripts/demo_scenarios.py --save               # annotated frames
```

| Scenario | Recommendation |
| --- | --- |
| `clear_road` | CRUISE |
| `pothole` | NUDGE LEFT |
| `pedestrian_crossing` | YIELD, then STOP as the pedestrian enters the path |
| `cow_blocking` | STOP |
| `bullock_cart` | NUDGE RIGHT |
| `slow_vehicle` | YIELD |
| `motorcycle_overtaking` | YIELD |
| `dense_market` | CREEP |
| `multiple_obstacles` | YIELD |
| `unknown_obstacle` | STOP |

These run on the **mock detector**: the detections are synthetic and labelled
as such on screen. They exercise the risk and decision layers deterministically
without needing footage of a cow standing in a road.

### Batch evaluation

```bash
python scripts/evaluate_dataset.py --input data/
python scripts/evaluate_dataset.py --input data/potholes --pothole-mode heuristic
```

Suggested dataset layout:

```
data/
├── village/   urban/   market/
├── highway/   cattle/  potholes/
```

The report shows the recommendation distribution, risk distribution, detected
classes and timing per scenario folder. It measures **what the system
recommended**, not accuracy against ground truth — there is no labelled ground
truth here. Use it as a behavioural regression report when changing a config
or a model.

### Session reports

```bash
python main.py --source video --input road.mp4 --save-results
python scripts/export_results.py --latest --plots
```

```
results/session_2026_09_13_214233/
├── annotated_video.mp4    events.json        detections.csv
├── recommendations.csv    metrics.csv        summary.json
├── session_report.md      screenshots/
└── recommendation_timeline.png  risk_distribution.png  latency.png
```

---

## Configuration

Everything tunable is in `configs/default.yaml` and validated on load, so a
typo fails immediately instead of producing a subtly wrong risk assessment ten
minutes into a video.

```yaml
detector:
  type: yolo                  # yolo | mock
  model: models/yolo11n.pt
  confidence: 0.35

risk:
  ttc:
    medium: 5.0               # seconds
    high: 3.0
    critical: 1.5
  priority:                   # configurable policy, NOT a safety standard
    VULNERABLE: 1.00
    ANIMAL: 0.95
    VEHICLE: 0.70

decision:
  cruise_speed_kmh: 30
  yield_speed_kmh: 12
  creep_speed_kmh: 5
  hysteresis_frames: 3        # suppress flicker; escalation bypasses this
```

Common CLI overrides, no file editing needed:

```bash
python main.py --source camera --confidence 0.5 --device cpu
python main.py --source video --input road.mp4 --pothole-mode heuristic
python main.py --source image --input x.jpg --detector mock --scenario dense_market
```

### Camera calibration

Distance quality depends on the camera parameters. The defaults assume a
nominal 65° horizontal field of view and a 1.25 m mount height. If you know
yours, set them:

```yaml
camera:
  horizontal_fov_deg: 78.0
  mount_height_m: 1.40
  pitch_deg: 3.0
```

Estimates remain labelled as estimates regardless — `calibrated: true` only
records that the values are yours rather than guesses.

---

## Known limitations

Stated plainly, because a decision-support system that overstates its own
reliability is worse than no system.

### 1. The stock model cannot detect the hazards that matter most

`yolo11n.pt` is COCO-trained. COCO has **no pothole, auto-rickshaw, bullock
cart, pushcart, debris, traffic cone or barrier class**. With stock weights
these are never detected at any threshold, and an auto-rickshaw is typically
reported as `car` or `truck`.

The system announces this at startup rather than leaving it to be inferred:

```
NOT DETECTABLE with these weights: pothole, road_damage, speed_bump,
auto_rickshaw, bullock_cart, pushcart, debris, construction_barrier,
traffic_cone, road_blockage, bull
```

Fixing this requires custom-trained weights — see
[`models/README.md`](models/README.md).

### 2. Distances are estimates, not measurements

A single camera cannot measure range. Two independent methods are fused —
known-object-size (pinhole) and ground-plane projection — but both rest on
assumptions: that the object is a typical size, that the road is flat, that the
object touches the ground, that the field of view is as configured.

Expect errors on the order of **15–30%** with uncalibrated intrinsics, worse
for unusual or partially occluded objects. Every value is marked `~` and
`distance_is_estimated=True` throughout the codebase.

### 3. Pothole detection is off by default

The default is `pothole.type: none`, and the dashboard reads
**"POTHOLE DETECTION OFF (no model configured)"**. The alternative is a
classical-CV heuristic that fires on shadows, wet patches, tar repairs and
manhole covers. It is available (`--pothole-mode heuristic`), capped at 0.45
confidence, tagged `HEUR (not ML)` on screen, and never presented as a model
detection.

### 4. Prediction is constant-velocity

Objects are assumed to keep their current speed and heading. Beyond roughly two
seconds this is weak, particularly for pedestrians and animals, which is why
prediction confidence decays with horizon length and is reduced for those
categories. The decision engine never relies on prediction alone.

### 5. The ego path is a geometric prior, not a detected lane

The corridor is a fixed trapezoid. This is deliberate — Indian roads frequently
lack usable lane markings — but it means the corridor does not follow curves,
does not adapt to road width, and assumes a roughly forward-facing camera.

### 6. No sensor fusion, no absolute positioning

Monocular camera only. No LiDAR, radar, stereo, IMU or GPS. The system reasons
in image space and makes no claim about world coordinates.

### 7. Not validated

There is no ground-truth evaluation, no benchmark against a reference system,
and no field validation. The thresholds are engineering defaults chosen to
produce sensible demo behaviour.

---

## Testing

```bash
python -m pytest tests/ -q          # 145 tests
python -m pytest tests/ -v          # verbose
```

| File | Covers |
| --- | --- |
| `test_ttc.py` | TTC maths, and every path that must yield no value rather than `inf`/`NaN` |
| `test_tracker.py` | ID persistence, occlusion survival, cross-category protection |
| `test_risk.py` | Risk bands, path overlap vs obstruction, priorities, confidence |
| `test_decision.py` | The scenario-to-recommendation contract for all five states |
| `test_distance.py` | Monotonicity, bounds, honesty of the estimate labelling |
| `test_prediction.py` | Path-entry timing, confidence decay, no lane assumption |
| `test_pothole.py` | Severity scaling, and that heuristics can't pass as ML |

The tests run objects through the **real** geometry analyser, so they cannot
pass against a shortcut the live pipeline never takes.

Two bugs these tests caught during development, both of the kind that matters:

- A pedestrian overlapping a parked car was absorbed into the car's track and
  reported as a car. Cross-category association is now forbidden outright.
- A crossing pedestrian was scored CRITICAL because "crossing the path" and
  "predicted to enter the path in 1.8 s" were counted as two separate risks —
  the same fact double-counted, pushing ordinary yield situations into the
  stop band.

---

## Extending the system

Each seam is an interface with a single obvious implementation point.

| To add | Implement | In |
| --- | --- | --- |
| A depth camera / LiDAR | `BaseDistanceEstimator` | `src/geometry/distance.py` |
| A learned trajectory model | `BasePredictor` | `src/prediction/predictor.py` |
| Road segmentation | `DrivableAreaDetector` | `src/perception/drivable_area.py` |
| A different tracker | `BaseTracker` | `src/perception/tracker.py` |
| An RTSP or ROS source | `InputSource` | `src/input/base.py` |
| A web UI | consume `WorldState` | `src/core/types.py` |

A real range sensor reports `provides_true_measurement = True`, and the `~`
estimate marker disappears from the UI automatically — the honesty of the
display is a property of the data, not of the renderer.

ByteTrack is already wired behind the tracker interface:

```bash
pip install supervision
```
```yaml
tracking:
  type: bytetrack
```

Planned extension points, in rough order: sensor fusion, occupancy grids,
local path planning, CARLA integration for closed-loop evaluation.

---

## Environment notes

Developed and tested on Python 3.14, Windows 11, **CPU-only** (no CUDA
available on the development machine).

| Configuration | Throughput |
| --- | --- |
| YOLO11n @ 640, CPU | ~5-12 fps |
| YOLO11n @ 480, 640 px frames, CPU | ~12-20 fps |
| Mock detector (pipeline only) | ~40-160 fps |
| Any of the above on a CUDA GPU | 30+ fps |

Detection dominates the frame budget; tracking, risk and decision together cost
well under a millisecond.

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Licence

MIT. See [`models/README.md`](models/README.md) for dataset and model-weight
licensing, which is separate and must be checked independently.
