# Models

This directory holds the detection weights the system loads at runtime.

---

## 1. What the stock model can and cannot detect

The default configuration uses `models/yolo11n.pt`, an Ultralytics YOLO
checkpoint trained on **COCO**. It is a good general object detector and a poor
Indian-road detector, and it is important to be precise about which is which.

**COCO contains** (and this system therefore detects out of the box):

| Canonical class | COCO label |
| --- | --- |
| `person` | person |
| `bicycle` | bicycle |
| `car` | car |
| `motorcycle` | motorcycle |
| `bus` | bus |
| `truck` | truck |
| `cow` | cow |
| `dog`, `cat`, `horse`, `sheep`, `elephant` | (same) |
| `traffic_light` | traffic light |
| `traffic_sign` | stop sign |

**COCO does NOT contain** any of the following. With stock weights these are
**never detected, at any confidence threshold**:

```
pothole            road_damage        speed_bump
auto_rickshaw      bullock_cart       pushcart
debris             construction_barrier
traffic_cone       road_blockage      bull
```

The application states this explicitly at startup rather than leaving you to
infer it from an absence of detections:

```
Detector class coverage (80 model classes):
  NOT DETECTABLE with these weights: pothole, road_damage, speed_bump,
  auto_rickshaw, bullock_cart, pushcart, debris, construction_barrier,
  traffic_cone, road_blockage, bull
  These classes need custom-trained weights - see models/README.md.
```

An auto-rickshaw in frame will usually be reported as `car` or `truck`, because
that is the closest thing COCO knows about. That is a real limitation, not a
bug to be worked around by relabelling.

---

## 2. Where to put a custom model

Place the checkpoint in this directory and point the config at it:

```
models/
├── yolo11n.pt        # stock COCO weights (downloaded automatically)
├── best.pt           # your custom Indian-road model
└── pothole.pt        # optional dedicated pothole model
```

```yaml
# configs/default.yaml
detector:
  type: yolo
  model: models/best.pt     # <- your weights
  confidence: 0.35

pothole:
  enabled: true
  type: yolo                # 'none' | 'yolo' | 'heuristic'
  model: models/pothole.pt
  confidence: 0.30
```

Or from the command line, without editing the config:

```bash
python main.py --source camera --model models/best.pt
python main.py --source video --input road.mp4 --pothole-mode yolo
```

If the file is missing, the application fails with an explanation of what is
absent and how to fix it. It does **not** silently fall back to a different
model, and it never substitutes fabricated detections.

---

## 3. Expected class names

Class names are canonicalised by `src/core/classes.py`, so your model's labels
do not have to match exactly. Case, spaces and hyphens are normalised, and a
broad alias table is applied — `auto`, `autorickshaw`, `tuk-tuk`,
`three_wheeler` and `rickshaw` all resolve to `auto_rickshaw`.

Recommended `data.yaml` class list for a custom Indian-road model:

```yaml
names:
  0: person
  1: bicycle
  2: motorcycle
  3: car
  4: auto_rickshaw
  5: bus
  6: truck
  7: cow
  8: bull
  9: dog
  10: bullock_cart
  11: pushcart
  12: pothole
  13: debris
  14: traffic_cone
  15: construction_barrier
  16: traffic_light
  17: traffic_sign
```

Any label that does not map to a known class becomes `unknown`. That is
deliberate: the system treats an unidentified obstacle cautiously (CREEP or
STOP depending on range) rather than inventing a plausible-sounding class for
it. Unmapped labels appear in the logs so you can extend the alias table in
`src/core/classes.py` — one entry in `_ALIASES` and one in `_CATEGORIES` is all
a new class requires; no other module changes.

### Adding a brand-new class

```python
# src/core/classes.py
_ALIASES["tempo_traveller"] = "tempo_traveller"
_CATEGORIES["tempo_traveller"] = SemanticCategory.VEHICLE
```

Then optionally give it a size prior for distance estimation and a risk weight:

```yaml
# configs/default.yaml
distance:
  object_dimensions:
    tempo_traveller: {height: 2.30, width: 1.90}
risk:
  class_priority:
    tempo_traveller: 0.78
```

---

## 4. Training your own model

### 4.1 Datasets

Openly available starting points (check each licence before use):

| Need | Dataset |
| --- | --- |
| Indian road scenes, 34 classes | **IDD** — India Driving Dataset (idd.insaan.iiit.ac.in) |
| Potholes | **RDD2022** (road damage, multi-country); pothole sets on Roboflow Universe |
| Indian traffic objects | **IDD-Detection**, **AUTONUE** challenge data |
| General fallback | COCO (already covers person/vehicle/animal) |

For a working prototype, a few hundred well-labelled frames of *your own*
footage typically beats a large mismatched public set, because camera height,
lens and lighting all shift the distribution.

### 4.2 Dataset layout

```
datasets/indian_road/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/          # YOLO format: class cx cy w h, all normalised 0-1
    ├── train/
    └── val/
```

```yaml
# datasets/indian_road/data.yaml
path: datasets/indian_road
train: images/train
val: images/val
names:
  0: person
  1: motorcycle
  # ... as above
```

### 4.3 Fine-tuning

Starting from COCO weights converges far faster than training from scratch:

```bash
pip install ultralytics

yolo detect train \
  model=yolo11n.pt \
  data=datasets/indian_road/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  patience=20 \
  name=indian_road
```

The best checkpoint lands in `runs/detect/indian_road/weights/best.pt`. Copy it
here:

```bash
cp runs/detect/indian_road/weights/best.pt models/best.pt
```

### 4.4 A pothole-only model

Potholes benefit from a dedicated model: they are small, low-contrast, vary
enormously, and are easily confused with shadows and tar patches. Training one
class well is usually more effective than adding `pothole` as an eighteenth
class to a general detector.

```bash
yolo detect train model=yolo11s.pt data=datasets/potholes/data.yaml \
  epochs=150 imgsz=832 batch=8 name=pothole
cp runs/detect/pothole/weights/best.pt models/pothole.pt
```

Note `imgsz=832` — potholes occupy few pixels at distance, so a larger input
resolution matters more than a larger model.

### 4.5 Verifying a model before trusting it

```bash
yolo detect val model=models/best.pt data=datasets/indian_road/data.yaml
```

Then check its behaviour through this system:

```bash
python main.py --source image --input data/samples/your_test.jpg --save
python scripts/evaluate_dataset.py --input data/ --model models/best.pt
```

The startup coverage report will confirm which target classes the new weights
actually cover.

---

## 5. Pothole detection modes

| Mode | What it is | Trustworthiness |
| --- | --- | --- |
| `none` *(default)* | Disabled. The UI states potholes are **not** being detected. | n/a |
| `yolo` | A custom-trained pothole model. Real ML inference. | Depends on your training data |
| `heuristic` | Classical CV: dark, compact, convex blobs on the road surface. | **Low.** Shadows, wet patches, tar repairs and manhole covers all trigger it |

The heuristic exists so the pipeline can be demonstrated without custom
weights. It is not a substitute for a trained model, and the system never lets
it masquerade as one:

- every detection is tagged `DetectionSource.HEURISTIC`;
- confidence is hard-capped at **0.45**, below what a trained model reports;
- boxes are labelled **`HEUR (not ML)`** on screen;
- a persistent banner reads *"Potholes shown are CV heuristic, not ML detection"*;
- the risk engine adds the reason *"heuristic detection (not model-confirmed)"*.

---

## 6. Model size and speed

Measured on the CPU-only machine this prototype was developed on
(no CUDA available). GPU figures will be far better.

| Model | Params | CPU inference @640 | Suggested use |
| --- | --- | --- | --- |
| `yolo11n.pt` | 2.6 M | ~80-150 ms | Real-time-ish on CPU, default |
| `yolo11s.pt` | 9.4 M | ~200-350 ms | Better accuracy, GPU preferred |
| `yolo11m.pt` | 20 M | ~500 ms+ | Offline analysis |

To speed up CPU inference, reduce the inference resolution rather than
dropping to a weaker model:

```yaml
detector:
  imgsz: 480          # from 640
app:
  process_width: 640  # from 960
```

Check whether a GPU is actually being used:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

`device: auto` selects CUDA when present, then MPS, then CPU. A `cuda` request
on a machine without CUDA logs a warning and falls back to CPU rather than
crashing.

---

## 7. Files in this directory

`.pt` checkpoints are excluded from version control by `.gitignore` — they are
large binaries. Keep them in release artifacts or object storage, not in git.
