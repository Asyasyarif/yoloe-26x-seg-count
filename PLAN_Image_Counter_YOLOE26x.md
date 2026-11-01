# PLAN.md — Firmngin Image Counter (YOLOE-26x + Replicate Cog)

**Status:** Implementation-ready plan for AI coding agents (implementation/deployment not yet performed)  
**Version:** MVP v1.1 — 2026-10-30  
**Project:** `firmngin-image-counter`  
**Deployment target:** One custom model hosted on Replicate, implemented with Cog (`BaseRunner` / `run`)  
**Primary model:** **Ultralytics YOLOE-26x**, checkpoint **`yoloe-26x-seg.pt`**, **text-prompted open-vocabulary inference**  
**Priorities:** High detection/counting quality > inference speed; maintain reproducibility and predictable costs  
**Out of scope:** Firmngin Cloud integration, frontend, video, line crossing, realtime streams, persistent object IDs

> ## NON-NEGOTIABLE CONTRACT
>
> - Required inputs: **`image`** and **`prompt`**.
> - Optional input: **`output_zip: bool = false`**. Optional quality controls: `confidence` and `image_size`.
> - **`output_zip=false` (default) → return exactly one annotated PNG image.**
> - **`output_zip=true` → return exactly one ZIP** with **`annotated.png` + `result.json`** (no extra members).
> - Both modes return **one `cog.Path`**, never a dict or list of outputs.
> - Every detected object displayed in the image has a **bounding box, category label and per-image ID**; JSON, when enabled, uses the same detections, labels and IDs.
> - Prompt is **open vocabulary** (not restricted to COCO's 80 classes). Use the original PyTorch `-seg.pt` checkpoint + `YOLOE.set_classes(...)` **per request**.
> - **Never** use `yoloe-26x-seg-pf.pt` for text prompting: `-pf` is prompt-free and does **not** accept `set_classes()`.

---

## 1. Product objective

Create a simple, reusable **image counting infrastructure component**, not a retail/warehouse-specific app. Users upload **one image** and tell the model **which object categories/descriptions to detect**. The model runs zero-shot detection, counts the returned instances, labels the image, and optionally packages the structured count in a ZIP.

Examples of supported intended usage:

- `person` → count people.
- `car, motorcycle, bicycle` → multi-category detection/counting.
- `red helmet` → attempt to detect a visually described category.
- `forklift, wooden pallet` → categories beyond standard COCO labels.
- `count red helmets and blue bottles` → normalize to two candidate class phrases (best effort; see parser limits).
- `hitung orang dan sepeda motor` → translate common Indonesian terms into English category phrases.

**Accuracy note:** Open vocabulary means the user can name categories absent from COCO; it **does not** guarantee exact natural-language reasoning or perfect detection for every category, color, texture, relation, or crowded scene. All count totals must be described as *detections*, and evaluated against manually labeled examples.

### In scope

- Image formats: PNG, JPEG and WebP; one image per prediction.
- Text prompt, allowing one or several visually grounded noun phrases.
- YOLOE-26x text prompting; zero fine-tuning required for initial MVP.
- Optional confidence threshold and inference image size.
- GPU inference using Replicate/Cog.
- Bounding-box drawing with exact class label and unique ID per image.
- Optional ZIP output with result JSON.
- Unit/integration tests, real-image accuracy evaluation, repeatable Cog build and Replicate publish workflow.

### Out of scope

- Training/fine-tuning, visual example prompting, prompt-free mode, LLM/VLM integration.
- Video tracking, motion/line counting, camera streams, real-time control and persistent identifiers.
- Object relationships, counting intentions/actions, OCR, facial recognition or identity claims.
- Firmngin-specific orchestration, multitenancy, webhooks, database, API gateway or billing.
- Segmentation-mask rendering (the official weight supports segmentation, but this MVP **only annotates bounding boxes**).

---

## 2. Verified base model and quality configuration

| Property | Decision |
|---|---|
| Framework | Ultralytics YOLOE (`from ultralytics import YOLOE`) |
| Weights | **`yoloe-26x-seg.pt`** |
| Prompting mode | Text: `model.set_classes(list[str])` |
| Model scale | **x** (largest YOLOE-26 release, highest reported text-prompt mAP within its size family) |
| Initialization | `YOLOE(local_checkpoint_path)` once per worker in Cog `setup()` |
| Text encoder | YOLOE-26 **`mobileclip2_b.ts`** + compatible `ultralytics/CLIP` installation |
| Inference | PyTorch `.pt` (avoid fixed-class ONNX/TensorRT export for dynamic prompts) |
| Image size | **1280** default candidate for high detail; expose optional `image_size` (`640`, `960`, `1280`) and benchmark |
| Confidence | Default **0.25**, configurable `0.05..0.95`, tune using evaluated dataset |
| NMS | Start with YOLOE's supported default; evaluate overlapping categories and duplicates |
| Output | Bounding boxes + IDs + counts (no masks in output) |

### Why this checkpoint

Official YOLOE docs list YOLOE-26x with **40.6 zero-shot text-prompt LVIS mAP50–95 at 640px** (non-end-to-end paper protocol), higher than the official YOLOE-26l at 37.8. These are **benchmark detection scores**, **not a 40.6% success rate**, **not counting accuracy**, and **not a promise that larger resolution always improves real-world results**.

**The `-seg.pt` suffix is intentional:** the available text-prompted YOLOE weights are instance-segmentation models, which also yield bounding boxes. No need to output masks. Do **not** swap to `yolo26x.pt` (closed vocabulary), `yoloe-26x-seg-pf.pt` (prompt-free), or an exported weight with baked-in classes.

### High-accuracy validation strategy

- Prioritize **recall + precision balanced for count accuracy**, not merely bounding-box confidence.
- Benchmark `image_size=640/960/1280` on representative inputs; 1280 is a **starting hypothesis**, not guaranteed winner.
- Compare confidence thresholds (0.15/0.25/0.35/0.5) and NMS choices on the same dataset.
- Include dense and tiny objects, occlusion, overlapping prompts (`car`, `red car`), bright/dark scenes, non-COCO categories.
- Do not enable multi-pass crops, TTA, or ensemble by default; allow as a later accuracy experiment with cost data.
- Determine GPU RAM requirement empirically for the **x** checkpoint at 1280; start benchmarking a GPU with generous VRAM (e.g. 24 GB) and downshift only after measured evidence. This is a planning starting point, not a verified hardware minimum.

---

## 3. High-level architecture

```text
Replicate UI / API
    |
    v
Cog Runner (single request: image, prompt, output_zip, confidence, image_size)
    |
    +-- validate image; apply EXIF orientation; convert RGB
    +-- normalize prompt -> category noun phrases (no closed-set COCO mapping)
    +-- YOLOE-26x + text encoder: set_classes([phrase1, phrase2, ...])
    +-- single image inference at selected image_size
    +-- filter confidences + validate boxes + suppress duplicate predictions
    +-- deterministic IDs within image: obj_001, obj_002, ...
    +-- annotate original oriented image -> annotated.png
    |
    +-- output_zip = false -> return annotated.png
    |
    +-- output_zip = true -> build result.json from same detections
                                  +-- ZIP_DEFLATED annotated.png + result.json
                                  +-- return results.zip
```

**Single source of truth:** a normalized, filtered and sorted `DetectedObject` list must drive counts, labels, image annotation, and JSON. Never infer the count again from a second run or from visual elements.

**Mutable-model safety:** `set_classes()` modifies model class state. Configure Cog to execute **one prediction at a time per worker**; do not enable async concurrent runs on the same YOLOE instance. If concurrency is introduced later, use isolated model instances or a strict lock around **set_classes + predict** and validate memory use.

---

## 4. Cog input/output contract

| Input | Type | Required | Default | Validation / behavior |
|---|---|---|---|---|
| `image` | `cog.Path` | Yes | — | JPEG, PNG or WebP, one file |
| `prompt` | `str` | Yes | — | Target class phrases; e.g. `red helmet, forklift` or `count red helmets` |
| `output_zip` | `bool` | No | `false` | Return ZIP with image+JSON when true; PNG only when false |
| `confidence` | `float` | No | `0.25` | Range `0.05..0.95` |
| `image_size` | `int` | No | `1280` | Choices `640`, `960`, `1280`; benchmark in production |

**Exactly one Cog file output** regardless of mode:

| `output_zip` | `run()` return | File format | Payload |
|---|---|---|---|
| Omitted / `false` | `Path` | PNG | `annotated.png` only |
| `true` | `Path` | ZIP | `results.zip` containing `annotated.png` and `result.json` |

No top-level `json_text`, no extra `.json` URL, no extra images. Clients can read the JSON **from the ZIP**, when requested. Avoid output filenames with ambiguous MIME extensions; validate PNG and ZIP file magic bytes in acceptance tests.

### Runner interface (implementation scaffold)

```python
from cog import BaseRunner, Input, Path
from ultralytics import YOLOE


class Runner(BaseRunner):
    def setup(self) -> None:
        # Weights and text encoder must already be provisioned/cached.
        self.model = YOLOE("weights/yoloe-26x-seg.pt")

    def run(
        self,
        image: Path = Input(description="Input photo (JPEG, PNG, WebP)"),
        prompt: str = Input(description="Objects to count, e.g. 'red helmet, forklift'"),
        output_zip: bool = Input(description="True returns annotated PNG and JSON in one ZIP", default=False),
        confidence: float = Input(description="Minimum detection confidence", default=0.25, ge=0.05, le=0.95),
        image_size: int = Input(description="Inference image size", default=1280, choices=[640, 960, 1280]),
    ) -> Path:
        # 1. normalize and validate image and prompt
        # 2. class_phrases = resolve_target_phrases(prompt)
        # 3. self.model.set_classes(class_phrases)
        # 4. result = self.model.predict(
        #      source=oriented_image, imgsz=image_size,
        #      conf=confidence, verbose=False
        #    )[0]
        # 5. inspect result.boxes.xyxy/conf/cls and model/result.names
        # 6. clip, filter, sort; assign unique per-image IDs
        # 7. write annotated.png; if requested also write result.json + ZIP
        # 8. return Path(existing_artifact_path)
        ...
```

**Important:** This scaffold intentionally omits processing code. Implement it using real Ultralytics outputs; smoke-test against the pinned package. `cog.Path` files must still exist when returned. Use request-specific directories created with `tempfile.mkdtemp()`; do **not** let a `TemporaryDirectory` context delete returned files before Cog serves them. Cog handles returned temporary files after serving.

### Minimal model API proof-of-concept

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26x-seg.pt")
model.set_classes(["red helmet", "forklift", "person"])
result = model.predict("sample.jpg", imgsz=1280, conf=0.25)[0]

for box in result.boxes:
    class_index = int(box.cls[0])
    print(result.names[class_index], box.conf[0], box.xyxy[0])
```

This proves **category-text conditioning**, not arbitrary free-form question answering. It is not a complete Cog project.

---

## 5. Prompt semantics — open vocabulary with controlled parsing

The model accepts a list of **category labels/descriptions**, not an unconstrained instruction-following question. The public input remains a simple `prompt: str`.

### Supported parsing strategy

1. Validate `prompt` length `1..256` chars (after trimming); reject control characters and empty input.
2. Prefer **explicit comma-separated classes** as the canonical format: `person, car, red helmet, forklift`.
3. Support simple command wrappers such as `count`, `find`, `detect`, `hitung`, `cari`, `deteksi` and common articles (`all`, `the`, `semua`, `yang ada`) when unambiguous.
4. Support simple conjunctions `and`, `dan` joining categories; maintain multi-word phrases (`red safety helmet`, `wooden pallet`).
5. Normalize a curated list of frequent Indonesian object names to English (`orang -> person`, `mobil -> car`, `sepeda motor -> motorcycle`, `helm merah -> red helmet`); preserve unknown noun phrases instead of forcing COCO classes.
6. Normalize *known* English singular/plural variants (`helmets -> helmet`, `forklifts -> forklift`, `people -> person`) conservatively. Do not apply naive trailing-`s` removal to arbitrary words; verify the output against parser tests.
7. Keep original prompt and resolved category phrases separately. Deduplicate categories, preserve stable order.
8. Limit to **8 category phrases** per request initially; performance and duplicate counts can degrade with very large overlapping class sets.
9. **Reject** prompts with unsupported relational/spatial/logic operators (e.g. `people holding two boxes`, `all objects to the left of the forklift`, `count cars except blue ones`) with a helpful explanation. Do not silently ignore constraints.
10. Warn in README that fine-grained visual adjectives (e.g. color/texture) are best effort, not guaranteed category correctness. Unrecognized noun phrases may return zero detections rather than a parser error, because the model is open vocabulary.
11. Never pass prompt text to `eval`, shell, Python execution or arbitrary code paths.

| Public prompt | Resolved category list | Expected behavior |
|---|---|---|
| `person` | `["person"]` | Detect person instances |
| `car, motorcycle` | `["car", "motorcycle"]` | Detect both categories |
| `count red helmets` | `["red helmet"]` | Open-vocabulary attempt |
| `find forklift and pallet` | `["forklift", "pallet"]` | Detect non-COCO types |
| `hitung orang dan mobil` | `["person", "car"]` | Normalize common Indonesian terms |
| `count red helmets and blue bottles` | `["red helmet", "blue bottle"]` | Best-effort attributes |
| `count everything` | Validation error | Too vague without categories |
| `people standing next to cars` | Validation error | Unsupported relationship logic |

Do not add an LLM-based parser in MVP. Write unit tests for phrase extraction and reject ambiguous constructs. Future enhancement can offer explicit `classes: list[str]` input or a separate prompt parser API, but do not expand the Replicate public schema in v1.1.

### Class index gotcha (critical)

**YOLOE dynamic class IDs reflect the provided `set_classes()` order**. The class index for `forklift` might be `0` in one request and `2` in another. **Do not use COCO class IDs** or a fixed COCO class map in inference or JSON. Resolve labels from `result.names[class_index]` (or a validated per-request phrase mapping) and validate the index is within the requested class list.

---

## 6. Asset provisioning / first-run requirements

Official docs state that the **first `set_classes()`** for YOLOE-26 installs/needs the `ultralytics/CLIP` tokenizer code and downloads the **`mobileclip2_b.ts`** TorchScript encoder (about 254 MB). The model checkpoint must also be available.

**Agents must make cold starts reproducible and avoid runtime installs/downloads**:

- Pin and test a compatible Ultralytics version (YOLOE-26 requires `ultralytics >= 8.4.0`), PyTorch/CUDA wheel and Cog SDK release; record exact versions in repository/lock or pip constraints.
- Prepare/download the exact `yoloe-26x-seg.pt` checkpoint before deployment, verify official source and record SHA256 in `weights/README.md`.
- Bake a **compatible, tested** `ultralytics/CLIP` dependency (pin its commit/version); do not leave a surprise `pip install` at first public request.
- Prepare/cache `mobileclip2_b.ts` at the working directory/path actually used by the installed Ultralytics version; test by running **two different prompt sets offline** after startup.
- Do not export to ONNX/TensorRT with class names baked in: official YOLOE exports become static and no longer support `set_classes()` for arbitrary user prompts.
- Make build reproducible using checked and cached assets or a build-time provisioning step. Note Cog `build.run` commands do **not** have access to repository source files; plan asset provisioning accordingly (e.g. prepare files locally before `cog build`, or use self-contained build commands).
- Confirm the actual checkpoint's input preprocessing and Tensor formats under the pinned dependency versions.
- Use `model.eval()` where supported; avoid unnecessary model reinitialization.

**Concurrency:** Single-threaded/single in-flight inference per worker in MVP (Cog defaults to one). Do not allow overlapping `set_classes()` requests. Any future caching of prompt embeddings must be keyed by normalized *ordered* classes and bound to the exact checkpoint + encoder version, and tested for leakage.

---

## 7. Image processing and detection-to-count logic

### Input validation

- Accept JPG/PNG/WebP, reject GIF, video, multi-frame files and invalid/corrupted payloads.
- Decode with Pillow, apply `ImageOps.exif_transpose()`, convert non-RGB to RGB with consistent alpha compositing.
- Suggested size guards: compressed image <= 20 MB, total pixel count <= 20 MP, dimension >= 32 px; make constants configurable.
- Reject decompression-bomb images; avoid arbitrary network download from untrusted paths inside model code.
- Preserve the **oriented source image** at full original resolution for annotation.

### Inference and postprocessing

1. Resolve text phrases from the prompt.
2. Set classes on the YOLOE-26x model **for this request**, appending automatic
   *contrast* classes chosen from the generic vocabulary
   `config/contrast_vocab.json` (1198 LVIS names) so a lone category has a
   competitor for YOLOE's per-class sigmoid scores. The vocabulary is embedded
   once in setup with the model's text encoder and cached in memory; per request
   only the requested phrases are embedded, then the top-K nearest entries by
   cosine similarity (K configurable, 6–10, default 8) are appended, excluding
   synonyms (exact/plural label match or similarity ≥0.90) and near-duplicates of
   already chosen entries (≥0.95). Requested classes keep indices `0..n-1`; at
   most 16 classes total per request. No hand-picked prompt pairs: any prompt,
   including open-vocabulary ones, gets the same treatment. `set_classes` receives
   precomputed embeddings so the per-request text-encoder cost stays at one small
   pass instead of a vocabulary-sized one.
3. Run prediction once (candidate `imgsz=image_size`; `conf=confidence`; no second hidden inference).
4. Read `result.boxes.xyxy`, `.conf`, `.cls`, and `result.names`; optionally inspect masks only for diagnostics, **not output**.
5. Ensure each reported class belongs to the requested phrase list (or to the
   contrast tail). Never assume fixed class IDs. A requested box is dropped only
   when a strictly higher-confidence contrast box overlaps it at IoU ≥0.60; then
   **all** contrast boxes are discarded. They must never appear in annotations,
   counts, `result.json`, or `obj_NNN` IDs.
6. Evaluate YOLOE's built-in NMS/class-agnostic behavior. In overlapping/near-synonymous prompts, prevent double-counting of the *same physical instance* with a documented, tested cross-class duplicate suppression policy. Do **not** suppress legitimately overlapping objects indiscriminately.
7. Clip boxes to oriented source dimensions; discard invalid/NaN/degenerate boxes.
8. Deterministically sort by `(y1, x1, label, -confidence)` (with stable tie break); assign `obj_001..obj_N`.
9. Counts derive only from the final surviving boxes.

### Annotation requirements

- Always create a **PNG** over the oriented original image, with visible rectangles.
- Each box label shows the category and per-image ID, e.g. `red helmet #001`.
- Use an appropriately licensed bundled font and draw text backgrounds for readability.
- Clamp labels to image edges and scale strokes/fonts to input resolution.
- Include a small total-count overlay if legible; **it must match JSON exactly**.
- If zero detections: still return a valid PNG with `Total: 0` (optional); ZIP JSON has `objects: []`.
- IDs belong only to the image being analyzed. **They are not track IDs or real-world identities.**

---

## 8. ZIP output and JSON schema

When `output_zip=true`, return one valid ZIP (UTF-8 JSON, ZIP_DEFLATED):

```text
results.zip
├── annotated.png
└── result.json
```

No other members. Serialize **one** canonical object list shared with the annotated image. No external/temporary URLs inside JSON.

### JSON example (illustrative fabricated detections; not benchmark data)

```json
{
  "schema_version": "1.1",
  "model": {
    "name": "yoloe-26x",
    "checkpoint": "yoloe-26x-seg.pt",
    "task": "open-vocabulary-object-detection"
  },
  "input": {
    "prompt": "count red helmets and forklifts",
    "resolved_classes": ["red helmet", "forklift"],
    "confidence_threshold": 0.25,
    "image_size": 1280,
    "image_width": 1280,
    "image_height": 720
  },
  "summary": {
    "total_objects": 3,
    "counts_by_class": {
      "red helmet": 2,
      "forklift": 1
    }
  },
  "objects": [
    {
      "id": "obj_001",
      "label": "red helmet",
      "prompt_class_index": 0,
      "confidence": 0.9563,
      "bbox": { "x1": 75, "y1": 44, "x2": 190, "y2": 175 }
    },
    {
      "id": "obj_002",
      "label": "forklift",
      "prompt_class_index": 1,
      "confidence": 0.8842,
      "bbox": { "x1": 322, "y1": 152, "x2": 914, "y2": 663 }
    },
    {
      "id": "obj_003",
      "label": "red helmet",
      "prompt_class_index": 0,
      "confidence": 0.8314,
      "bbox": { "x1": 1012, "y1": 101, "x2": 1109, "y2": 231 }
    }
  ]
}
```

### JSON invariants

- `total_objects == len(objects) == sum(counts_by_class.values())`.
- `counts_by_class` includes each requested class even when its count is zero.
- `prompt_class_index` is **request-relative**, zero-based, and corresponds to the `resolved_classes` list. It is **NOT** a COCO ID.
- `bbox` uses integer pixel coordinates on the oriented original image; `0 <= x1 < x2 <= image_width` and equivalent for Y.
- `confidence` in `[0,1]`, rounded for serialization **after** threshold filtering.
- IDs unique, consecutive, repeatable for same input/model/configuration within practical GPU determinism constraints; visible image labels use `#001..` corresponding to JSON IDs.
- The model and parser versions should be included in metadata / README so results can be traced to a specific deployed version.
- When `output_zip=false`, JSON can be formed internally as needed for counts, but **no JSON file/response may be emitted**.

---

## 9. Repository layout

```text
firmngin-image-counter/
├── PLAN.md
├── README.md
├── cog.yaml
├── requirements.txt
├── run.py                      # Cog BaseRunner public inputs + one-file output
├── src/
│   ├── __init__.py
│   ├── prompts.py              # Parse text -> ordered class phrases
│   ├── image_io.py             # File validation, EXIF, RGB
│   ├── detector.py             # YOLOE-26x adapter; dynamic set_classes
│   ├── schema.py               # Typed normalized detection records
│   ├── postprocess.py          # Dedup, clipping, sorting, IDs, counts
│   ├── annotate.py             # Bounding boxes + labels + IDs
│   └── packaging.py            # JSON, ZIP/PNG output
├── config/
│   └── language_aliases.json   # Minimal ID <-> EN terms, not COCO classes
├── assets/
│   └── fonts/                 # License-compliant font asset
├── weights/
│   ├── README.md              # Model + encoder asset origins, SHA256
│   └── ...                    # Pre-cached/baked exact weights (not necessarily Git-tracked)
├── scripts/
│   ├── prepare_assets.py      # Pre-build, host-side asset provisioning
│   └── verify_offline.py      # Ensures dynamic prompts work without downloads
├── examples/
│   ├── sample.jpg
│   └── requests.md
├── tests/
│   ├── test_prompts.py
│   ├── test_image_io.py
│   ├── test_postprocess.py
│   ├── test_annotations.py
│   ├── test_zip.py
│   ├── test_dynamic_classes.py
│   └── test_cog_integration.py
└── .gitignore
```

Do not commit large weights unless the repository distribution strategy permits. Build/push must nonetheless include or safely provision them. Document license of assets and exact fingerprints.

---

## 10. Cog config and local runs

Starter `cog.yaml` (agents must pin compatible dependency versions and test build):

```yaml
build:
  gpu: true
  python_version: "3.11"
  python_requirements: requirements.txt
  system_packages:
    - git
    - libgl1
    - libglib2.0-0
run: "run.py:Runner"
```

Use modern `run:` with Cog `BaseRunner`. Legacy `predict:` / `BasePredictor` should not be used for new code. Verify `cog doctor` and dependency compatibility. Keep default single-run per worker for `set_classes()` safety.

The output shape is one `Path` and the generated Output schema is a single
`{"type":"string","format":"uri"}`. Note that `sdk_version` pins only the `cog`
wheel: coglet is resolved separately, so the build must set `COGLET_WHEEL` to pin
the inference server too. A published prediction may still be read back as
`["https://…"]` because of coglet's array-shaped progress webhook
(replicate/cog#3205); that is a runtime bug, not a reason to change the interface
or switch to `BasePredictor`.

### Required example tests

```bash
cog doctor
cog build

# Normal: PNG only
cog run \
  -i image=@examples/sample.jpg \
  -i prompt="red helmet, forklift" \
  -i output_zip=false \
  -o annotated.png

# JSON + annotated PNG inside exactly one ZIP
cog run \
  -i image=@examples/sample.jpg \
  -i prompt="red helmet, forklift" \
  -i output_zip=true \
  -o results.zip

unzip -l results.zip
unzip -p results.zip result.json | python -m json.tool
```

Use two consecutive requests with **different prompt sets** to prove no class leakage. Try same image prompt with confidence 0.25 and 0.5 to verify threshold wiring.

---

## 11. Replicate deployment

1. Verify Ultralytics license suitability for intended hosted/commercial use (AGPL-3.0 vs Enterprise). **Do not assume proprietary hosted use is automatically exempt.**
2. Create Replicate model `firmngin-image-counter` under the intended Replicate account.
3. Install Docker + Cog, login (`cog login`).
4. Create `requirements.txt` with **tested exact versions** of torch/torchvision/ultralytics/Pillow and any CLIP dependency; verify CUDA pairing.
5. Provision/cache the YOLOE-26x checkpoint and tokenizer/text-encoder assets. Test `set_classes()` with two different prompt lists **without network access**.
6. `cog build`, then `cog run` both image-only and ZIP modes; validate actual output signatures and schema.
7. Measure peak VRAM and latency with `imgsz=640,960,1280`. Select a suitable Replicate GPU; do not publish an untested default that regularly OOMs.
8. Push:

```bash
cog push r8.im/<OWNER>/firmngin-image-counter
```

9. On Replicate UI/API confirm all inputs (`image`, `prompt`, `output_zip`, `confidence`, `image_size`) and exactly one output artifact.
10. Test new categories (`forklift`, `red helmet`) and English/Indonesian command wrappers.
11. Record cold start, warm latency, GPU type, maximum VRAM, success/error rates, and cost per run from **measured** Replicate predictions. Do not invent price/performance figures.
12. Publish once all acceptance tests and license/privacy checks pass. No public model URL is claimed until successfully pushed.

**Replicate output retention:** API-generated output artifacts are deleted after **one hour** by default; downstream clients must save the returned file if they need it longer. Web UI retention differs; consult current docs.

---

## 12. Test coverage and acceptance criteria

### Functional matrix

| ID | Scenario | Acceptance requirement |
|---|---|---|
| T01 | `output_zip` omitted | Returns exactly one PNG |
| T02 | `output_zip=false` | Valid annotated PNG, no JSON artifact |
| T03 | `output_zip=true` | Returns exactly one valid ZIP |
| T04 | ZIP contents | Exactly `annotated.png`, `result.json` |
| T05 | Open vocab | Detect attempts for `forklift`, `red helmet`, etc.; not rejected for being non-COCO |
| T06 | Changing prompt | Sequential distinct `set_classes` calls do not leak old labels |
| T07 | Dynamic class index | JSON label and `prompt_class_index` map to current request classes |
| T08 | Multi-target | `person, red helmet` returns requested classes only |
| T09 | Indonesian aliases | `hitung orang dan mobil` resolves expected English categories |
| T10 | Relational/ambiguous prompt | Clear validation error; no silent dropping of meaning |
| T11 | IDs | One unique per-image ID for each returned object; image matches JSON |
| T12 | Counts | Total and per-class counts match objects list |
| T13 | Zero detections | Valid PNG; ZIP JSON `total_objects=0` |
| T14 | Image validation | Corrupt/huge/unsupported files rejected cleanly |
| T15 | EXIF / alpha images | Output coordinates align to oriented RGB image |
| T16 | Threshold/image size | Inputs actually affect inference and validate choices |
| T17 | Cold-start assets | Container runs dynamic prompts with no runtime pip/download |
| T18 | Memory safety | Peak GPU VRAM within chosen Replicate instance capacity |
| T19 | Replicate smoke test | Both modes work via Replicate model UI/API |
| T20 | Concurrent requests | No shared-class mutation race (concurrency disabled / isolated) |

### Accuracy evaluation (mandatory for the "high accuracy" requirement)

- Assemble **at least 50** legally usable images with manually reviewed counts; include person/car, crowded scenes, small objects, `red helmet`, `forklift`, multi-category and empty-target images.
- Collect `ground_truth_count`, `predicted_count`, absolute count error, false positives, false negatives, and examples of errors **for each prompt and image**.
- Report **count MAE** (`mean(abs(predicted - actual))`), **exact-count match rate**, and object-level precision/recall or AP where box annotations exist.
- Evaluate at `640`, `960`, `1280` and several confidence thresholds on the same dataset.
- Benchmark both exact single-category prompts and overlapping/attribute prompts; document double-count failures.
- Establish acceptance thresholds from the actual validation dataset. Do **not** declare high accuracy solely because the model's LVIS mAP is 40.6.
- Save a concise benchmark report with representative false-positive/false-negative examples.

### Definition of done

- [ ] Cog builds reproducibly with cached YOLOE-26x + text encoder assets.
- [ ] Prompt categories work beyond COCO; `set_classes` invoked correctly for each request.
- [ ] Both `output_zip` modes return one file of the specified type.
- [ ] ZIP contains exactly `annotated.png` and valid `result.json`.
- [ ] Annotation boxes show category and unique per-image ID.
- [ ] JSON categories, class indices, bounding boxes, IDs and counts match annotation.
- [ ] Prompt parsing and unsupported-input behavior are documented and tested.
- [ ] Test results, real-image accuracy metrics, VRAM, latency and costs are recorded.
- [ ] Replicate model pushed and smoke-tested (if account access and permission available).
- [ ] README clarifies open-vocabulary limits, privacy, output retention, and licensing.

---

## 13. Agent execution tasks

**Agent 1 — Cog + model provisioning**

- Implement `cog.yaml`, `requirements.txt`, `run.py`, preload checkpoint + `mobileclip2_b.ts` and compatible CLIP code.
- Enforce single in-flight request, no first-run runtime installation/download.
- Provide repeatable local and Replicate GPU smoke-test commands.

**Agent 2 — Prompt + detection**

- Implement robust category-phrase parser, Indonesian aliases, error messages.
- Integrate YOLOE-26x `set_classes()` dynamic prompting; obtain actual class labels from per-request results, not COCO map.
- Implement filtering, validation and duplicate handling.

**Agent 3 — Annotation + packaging**

- Draw full-resolution annotated PNG: bounding boxes, labels, IDs and total.
- ZIP builder with the exact two named files; JSON schema v1.1 and invariants.
- Keep output type strictly one `Path` and verify ZIP=bool behavior.

**Agent 4 — QA + benchmarking + deployment**

- Unit/integration tests, 50+ image accuracy corpus, VRAM/latency benchmarks.
- Test distinct sequential prompts, class contamination, attributes, small/crowded objects.
- Push Replicate model when authorized; record model URL and results, or report specific deployment blockers.

### Handoff artifacts

Deliver full source repository, `README.md`, `cog.yaml`, pinned deps, asset provisioning instructions/checksums, automated test report, real-image accuracy report, GPU performance/cost results, and deployed model URL **only if deployment genuinely succeeded**. No invented metrics, links, or claims of deployment.

**Coding agent instruction:** Implement this plan, not the previous `yolo26n.pt` closed-vocabulary version. Do not add frontend or Firmngin-specific APIs. Preserve the strict output ZIP/image behavior above.

---

## 14. Official references

- Ultralytics YOLOE model docs, weights, prompt modes, benchmarks and text encoder: https://docs.ultralytics.com/models/yoloe/
- Ultralytics YOLO26 official model and license notes: https://docs.ultralytics.com/models/yolo26/
- Cog BaseRunner Python API: https://cog.run/python/
- Cog YAML: https://cog.run/yaml/
- Cog CLI / deployment: https://cog.run/deploy/
- Replicate publish guide: https://replicate.com/docs/guides/build/push-a-model
- Replicate output files and retention: https://replicate.com/docs/topics/predictions/output-files
