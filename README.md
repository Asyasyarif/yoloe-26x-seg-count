# Firmngin Image Counter — YOLOE-26x

Count detections in one image using open-vocabulary text categories. Uses the
original `yoloe-26x-seg.pt` checkpoint with dynamic `YOLOE.set_classes()` on every
request. No training, frontend, video tracking, or fixed COCO class map.

| Input | Default | Contract |
| --- | --- | --- |
| `image` | required | One JPEG/PNG/WebP, ≤20 MB, ≤20 MP, each dimension ≥32px |
| `prompt` | required | 1–256 characters, up to 8 category phrases |
| `output_zip` | `false` | PNG when false; ZIP when true |
| `confidence` | `0.25` | `0.05`–`0.95` |
| `image_size` | `1280` | `640`, `960`, or `1280` |

The output is **exactly one file**: an annotated PNG, or a ZIP containing exactly
`annotated.png` and `result.json`. All bounding boxes show category and per-image
ID. JSON schema 1.1 uses those same detections/IDs, pixel coordinates, counts,
and request-relative class indices. IDs are not persistent identities.
Labels avoid each other where space permits. Long labels scale to fit;
tiny or crowded images can still limit readability. ZIP JSON retains full IDs
and category names.

## Output shape on Replicate

The model's declared Output schema is a single `{"type": "string", "format": "uri"}`,
and the final prediction value is always one URL. A published prediction can
nonetheless be read back as a one-element array (`["https://…"]`).

That is a bug in the Cog inference server (coglet), not in this model, and it is
not caused by `BaseRunner`/`run`. For a scalar `-> Path`, coglet uploads the file
and appends the URL to its internal streaming-output list, then emails a
non-terminal `output` webhook carrying that list (`["url"]`). The terminal value
is correctly unwrapped, but each progress webhook is sent from its own task, so
the list-shaped one can arrive **after** the terminal one. Any client that applies
webhooks in arrival order ends up with the array. See
[replicate/cog#3205](https://github.com/replicate/cog/issues/3205); present in
0.18–0.23.1, fixed upstream after 0.23.1.

Consequences here:

- `scripts/replicate_smoke.py` accepts a one-element array as the same single
  artifact and records which shape it saw under `output_shape` in the report, so
  a green workflow cannot hide a runtime regression.
- `COGLET_WHEEL=pypi:0.23.1` is set on the build step. `build.sdk_version` pins
  only the `cog` wheel; coglet is resolved separately and would otherwise float
  to whatever is newest on PyPI. Pinning makes the inference server reproducible.
- `tests/test_output_schema.py` asserts the generated Output schema is one URI and
  never `array`. It runs on CPU without weights.

## Prompts

Prefer `person, red helmet, forklift`. Simple wrappers (`count`, `find`, `detect`,
`hitung`, `cari`, `deteksi`) and `and`/`dan` work. Common aliases include
`hitung orang dan mobil` → `person, car`; unknown category phrases stay open
vocabulary. Conservative known plurals are normalized; arbitrary trailing `s`
is retained. Parser version is recorded in ZIP metadata.

Relationships/exclusions/questions such as `people holding boxes`, `cars except
blue ones`, and `people standing next to cars` are rejected. This parser is a
small grammar, not an LLM; explicit comma-separated category names work best.
Colors, materials, and unfamiliar categories are best effort and can return
zero detections. Counts describe detections and can be wrong.

Cross-class boxes with IoU ≥0.60 and a higher-confidence contrast class are
suppressed (see below). Cross-class boxes with IoU ≥0.90 are treated as duplicate
predictions; the higher confidence survives with a stable tie break. Same-class
suppression is left to YOLOE. This conservative policy reduces duplicate
overlapping prompts but can still merge nearly coincident real objects; evaluate
it on your domain.
Inference uses external class-aware NMS (`nms=None`, `iou=0.45`,
`agnostic_nms=False`), preventing broad cross-class suppression before the
documented near-identical-box policy. IoU 0.45 removed duplicate detections on
the included bus fixture at 1280; it is a provisional baseline and can suppress
distinct same-category objects with heavy overlap. Survivors are clipped,
validated, sorted by `(y1,x1,label,-confidence)`, then
assigned `obj_001…`. The same list drives annotation and JSON. The synchronous
Cog runner plus a model lock serializes class mutation and inference.

### Hidden contrast classes

YOLOE scores each class with its own sigmoid, so a lone requested category has
no competitor and labels every look-alike region with that category (`man` as
`woman`, a clipboard as `laptop`). Every request therefore gets hidden
competitors chosen automatically from a generic 1198-entry vocabulary
(`config/contrast_vocab.json`, the LVIS class names). Nothing is hand-picked,
so any prompt — including open-vocabulary ones such as `kambing`, `helm proyek`,
or `woman` — gets the same benefit.

At setup the detector embeds the whole vocabulary once with the model's own text
encoder (`model.get_text_pe`, MobileCLIP2) and keeps the vectors in memory
(~20 MB). Per request it embeds only the requested phrases, ranks vocabulary
entries by cosine similarity, and appends the top 8 as contrast classes. The
vocabulary is never re-embedded and `set_classes` embeds just the ≤16 class
names it is given, so a request adds one small text-encoder pass. Classes
total at most 16. Entries that are synonyms of a requested class are skipped:
exact and plural-insensitive label matches, and anything at or above 0.90
cosine similarity. Near-duplicates of an already chosen competitor (≥0.95) are
skipped too, so the tail covers different confusions. Ranking is by similarity,
then request order, then vocabulary order, so selection is deterministic.

The 0.90/0.95 thresholds are calibrated on MobileCLIP2's text space, not
measured against real images; verify them on a GPU with `scripts/evaluate.py`
before trusting a new prompt domain.

Contrast detections never reach the output. They only drop a requested box when
they overlap it at IoU ≥0.60 with a strictly higher confidence. Contrast boxes,
IDs, labels, counts, and JSON entries are otherwise discarded. A wrong contrast
can only cost a detection when it overlaps the requested box and outscores it;
an unrelated request is unchanged.

## Build and publish with GitHub Actions

The workflow `.github/workflows/replicate.yml` runs CPU unit/contract tests,
verifies official model checksums, builds the GPU container, then runs the
actual model with two distinct prompt sets **without network access**. It saves
JUnit results, an offline report, and the resolved runtime dependency list.
An ordinary GitHub Linux runner can build the image and run this CPU smoke test;
it does not measure GPU performance.

To enable push after build:

1. Create a Replicate model under the intended account, initially using a GPU
   with generous memory. Record its actual hardware in the benchmark report.
2. Optionally set repository variable `REPLICATE_MODEL` to `owner/model`.
   Default: `asyasyarif/object-counter`.
3. Set secret `REPLICATE_CLI_AUTH_TOKEN` from https://replicate.com/auth/token.
4. Optionally set secret `REPLICATE_API_TOKEN` from https://replicate.com/account/api-tokens
   for checking the model and testing the published API. These tokens have
   different purposes; do not commit either.
5. Push `main`, or run **Build and publish to Replicate** manually. Its optional
   `model_name` input overrides the variable. Pull requests run tests only.

Without a valid CLI secret, the workflow fails explicitly before building; it cannot
report deployment success. The API token is optional for push, but required to
verify the cloud API. If absent, that verification is clearly reported pending.
Once configured, the workflow logs in, `cog push` publishes, and real Replicate
predictions validate both modes at 1280 and the five-input schema. The workflow
summary links the deployed model only after those checks pass.
The immutable digest from `cog push --json` is tested again offline. ZIP runtime
metadata must match that exact image and the requested version; a stale API
version cannot accidentally pass publication checks.

## Local development

Runtime: Python 3.11, Cog SDK/CLI 0.23.1, Ultralytics 8.4.193, PyTorch 2.8.0 /
torchvision 0.23.0 CUDA 12.8, Pillow 11.3.0. coglet is pinned to 0.23.1 via
`COGLET_WHEEL` in the publish workflow (and can be set locally the same way);
`sdk_version` alone does not pin it. The CLIP tokenizer is pinned to a
full commit. Direct dependencies are pinned; each build archives all resolved
transitive versions. See [weights/README.md](weights/README.md) for model hashes.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
python3 scripts/prepare_assets.py
cog doctor
COGLET_WHEEL=pypi:0.23.1 cog build -t image-counter:local
cog run image-counter:local -i image=@examples/sample.jpg -i prompt="person, bus" -o annotated.png
cog run image-counter:local -i image=@examples/sample.jpg -i prompt="person, bus" -i output_zip=true -o results.zip
```

Real offline GPU verification (requires NVIDIA Docker runtime):

```bash
mkdir -p reports
docker run --rm --gpus all --network none --entrypoint python -v "$PWD/reports:/reports" image-counter:local scripts/verify_offline.py --sizes 640 960 1280 --output /reports/gpu.json
```

Pass `--device cpu` for a CPU smoke check. Assets must be present at build time;
`prepare_assets.py` runs on the host because Cog build commands cannot read
repository scripts. Text encoder stays at the image's working-directory root.
The model is never exported with baked class labels. No runtime dependency
installation is allowed. Missing assets fail during setup.

## Validation, privacy, and license

See [evaluation/README.md](evaluation/README.md) for the required 50+ reviewed
image evaluation and resolution/threshold sweeps. No high-accuracy or GPU-memory
claim is made from model benchmark mAP or unit tests. 1280 is the initial quality
configuration and needs domain validation.

Input EXIF is applied before detection; transparency is composited on white.
Multi-frame, unsupported, corrupt, oversized, and decompression-bomb images fail
validation. Output PNGs discard source metadata. Request files use isolated
temporary directories; returned artifacts remain present for Cog to serve.
Images are processed by Replicate; do not upload sensitive images without
appropriate permission. Save API output files within Replicate's default
one-hour retention window if needed longer; web UI retention differs.

Source is licensed AGPL-3.0 (see LICENSE), matching Ultralytics' open-source
distribution. Hosted/proprietary use requires reviewing AGPL obligations or
obtaining a suitable Ultralytics Enterprise license. Bundled DejaVu font keeps
its own license in `assets/fonts/LICENSE_DEJAVU`.

Official references checked for implementation:

- [YOLOE models and prompting](https://docs.ultralytics.com/models/yoloe/)
- [Pinned encoder source](https://github.com/ultralytics/ultralytics/blob/v8.4.193/ultralytics/nn/text_model.py)
- [Cog Python API](https://cog.run/python/), [YAML](https://cog.run/yaml/), [deploy](https://cog.run/deploy/)
- [Replicate GitHub Actions publishing](https://replicate.com/docs/guides/build/push-a-model-using-github-actions)
- [Replicate HTTP API](https://replicate.com/docs/reference/http) and [output retention](https://replicate.com/docs/topics/predictions/output-files)
- [PyTorch version/CUDA pairing](https://pytorch.org/get-started/previous-versions/)
- [Pillow API](https://pillow.readthedocs.io/en/stable/reference/Image.html)
