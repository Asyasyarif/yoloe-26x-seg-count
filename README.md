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

Cross-class boxes with IoU ≥0.90 are treated as duplicate predictions; the higher
confidence survives with a stable tie break. Same-class suppression is left to
YOLOE. This conservative policy reduces duplicate overlapping prompts but can
still merge nearly coincident real objects; evaluate it on your domain.
Inference uses external class-aware NMS (`nms=None`, `iou=0.45`,
`agnostic_nms=False`), preventing broad cross-class suppression before the
documented near-identical-box policy. IoU 0.45 removed duplicate detections on
the included bus fixture at 1280; it is a provisional baseline and can suppress
distinct same-category objects with heavy overlap. Survivors are clipped,
validated, sorted by `(y1,x1,label,-confidence)`, then
assigned `obj_001…`. The same list drives annotation and JSON. The synchronous
Cog runner plus a model lock serializes class mutation and inference.

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
torchvision 0.23.0 CUDA 12.8, Pillow 11.3.0. The CLIP tokenizer is pinned to a
full commit. Direct dependencies are pinned; each build archives all resolved
transitive versions. See [weights/README.md](weights/README.md) for model hashes.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
python3 scripts/prepare_assets.py
cog doctor
cog build -t image-counter:local
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
