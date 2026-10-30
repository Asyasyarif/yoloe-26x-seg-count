# Verification status

2026-10-30 local verification: **63 tests passed**. Official checkpoint and
encoder SHA256/byte sizes verified. Real model ran sequential prompts at
640/960/1280 under a socket-level network block and the sandbox's network
restriction, successfully returning valid PNG and ZIP artifacts.

Local installed packages: Ultralytics 8.4.192, torch 2.15.0, torchvision 0.29.2,
Pillow 11.3.0, Cog SDK 0.23.1. This differs from the intended pinned container
(Ultralytics 8.4.193, torch 2.8.0+cu128, torchvision 0.23.0+cu128); local smoke
does **not** verify that exact container. The newer host torch warns that
TorchScript loading is deprecated; the officially supplied YOLOE text encoder
still requires it. The container's pinned torch version supports this format.

GitHub Actions build and Replicate publication remain pending authenticated
write access. The connector's first write returned HTTP 403, and sandbox
restrictions prevent local GitHub network/Docker socket access. No deployed
model URL or GPU/cost/accuracy result is claimed.

Local contract tests use the real Pillow/Cog SDK with an injected model boundary.
The Actions workflow performs actual YOLOE inference at 640 on CPU inside the
built GPU image, isolated with `--network none`. Its `offline.json` and
`runtime-requirements.txt` artifacts record measurements and resolved packages.

After push, the workflow checks the actual Replicate schema and runs both output
modes at 1280. `replicate.json` records version, prediction IDs, and API timings.

Pending until measured: GPU peak VRAM at all resolutions, accuracy on 50+
reviewed images, counting acceptance thresholds, actual billed cost, and
production GPU selection. A successful smoke test is not an accuracy benchmark.
