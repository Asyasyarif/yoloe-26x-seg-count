# Pinned model assets

Run `python scripts/prepare_assets.py` on the host **before** `cog build`.
Use `--check` for offline verification. Model files are excluded from Git but
included in the Cog image (they must not appear in `.dockerignore`).

Official release: https://github.com/ultralytics/assets/releases/tag/v8.4.0

| Asset | Location in image | SHA256 |
| --- | --- | --- |
| YOLOE-26x text/visual checkpoint | `weights/yoloe-26x-seg.pt` | `d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639` |
| MobileCLIP2 TorchScript encoder | `mobileclip2_b.ts` (working directory) | `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f` |

These digests and byte sizes were checked against the official GitHub release
asset metadata on 2026-10-30. The manifest is the source of truth. Corrupt cache
entries fail the build; they are never silently accepted or replaced.

Tokenizer dependency: `ultralytics/CLIP` commit
`7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40`, installed by requirements.txt, including
its packaged BPE vocabulary. Source:
https://github.com/ultralytics/CLIP/tree/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40

Only original `yoloe-26x-seg.pt` is supported. The `-pf` checkpoint cannot accept
dynamic classes. Original PyTorch inference is retained; no class-baked export.
See Ultralytics' AGPL/Enterprise terms: https://www.ultralytics.com/license.
