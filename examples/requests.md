# Request examples

```bash
cog run -i image=@examples/sample.jpg -i prompt="person, bus" -o annotated.png
cog run -i image=@examples/sample.jpg -i prompt="hitung orang dan mobil" -i output_zip=true -o results.zip
cog run -i image=@examples/sample.jpg -i prompt="forklift, red helmet" -i confidence=0.5 -i image_size=960 -o annotated.png
```

`sample.jpg` is Ultralytics' bus example, copied from the distributed package
assets. Upstream source: https://github.com/ultralytics/ultralytics/tree/main/ultralytics/assets
(Ultralytics AGPL-3.0 distribution). It is a smoke-test fixture, not an accuracy
corpus or a manually verified counting ground truth.

The public API output is a single URL. Download it immediately if needed beyond
Replicate's API retention window. ZIP mode contains the structured result;
PNG mode has no JSON response.

A published prediction can occasionally be read back as `["https://…"]`
(a one-element array). The model declares and returns one URI; the array comes from
coglet's progress webhook (see "Output shape on Replicate" in the README). Treat
a one-element list as the same single artifact.
