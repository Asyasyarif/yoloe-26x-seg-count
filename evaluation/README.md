# Accuracy corpus and measured benchmark

No accuracy, hardware minimum, latency, or cost claim is made from mock tests.
Supply at least **50 distinct, legally usable, manually reviewed images** and
a `corpus.jsonl` containing rows such as:

```json
{"image":"images/001.jpg","prompt":"person, car","counts":{"person":3,"car":0},"reviewed":true,"license":"Your source URL and applicable license"}
```

The numbers above only illustrate the file format; they are not measured data.
Class keys must equal the parser's resolved categories. Include dense/small
objects, occlusions, bright/dark scenes, empty targets, non-COCO classes,
attributes, and overlapping categories. Annotate object boxes as well if
object-level FP/FN, precision/recall or AP are required. Counts alone cannot
identify those quantities.

```bash
python scripts/evaluate.py evaluation/corpus.jsonl --device cuda:0 --output reports/accuracy.json
```

This runs the same corpus at 640/960/1280 and confidence .15/.25/.35/.5, recording
total-count MAE, exact-count match rate, per-class errors, latency, and peak
allocated GPU memory. Review per-class errors because a class swap can leave
the total count unchanged. Record billed Replicate costs separately; the
script does not invent prices. Set acceptance thresholds from reviewed results.
