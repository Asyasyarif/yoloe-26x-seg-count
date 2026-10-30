# Real-image duplicate investigation — 2026-10-30

Image: the included Ultralytics bus fixture. Visual review: four visible people,
one bus (including people cut off by the image edges). This is one reviewed
smoke fixture, **not** the required 50-image validation corpus.

Runtime: CPU, Ultralytics 8.4.192, torch 2.15.0, torchvision 0.29.2,
Pillow 11.3.0. The final container pins different Ultralytics/torch versions;
these measurements do not substitute for its CI check or GPU benchmarks.

| Resolution | Confidence | External NMS IoU | Person detections | Bus detections |
| --- | --- | --- | --- | --- |
| 640 | .25 | .70 | 4 | 1 |
| 960 | .25 | .70 | 4 | 1 |
| 1280 | .25 | .70 | 6 | 2 |
| 1280 | .50 | .70 | 4 | 1 |
| 640 | .25 | .45 | 4 | 1 |
| 960 | .25 | .45 | 4 | 1 |
| 1280 | .25 | .45 | 4 | 1 |

Default .70 IoU at 1280 produced partial duplicate same-class boxes. The
NMS-free head did not solve it. Production therefore starts with supported
`nms=None, iou=0.45, agnostic_nms=False`; request confidence remains .25. The
deprecated `end2end` argument is not used. Cross-class near-identical suppression
remains a separate IoU ≥.90 policy.

Validate this threshold on crowded/occluded examples: lower same-class NMS IoU
can merge overlapping real people. No aggregate precision/recall, counting MAE,
high-accuracy, memory, or cost claim follows from this fixture. Raw sweep data
was written locally to `nms-comparison.json` and `nms-iou-comparison.json`.

Official pinned configuration reference:
https://github.com/ultralytics/ultralytics/blob/v8.4.193/ultralytics/cfg/default.yaml
