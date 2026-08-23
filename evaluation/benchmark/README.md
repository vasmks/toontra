# Independent Webtoon Benchmark

This benchmark contains 367 manual speech-bubble annotations across three
webtoons. `images.json` records the stable image IDs and source information,
while `annotations.json` contains the benchmark annotations.

The source artwork is not redistributed in this repository. The original
episodes are available from the official pages listed below.

The annotations in this repository were created for this benchmark and are
provided without the source artwork. The repository license does not grant any
rights to the underlying webtoon images, which remain subject to the rights and
terms of their respective owners.

Users who wish to run inference on the original images are responsible for
obtaining and using the source material in accordance with the applicable
terms and permissions.

| ID | Annotations | Official source |
| --- | ---: | --- |
| 01 | 204 | [A Crush Curse — Episode 1](https://www.webtoons.com/en/romance/a-crush-curse/episode-1/viewer?title_no=9858&episode_no=1) |
| 02 | 40 | [A Spell for a Smith — Episode 1](https://www.webtoons.com/en/fantasy/a-spell-for-a-smith/episode-1/viewer?title_no=6078&episode_no=1) |
| 03 | 123 | [This Wasn’t in My Adoption Plan! — Episode 1](https://www.webtoons.com/en/drama/this-wasnt-in-my-adoption-plan/episode-1/viewer?title_no=7566&episode_no=1) |

## Reproducing the detector comparison

The repository includes the raw full-page detection candidates used for the
reported YOLO26 and YOLOv7 comparison. The source webtoon images are not
required to reproduce the evaluation metrics.

From the repository root:

```bash
python evaluation/benchmark/evaluate.py
```

The evaluator reports three post-processing configurations:

- global NMS
- the runtime path: tile ownership with cross-tile NMS
- tile ownership followed by additional same-tile NMS

using the fixed benchmark configuration recorded in
`predictions/metadata.json`.

The expected aggregate results are also stored in `results.json`.

The cached prediction files contain bounding boxes, confidence scores, labels,
and source tile IDs only. They do not contain the original webtoon artwork.
