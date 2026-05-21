# VLABench VQA Dataset Statistics

This file records sample counts under the two VLABench VQA `jsons_train` roots:

- Track 1 primitive: `/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/vlabench_vqa_assets/primitive/jsons_train`
- Track 2 primitive: `/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset/vlabench_vqa_assets_track_2/primitive/jsons_train`

## JSON Training Samples

| Split | Affordance | Goal Description | Spatial Understanding | Task Planning | Trajectory | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Track 1 primitive | 34,134 | 57,238 | 32,733 | 55,792 | 30,674 | 210,571 |
| Track 2 primitive | 3,843 | 3,006 | 1,360 | 2,792 | 2,081 | 13,082 |

The raw data-size ratio is:

```text
Track 1 : Track 2 = 210,571 : 13,082 ~= 16.10 : 1
```

As percentages:

```text
Track 1 ~= 94.15%
Track 2 ~= 5.85%
```

A `9:1` co-training mix corresponds to:

```text
Track 1 = 90%
Track 2 = 10%
```

So `9:1` is not the natural data-size ratio. It oversamples Track 2 by about:

```text
10% / 5.85% ~= 1.7x
```

To match the raw JSON sample counts, use approximately `16:1` instead.

## Trajectory Samples By Task

Counts below are from each split's `jsons_train/trajectory/trajectory_all_train.json`.

### Track 1 Primitive

Track 1 trajectory data contains 6 tasks in the current JSON.

| Task | Trajectory Samples |
| --- | ---: |
| add_condiment | 6,296 |
| select_chemistry_tube | 6,358 |
| select_fruit | 5,366 |
| select_mahjong | 6,134 |
| select_painting | 3,221 |
| select_poker | 3,299 |
| **Total** | **30,674** |

### Track 2 Primitive

Track 2 trajectory data contains 9 tasks in the current JSON.

| Task | Trajectory Samples |
| --- | ---: |
| add_condiment | 180 |
| select_book | 287 |
| select_chemistry_tube | 150 |
| select_drink | 270 |
| select_fruit | 300 |
| select_mahjong | 294 |
| select_painting | 150 |
| select_poker | 150 |
| select_toy | 300 |
| **Total** | **2,081** |
