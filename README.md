# DCKD core code

This directory contains the cleaned CIFAR-100 training core for two methods:

- `OURS`: dynamic multi-teacher weighting, learned augmentation, optional EMA,
  late-epoch evaluation, cosine/multistep LR, and optional loss annealing.
- `AVG`: uniform averaging of the same teacher ensemble.

Historical logs, old training variants, visualizations, datasets, and model
checkpoints are intentionally excluded.

## Layout

```text
train_ours.py              OURS training entry point
train_avg.py               AVG training entry point
run_ours.sh                reproducible OURS launcher
run_avg.sh                 reproducible AVG launcher
helper/                    training and evaluation loops
models/cifar100/           teacher/student models and policy networks
models/cub200/util.py      shared feature projection layers
dataset/                   CIFAR-100 loaders and augmentation policy
distiller_zoo/             distillation and feature losses
setting.py                 teacher checkpoint aliases
utils.py                   metrics, logging, and LR schedules
```

## Data and teacher weights

The launchers read CIFAR-100 from `DATA_ROOT` and teacher weights from
`DCKD_TEACHER_ROOT`. On the current machine, the teacher root defaults to:

```text
/data1/xujianyang/MTKD-RL-two/checkpoint/teachers/models
```

Expected teacher files:

```text
resnet32x4_cifar100/resnet32x4_A{0,1,2,3}_best.pth
ResNet50_cifar100/ResNet50_A{0,1,2,3}_best.pth
vgg13_cifar100/vgg13_A{0,1,2,3}_best.pth
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Run OURS

VGG13 to VGG8 with the 240-epoch cosine configuration:

```bash
env \
  GPU=0 \
  DATA_ROOT=/data1/xujianyang/MTKD-RL-two/data \
  T_MODEL=vgg13 \
  ARCH=vgg8 \
  TEACHER_LIST="vggA0 vggA1 vggA2 vggA3" \
  TRAIN_PASSES=1 \
  MODEL_EMA_DECAY=0.9999 \
  EPOCHS=240 \
  FEAT_WEIGHT=3 \
  LR_TYPE=cosine \
  LATE_TEST_START=210 \
  RUN_LABEL=vgg13-vgg8-cosine \
  ./run_ours.sh
```

Supported architecture pairs:

| Teacher | Student | `TEACHER_LIST` |
|---|---|---|
| `resnet32x4` | `MobileNetV2` | `A0 A1 A2 A3` |
| `resnet32x4` | `wrn_16_2` | `A0 A1 A2 A3` |
| `resnet32x4` | `ShuffleV2` | `A0 A1 A2 A3` |
| `ResNet50` | `resnet32` | `50A0 50A1 50A2 50A3` |
| `vgg13` | `vgg8` | `vggA0 vggA1 vggA2 vggA3` |

## Run AVG

```bash
env \
  GPU=0 \
  DATA_ROOT=/data1/xujianyang/MTKD-RL-two/data \
  T_MODEL=vgg13 \
  ARCH=vgg8 \
  TEACHER_LIST="vggA0 vggA1 vggA2 vggA3" \
  EPOCHS=240 \
  FEAT_WEIGHT=2 \
  MILESTONES_OVERRIDE="150 180 210" \
  RUN_LABEL=vgg13-vgg8-avg \
  ./run_avg.sh
```

Both launchers write parameter snapshots and logs under `logs/` and models
under `checkpoint/`. Set `FOREGROUND=1` when another process manager such as
`tmux` should own the process.

## Tiny-ImageNet teachers

Train the four `resnet32x4` teachers with the original teacher optimization
settings (240 epochs, batch size 64, SGD 0.05, momentum 0.9, weight decay
5e-4, and LR decays at 150/180/210):

```bash
DATA_ROOT=/data2/dingyu/datasets/tiny-imagenet-200 \
  ./run_teachers_tinyimagenet.sh
```

The launcher runs A0/A1/A2 on GPUs 0/1/2 and queues A3 on GPU 0 after A0.
Logs are stored in `logs/tinyimagenet_teachers/`; checkpoints are stored in
`checkpoint/teachers/models/resnet32x4_tinyimagenet/`.
