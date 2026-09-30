"""Teacher checkpoint aliases used by the CIFAR-100 experiments."""

import os
from pathlib import Path


_DEFAULT_TEACHER_ROOT = Path(
    "/data1/xujianyang/MTKD-RL-two/checkpoint/teachers/models"
)
TEACHER_ROOT = Path(
    os.environ.get("DCKD_TEACHER_ROOT", str(_DEFAULT_TEACHER_ROOT))
).expanduser()
LOCAL_TEACHER_ROOT = Path(__file__).resolve().parent / "checkpoint" / "teachers" / "models"


def _teacher_path(*parts):
    return str(TEACHER_ROOT.joinpath(*parts))


teacher_model_path_dict = {
    "A0": _teacher_path("resnet32x4_cifar100", "resnet32x4_A0_best.pth"),
    "A1": _teacher_path("resnet32x4_cifar100", "resnet32x4_A1_best.pth"),
    "A2": _teacher_path("resnet32x4_cifar100", "resnet32x4_A2_best.pth"),
    "A3": _teacher_path("resnet32x4_cifar100", "resnet32x4_A3_best.pth"),
    "50A0": _teacher_path("ResNet50_cifar100", "ResNet50_A0_best.pth"),
    "50A1": _teacher_path("ResNet50_cifar100", "ResNet50_A1_best.pth"),
    "50A2": _teacher_path("ResNet50_cifar100", "ResNet50_A2_best.pth"),
    "50A3": _teacher_path("ResNet50_cifar100", "ResNet50_A3_best.pth"),
    "vggA0": _teacher_path("vgg13_cifar100", "vgg13_A0_best.pth"),
    "vggA1": _teacher_path("vgg13_cifar100", "vgg13_A1_best.pth"),
    "vggA2": _teacher_path("vgg13_cifar100", "vgg13_A2_best.pth"),
    "vggA3": _teacher_path("vgg13_cifar100", "vgg13_A3_best.pth"),
    "tinyA0": str(LOCAL_TEACHER_ROOT / "resnet32x4_tinyimagenet" / "resnet32x4_A0_best.pth"),
    "tinyA1": str(LOCAL_TEACHER_ROOT / "resnet32x4_tinyimagenet" / "resnet32x4_A1_best.pth"),
    "tinyA2": str(LOCAL_TEACHER_ROOT / "resnet32x4_tinyimagenet" / "resnet32x4_A2_best.pth"),
    "tinyA3": str(LOCAL_TEACHER_ROOT / "resnet32x4_tinyimagenet" / "resnet32x4_A3_best.pth"),
    "tinyvggA0": str(LOCAL_TEACHER_ROOT / "vgg13_tinyimagenet" / "vgg13_A0_best.pth"),
    "tinyvggA1": str(LOCAL_TEACHER_ROOT / "vgg13_tinyimagenet" / "vgg13_A1_best.pth"),
    "tinyvggA2": str(LOCAL_TEACHER_ROOT / "vgg13_tinyimagenet" / "vgg13_A2_best.pth"),
    "tinyvggA3": str(LOCAL_TEACHER_ROOT / "vgg13_tinyimagenet" / "vgg13_A3_best.pth"),
    "dtdA0": str(LOCAL_TEACHER_ROOT / "resnet32x4_dtd" / "resnet32x4_A0_best.pth"),
    "dtdA1": str(LOCAL_TEACHER_ROOT / "resnet32x4_dtd" / "resnet32x4_A1_best.pth"),
    "dtdA2": str(LOCAL_TEACHER_ROOT / "resnet32x4_dtd" / "resnet32x4_A2_best.pth"),
    "dtdA3": str(LOCAL_TEACHER_ROOT / "resnet32x4_dtd" / "resnet32x4_A3_best.pth"),
    "dtdvggA0": str(LOCAL_TEACHER_ROOT / "vgg13_dtd" / "vgg13_A0_best.pth"),
    "dtdvggA1": str(LOCAL_TEACHER_ROOT / "vgg13_dtd" / "vgg13_A1_best.pth"),
    "dtdvggA2": str(LOCAL_TEACHER_ROOT / "vgg13_dtd" / "vgg13_A2_best.pth"),
    "dtdvggA3": str(LOCAL_TEACHER_ROOT / "vgg13_dtd" / "vgg13_A3_best.pth"),
    "tiny50A0": str(LOCAL_TEACHER_ROOT / "ResNet50_tinyimagenet" / "ResNet50_A0_best.pth"),
    "tiny50A1": str(LOCAL_TEACHER_ROOT / "ResNet50_tinyimagenet" / "ResNet50_A1_best.pth"),
    "tiny50A2": str(LOCAL_TEACHER_ROOT / "ResNet50_tinyimagenet" / "ResNet50_A2_best.pth"),
    "tiny50A3": str(LOCAL_TEACHER_ROOT / "ResNet50_tinyimagenet" / "ResNet50_A3_best.pth"),
    "dtd50A0": str(LOCAL_TEACHER_ROOT / "ResNet50_dtd" / "ResNet50_A0_best.pth"),
    "dtd50A1": str(LOCAL_TEACHER_ROOT / "ResNet50_dtd" / "ResNet50_A1_best.pth"),
    "dtd50A2": str(LOCAL_TEACHER_ROOT / "ResNet50_dtd" / "ResNet50_A2_best.pth"),
    "dtd50A3": str(LOCAL_TEACHER_ROOT / "ResNet50_dtd" / "ResNet50_A3_best.pth"),
    "rafA0": str(LOCAL_TEACHER_ROOT / "resnet32x4_rafdb" / "resnet32x4_A0_final.pth"),
    "rafA1": str(LOCAL_TEACHER_ROOT / "resnet32x4_rafdb" / "resnet32x4_A1_final.pth"),
    "rafA2": str(LOCAL_TEACHER_ROOT / "resnet32x4_rafdb" / "resnet32x4_A2_final.pth"),
    "rafA3": str(LOCAL_TEACHER_ROOT / "resnet32x4_rafdb" / "resnet32x4_A3_final.pth"),
    "rafvggA0": str(LOCAL_TEACHER_ROOT / "vgg13_rafdb" / "vgg13_A0_final.pth"),
    "rafvggA1": str(LOCAL_TEACHER_ROOT / "vgg13_rafdb" / "vgg13_A1_final.pth"),
    "rafvggA2": str(LOCAL_TEACHER_ROOT / "vgg13_rafdb" / "vgg13_A2_final.pth"),
    "rafvggA3": str(LOCAL_TEACHER_ROOT / "vgg13_rafdb" / "vgg13_A3_final.pth"),
    "raf50A0": str(LOCAL_TEACHER_ROOT / "ResNet50_rafdb" / "ResNet50_A0_final.pth"),
    "raf50A1": str(LOCAL_TEACHER_ROOT / "ResNet50_rafdb" / "ResNet50_A1_final.pth"),
    "raf50A2": str(LOCAL_TEACHER_ROOT / "ResNet50_rafdb" / "ResNet50_A2_final.pth"),
    "raf50A3": str(LOCAL_TEACHER_ROOT / "ResNet50_rafdb" / "ResNet50_A3_final.pth"),
}
