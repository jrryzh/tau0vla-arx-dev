"""ARX LIFT2s right-arm pick-place fallback finetuning route."""

from __future__ import annotations

import os
from pathlib import Path

from tau0_vla.adapters.arx_lift2s import ArxLift2sUnified, validate_dataset_contract
from tau0_vla.data import FrameFilter, PromptSource, register_config
from tau0_vla.data.modalities import ArmJoint, Gripper, Image, Prompt
from tau0_vla.data.modalities.image import ColorJitter, ResizeWithPad

_REPO = os.environ.get(
    "ARX_LEROBOT_ROOT",
    "/home/xiangchengliu/code/tau-0-vla/data/arx_pickplace/"
    "lerobot_v3_30fps_state_t_plus_1",
)
_NORM_STATS = str(Path(__file__).with_name("norm_stats.json"))
_TASK = "Pick up the object and place it into the bowl."
_IMAGE_TRANSFORMS = [
    ColorJitter(prob=0.33, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03),
    ResizeWithPad(224, 224),
]


@register_config
def arx_lift2s_pickplace_ft() -> ArxLift2sUnified:
    validate_dataset_contract(_REPO)
    return ArxLift2sUnified(
        repo_id=_REPO,
        images=[
            Image("head", transforms=_IMAGE_TRANSFORMS),
            Image("left_wrist", transforms=_IMAGE_TRANSFORMS),
            Image("right_wrist", transforms=_IMAGE_TRANSFORMS),
        ],
        prompt_source=PromptSource.fix(_TASK),
        prompt=Prompt(
            template=(
                "You are controlling an ARX LIFT2s robot.\n"
                "Control mode: joint\n"
                "Whole-body control: disabled\n"
                "Task: {instruction}"
            )
        ),
        filter_by_segments=False,
        frame_filter=FrameFilter(positive=(), negative=()),
        state=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action_horizon=30,
        action_semantics="state_t_plus_1",
        action_offset_frames=1,
        state_padding_dim=40,
        action_padding_dim=40,
        source_kwargs={"video_backend": "pyav"},
        norm_stats_path=_NORM_STATS,
        return_all_norm_forms=True,
    )
