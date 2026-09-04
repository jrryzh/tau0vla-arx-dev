"""ARX LIFT2s tool-to-tray state-as-action finetuning route."""

from __future__ import annotations

import os
from pathlib import Path

from tau0_vla.adapters.arx_lift2s import ArxLift2sUnified, validate_dataset_contract
from tau0_vla.data import FrameFilter, register_config
from tau0_vla.data.modalities import ArmJoint, Gripper, Image, Prompt
from tau0_vla.data.modalities.image import ColorJitter, ResizeWithPad

_REPO = os.environ.get(
    "ARX_TOOL_YIPAN_LEROBOT_ROOT",
    str(
        Path(__file__).resolve().parents[2]
        / "data/0904_pickplace_tool_yipan/lerobot_v3_30fps_state_t_plus_1"
    ),
)
_NORM_STATS = str(Path(__file__).with_name("norm_stats.json"))
_IMAGE_TRANSFORMS = [
    ColorJitter(prob=0.33, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03),
    ResizeWithPad(224, 224),
]


@register_config
def arx_lift2s_pickplace_tool_yipan_ft() -> ArxLift2sUnified:
    validate_dataset_contract(_REPO)
    return ArxLift2sUnified(
        repo_id=_REPO,
        images=[
            Image("head", transforms=_IMAGE_TRANSFORMS),
            Image("left_wrist", transforms=_IMAGE_TRANSFORMS),
            Image("right_wrist", transforms=_IMAGE_TRANSFORMS),
        ],
        prompt=Prompt(
            template=(
                "You are controlling a robot.\n"
                "Robot type: ARX LIFT2s\n"
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
