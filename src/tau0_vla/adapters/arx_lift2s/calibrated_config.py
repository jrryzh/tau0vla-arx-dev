"""Shared construction for the six independent calibrated training routes."""
import json
import hashlib
from pathlib import Path

from tau0_vla.data import FrameFilter
from tau0_vla.data.modalities import ArmJoint, Gripper, Image, Prompt
from tau0_vla.data.modalities.image import ColorJitter, ResizeWithPad
from .calibrated import ArxCalibratedJoint, ArxCalibratedEEF, contract, validate_calibrated_contract


def make_config(config_file, name, mode):
    directory = Path(config_file).resolve().parent
    root = directory.parents[1] / "data/0907_blue_t_v1" / name / mode
    validate_calibrated_contract(root, mode)
    stats = directory / "norm_stats.json"
    metadata = json.loads(stats.read_text()).get("calibrated_provenance", {})
    if metadata.get("dataset") != name or metadata.get("contract") != contract(mode):
        raise ValueError("normalization statistics do not belong to this calibrated experiment")
    manifest_sha = hashlib.sha256((root / "meta/arx.json").read_bytes()).hexdigest()
    if metadata.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("normalization statistics were fitted to a different source manifest")
    transforms = [ColorJitter(prob=.33, brightness=.3, contrast=.4, saturation=.5, hue=.03), ResizeWithPad(224, 224)]
    cls = ArxCalibratedEEF if mode == "eef-vr" else ArxCalibratedJoint
    return cls(repo_id=str(root), images=[Image(camera, transforms=transforms) for camera in ("head", "left_wrist", "right_wrist")],
        prompt=Prompt(template="You are controlling a robot.\nRobot type: ARX LIFT2s\nControl mode: " + mode.split("-")[0] + "\nWhole-body control: disabled\nTask: {instruction}"),
        filter_by_segments=False, frame_filter=FrameFilter(positive=(), negative=()),
        state=[ArmJoint(normalize="none"), Gripper(normalize="none")], action=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action_horizon=30, action_semantics=contract(mode)["action_semantics"], action_offset_frames=None,
        deployment_contract=contract(mode), state_padding_dim=40, action_padding_dim=40,
        source_kwargs={"video_backend": "pyav"}, norm_stats_path=str(stats), return_all_norm_forms=True)
