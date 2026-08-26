"""Regression coverage for the final Tau VLA trainability policy."""

import os
import types
import unittest

import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from tau0_vla.models.model_builder import ModelBuilder
from tau0_vla.models.vision_language_action_models.tau0_vla import Tau0VLAConfig, Tau0VLAModel


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.merger = nn.Linear(2, 2)


class _VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.language_model = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 2)


class _FlowMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwenvl_with_expert = nn.Module()
        self.qwenvl_with_expert.qwenvl = _VLM()
        self.qwenvl_with_expert.qwen_expert = nn.Linear(2, 2)
        for name in (
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ):
            setattr(self, name, nn.Linear(2, 2))


class _TauModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_matching = _FlowMatching()


def _builder_for(model, *, vla_type="tau0_vla", vision=False, mlp=False, llm=False, dit=False):
    model_args = types.SimpleNamespace(
        vla_model_type=vla_type,
        vlm_model_type="qwen3.5",
        tune_mm_vision=vision,
        tune_mm_mlp=mlp,
        tune_mm_llm=llm,
        tune_vla_dit=dit,
    )
    builder = ModelBuilder(types.SimpleNamespace(lora_enable=False), model_args, types.SimpleNamespace(), True)
    builder.model = model
    return builder


def _all_trainable(module):
    return all(parameter.requires_grad for parameter in module.parameters())


def _all_frozen(module):
    return all(not parameter.requires_grad for parameter in module.parameters())


class ModelTrainabilityPolicyTest(unittest.TestCase):
    def apply_policy(self, *, vla_type="tau0_vla", vision=False, mlp=False, llm=False, dit=False):
        model = _TauModel()

        # Simulate Transformers checkpoint loading replacing constructor-frozen
        # Parameters with new, trainable Parameter objects.
        model.requires_grad_(False)
        model.requires_grad_(True)

        builder = _builder_for(
            model,
            vla_type=vla_type,
            vision=vision,
            mlp=mlp,
            llm=llm,
            dit=dit,
        )
        builder.set_training_parameters()
        return model.flow_matching

    def test_expert_only_freezes_the_entire_vlm(self):
        flow_matching = self.apply_policy(dit=True)

        self.assertTrue(_all_frozen(flow_matching.qwenvl_with_expert.qwenvl))

    def test_action_expert_and_projections_follow_tune_vla_dit(self):
        projection_names = (
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        )
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                flow_matching = self.apply_policy(dit=enabled)
                action_modules = [flow_matching.qwenvl_with_expert.qwen_expert]
                action_modules.extend(getattr(flow_matching, name) for name in projection_names)
                self.assertTrue(all(_all_trainable(module) == enabled for module in action_modules))

    def test_vision_merger_and_llm_switches_are_independent(self):
        for vision in (False, True):
            for mlp in (False, True):
                for llm in (False, True):
                    with self.subTest(vision=vision, mlp=mlp, llm=llm):
                        flow_matching = self.apply_policy(vision=vision, mlp=mlp, llm=llm)
                        vlm = flow_matching.qwenvl_with_expert.qwenvl
                        self.assertEqual(_all_trainable(vlm.visual.encoder), vision)
                        self.assertEqual(_all_trainable(vlm.visual.merger), mlp)
                        self.assertEqual(_all_trainable(vlm.language_model), llm)
                        self.assertEqual(_all_trainable(vlm.lm_head), llm)

    def test_both_tau_vla_names_apply_the_policy(self):
        for vla_type in ("tau_vla", "tau0_vla"):
            with self.subTest(vla_type=vla_type):
                flow_matching = self.apply_policy(vla_type=vla_type, dit=True)
                self.assertTrue(_all_frozen(flow_matching.qwenvl_with_expert.qwenvl))
                self.assertTrue(_all_trainable(flow_matching.qwenvl_with_expert.qwen_expert))


def _issue_10_architecture_config():
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
    qwen_config = Qwen3_5Config(
        text_config={
            "vocab_size": 250125,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "layer_types": layer_types,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": True,
            "eos_token_id": 248044,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
        },
        tie_word_embeddings=True,
    )
    return Tau0VLAConfig(
        qwenvl_config=qwen_config.to_dict(),
        max_action_dim=16,
        max_state_dim=16,
        action_dim=16,
        n_action_steps=30,
        use_lm_head=True,
        freeze_vision_encoder=True,
        train_expert_only=True,
    )


@unittest.skipUnless(
    os.environ.get("TAU0_RUN_FULL_MODEL_TEST") == "1",
    "set TAU0_RUN_FULL_MODEL_TEST=1 to run the 2.5B-parameter random-init regression",
)
class FullModelTrainabilityTest(unittest.TestCase):
    def test_expert_only_parameter_count_after_checkpoint_like_replacement(self):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = Tau0VLAModel(_issue_10_architecture_config())
        finally:
            torch.set_default_dtype(previous_dtype)

        self.assertFalse(any(parameter.is_meta for parameter in model.parameters()))
        model.requires_grad_(True)

        builder = _builder_for(model, dit=True)
        builder.set_training_parameters()

        vlm = model.flow_matching.qwenvl_with_expert.qwenvl
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        trainable_vlm_parameters = sum(parameter.numel() for parameter in vlm.parameters() if parameter.requires_grad)

        self.assertEqual(total_parameters, 2_503_092_816)
        self.assertEqual(trainable_parameters, 286_154_512)
        self.assertEqual(trainable_vlm_parameters, 0)


if __name__ == "__main__":
    unittest.main()
