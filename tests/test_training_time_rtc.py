from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import HfArgumentParser

from tau0_vla.configs.model_config import ModelArguments
from tau0_vla.models.vision_language_action_models.tau0_vla.modeling_tau_vla import (
    AdaRMSNorm,
    FlowMatching,
    Tau0VLAConfig,
    Tau0VLAModel,
)
from tau0_vla.models.vision_language_action_models.tau0_vla.utils import create_sinusoidal_pos_embedding
from tau0_vla.utils.utils import load_config_from_yaml


def _rtc_config(**overrides):
    values = dict(
        training_time_rtc=True,
        rtc_max_delay=2,
        n_action_steps=4,
        max_action_dim=3,
        action_dim=3,
        max_prefix_len=0,
        num_steps=2,
        vlm_causal=True,
        tau_vla_prefix_flash_backend=False,
        vla_inactive_input_zero=False,
        zero_state_emb=False,
        action_rope_offset=0,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _bare_flow(config):
    flow = FlowMatching.__new__(FlowMatching)
    nn.Module.__init__(flow)
    flow.config = config
    flow._n_action_steps = config.n_action_steps
    flow._use_adanorm = False
    flow._compiled_predict_velocity = None
    flow._cuda_graph_prefix = None
    flow._cuda_graph_prefix_signature = None
    return flow


class RTCConfigTest(unittest.TestCase):
    def test_defaults_and_checkpoint_round_trip(self):
        default = Tau0VLAConfig(qwenvl_config={})
        self.assertFalse(default.training_time_rtc)
        self.assertEqual(default.rtc_max_delay, 0)

        config = Tau0VLAConfig(
            qwenvl_config={}, n_action_steps=4, training_time_rtc=True, rtc_max_delay=2
        )
        restored = Tau0VLAConfig.from_dict(config.to_dict())
        self.assertTrue(restored.training_time_rtc)
        self.assertEqual(restored.rtc_max_delay, 2)

        legacy_dict = config.to_dict()
        legacy_dict.pop("training_time_rtc")
        legacy_dict.pop("rtc_max_delay")
        legacy = Tau0VLAConfig.from_dict(legacy_dict)
        self.assertFalse(legacy.training_time_rtc)
        self.assertEqual(legacy.rtc_max_delay, 0)

    def test_invalid_enabled_ranges(self):
        for max_delay in (-1, 0, 4, 5):
            with self.subTest(max_delay=max_delay), self.assertRaises(ValueError):
                Tau0VLAConfig(
                    qwenvl_config={}, n_action_steps=4, training_time_rtc=True, rtc_max_delay=max_delay
                )

    def test_cli_and_yaml_override_parsing(self):
        parsed, = HfArgumentParser(ModelArguments).parse_dict(
            {"training_time_rtc": True, "rtc_max_delay": 3}
        )
        self.assertTrue(parsed.training_time_rtc)
        self.assertEqual(parsed.rtc_max_delay, 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "experiment:\n  run_name: rtc\n"
                "model_args:\n  model_name_or_path: /tmp/model\n"
                "data_args: {}\ntraining_args:\n  output_dir: /tmp/out\n"
            )
            loaded = load_config_from_yaml(
                path, overrides={"training_time_rtc": "true", "rtc_max_delay": "3"}
            )
        self.assertIs(loaded["model_args"]["training_time_rtc"], True)
        self.assertEqual(loaded["model_args"]["rtc_max_delay"], 3)


class RTCTimestepTest(unittest.TestCase):
    def test_sinusoidal_embedding_accepts_shared_and_token_times(self):
        shared = create_sinusoidal_pos_embedding(torch.tensor([0.2, 0.7]), 4, 0.01, 1.0)
        token = create_sinusoidal_pos_embedding(
            torch.tensor([[0.2, 0.2], [0.7, 0.7]]), 4, 0.01, 1.0
        )
        self.assertEqual(shared.shape, (2, 4))
        self.assertEqual(token.shape, (2, 2, 4))
        torch.testing.assert_close(token[:, 0], shared)

    def test_embed_suffix_builds_token_level_adanorm_condition(self):
        config = _rtc_config(proj_width=4)
        flow = _bare_flow(config)
        flow.state_proj = nn.Linear(3, 4)
        flow.action_in_proj = nn.Linear(3, 4)
        flow.action_time_mlp_in = nn.Linear(8, 4)
        flow.action_time_mlp_out = nn.Linear(4, 4)

        timestep = torch.tensor([[0.0, 0.0, 0.6, 0.6], [0.0, 0.4, 0.4, 0.4]])
        ada_cond, suffix, _, _ = flow.embed_suffix(
            torch.zeros(2, 3), torch.zeros(2, 4, 3), timestep
        )
        self.assertEqual(ada_cond.shape, (2, 5, 4))
        self.assertEqual(suffix.shape, (2, 5, 4))
        # State token uses the unmasked global timestep carried by the last action.
        torch.testing.assert_close(ada_cond[:, 0], ada_cond[:, -1])

        norm = AdaRMSNorm(4, 4)
        output, gate = norm(suffix, ada_cond)
        self.assertEqual(output.shape, suffix.shape)
        self.assertEqual(gate.shape, suffix.shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the compiled RTC smoke test")
    def test_compiled_velocity_accepts_token_level_timestep(self):
        config = _rtc_config(proj_width=4, adanorm_time=True)
        flow = _bare_flow(config)
        flow.state_proj = nn.Linear(3, 4)
        flow.action_in_proj = nn.Linear(3, 4)
        flow.action_time_mlp_in = nn.Linear(8, 4)
        flow.action_time_mlp_out = nn.Linear(4, 4)
        flow.action_out_proj = nn.Linear(4, 3)
        flow._use_adanorm = True

        class Joint(nn.Module):
            def forward(self, inputs_embeds, **kwargs):
                return (None, inputs_embeds[1]), None

        flow.qwenvl_with_expert = Joint()
        flow = flow.cuda().eval()
        state = torch.zeros(2, 3, device="cuda")
        actions = torch.zeros(2, 4, 3, device="cuda")
        timestep = torch.tensor(
            [[0.0, 0.0, 0.5, 0.5], [0.0, 0.25, 0.25, 0.25]], device="cuda"
        )
        args = (state, actions, timestep, None, None, None)
        eager = flow._predict_velocity_core(*args)
        compiled = torch.compile(flow._predict_velocity_core, backend="eager", fullgraph=True)(*args)
        torch.testing.assert_close(compiled, eager)


class RTCFlowTest(unittest.TestCase):
    def test_delay_sampling_is_inclusive_and_validation_is_per_batch(self):
        flow = _bare_flow(_rtc_config())
        with mock.patch("torch.randint", return_value=torch.tensor([0, 2])) as randint:
            sampled = flow.sample_rtc_delay(2, torch.device("cpu"))
        randint.assert_called_once_with(0, 3, (2,), dtype=torch.long, device=torch.device("cpu"))
        torch.testing.assert_close(sampled, torch.tensor([0, 2]))

        torch.testing.assert_close(
            flow._normalize_rtc_delay(torch.tensor([0, 2]), 2, torch.device("cpu")),
            torch.tensor([0, 2]),
        )
        for bad in (torch.tensor([-1, 0]), torch.tensor([0, 3]), torch.tensor([0.0, 1.0])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                flow._normalize_rtc_delay(bad, 2, torch.device("cpu"))

        non_rtc = _bare_flow(_rtc_config(training_time_rtc=False, rtc_max_delay=0))
        with self.assertRaisesRegex(ValueError, "training_time_rtc"):
            non_rtc._normalize_rtc_delay(1, 2, torch.device("cpu"))

    def test_training_prefix_is_clean_and_prefix_loss_is_zero(self):
        config = _rtc_config(proj_width=4, adanorm_time=False)
        flow = _bare_flow(config)
        flow.action_out_proj = nn.Linear(4, 3, bias=False)
        nn.init.zeros_(flow.action_out_proj.weight)

        recorded = {}

        def build_prefix(*args, **kwargs):
            return torch.zeros(2, 1, 4), torch.ones(2, 1, dtype=torch.bool), torch.zeros(3, 2, 1)

        def embed_suffix(state, noisy_actions, timestep):
            recorded["actions"] = noisy_actions.detach().clone()
            recorded["timestep"] = timestep.detach().clone()
            suffix = torch.zeros(2, 5, 4)
            return torch.zeros(2, 4), suffix, torch.ones(2, 5, dtype=torch.bool), torch.zeros(2, 5, dtype=torch.bool)

        class Joint(nn.Module):
            def forward(self, inputs_embeds, **kwargs):
                return (None, inputs_embeds[1]), None

        flow._build_action_prefix = build_prefix
        flow.embed_suffix = embed_suffix
        flow.qwenvl_with_expert = Joint()

        actions = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        noise = torch.full_like(actions, 10.0)
        time = torch.tensor([0.5, 0.25])
        losses, postfix = flow.forward(
            state=torch.zeros(2, 3),
            actions=actions,
            input_ids=torch.zeros(2, 1, dtype=torch.long),
            attention_mask=torch.ones(2, 1, dtype=torch.bool),
            noise=noise,
            time=time,
            rtc_delay=torch.tensor([0, 2]),
            return_postfix_mask=True,
        )

        torch.testing.assert_close(recorded["actions"][1, :2], actions[1, :2])
        torch.testing.assert_close(recorded["timestep"][1], torch.tensor([0.0, 0.0, 0.25, 0.25]))
        self.assertTrue(torch.equal(losses[1, :2], torch.zeros_like(losses[1, :2])))
        self.assertTrue(torch.equal(postfix[1], torch.tensor([False, False, True, True])))

    def _sampling_flow(self, *, rtc=True):
        config = _rtc_config(training_time_rtc=rtc, rtc_max_delay=2 if rtc else 0)
        flow = _bare_flow(config)
        flow._build_action_prefix = lambda *args, **kwargs: (
            torch.zeros(2, 1, 3),
            torch.ones(2, 1, dtype=torch.bool),
            torch.zeros(3, 2, 1),
        )
        flow._prefix_forward_core = lambda *args, **kwargs: None

        def predict_velocity(this, state, prefix_pad_masks, past_key_values, x_t, timestep, timing=None):
            this.seen_timestep = timestep.detach().clone()
            return torch.ones_like(x_t)

        flow.predict_velocity = types.MethodType(predict_velocity, flow)
        return flow

    def test_eager_sampling_pins_mixed_prefix_and_inactive_dimensions(self):
        flow = self._sampling_flow()
        noise = torch.zeros(2, 4, 3)
        prefix = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        result = flow.sample_actions(
            state=torch.zeros(2, 3),
            input_ids=torch.zeros(2, 1, dtype=torch.long),
            attention_mask=torch.ones(2, 1, dtype=torch.bool),
            noise=noise,
            action_mask=torch.tensor([[1, 0, 1], [1, 0, 1]]),
            action_prefix=prefix,
            rtc_delay=torch.tensor([0, 2]),
        )
        torch.testing.assert_close(result[1, :2], prefix[1, :2])
        self.assertTrue(torch.isfinite(result[:, 2:]).all())
        self.assertEqual(result.shape, (2, 4, 3))
        self.assertEqual(flow.seen_timestep.shape, (2, 4))
        self.assertTrue(torch.equal(flow.seen_timestep[1, :2], torch.zeros(2)))

    def test_no_rtc_inputs_preserve_legacy_sampling_path(self):
        rtc_flow = self._sampling_flow(rtc=True)
        legacy_flow = self._sampling_flow(rtc=False)
        kwargs = dict(
            state=torch.zeros(2, 3),
            input_ids=torch.zeros(2, 1, dtype=torch.long),
            attention_mask=torch.ones(2, 1, dtype=torch.bool),
            noise=torch.zeros(2, 4, 3),
        )
        torch.testing.assert_close(rtc_flow.sample_actions(**kwargs), legacy_flow.sample_actions(**kwargs))
        self.assertEqual(rtc_flow.seen_timestep.ndim, 1)

    def test_sampling_rejects_missing_or_malformed_prefix(self):
        flow = self._sampling_flow()
        kwargs = dict(
            state=torch.zeros(2, 3),
            input_ids=torch.zeros(2, 1, dtype=torch.long),
            attention_mask=torch.ones(2, 1, dtype=torch.bool),
            noise=torch.zeros(2, 4, 3),
            rtc_delay=1,
        )
        with self.assertRaisesRegex(ValueError, "action_prefix is required"):
            flow.sample_actions(**kwargs)
        with self.assertRaisesRegex(ValueError, "action_prefix must have shape"):
            flow.sample_actions(**kwargs, action_prefix=torch.zeros(2, 3, 3))


class _LossFlow(nn.Module):
    def forward(self, actions, rtc_delay, return_postfix_mask, **kwargs):
        delay = rtc_delay
        postfix = torch.arange(actions.shape[1])[None, :] >= delay[:, None]
        losses = torch.full_like(actions, 2.0) * postfix[:, :, None]
        return losses, postfix


class _SampleFlow(nn.Module):
    def sample_actions(self, **kwargs):
        self.received_prefix = kwargs["action_prefix"]
        self.received_delay = kwargs["rtc_delay"]
        return kwargs["action_prefix"]


class RTCLossNormalizationTest(unittest.TestCase):
    def test_postfix_and_action_masks_use_actual_element_denominator(self):
        model = Tau0VLAModel.__new__(Tau0VLAModel)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(
            loss_type="fm",
            action_dim=3,
            use_action_mask_loss=True,
            vla_inactive_input_zero=False,
        )
        model.flow_matching = _LossFlow()
        output = model.forward(
            action=torch.zeros(2, 4, 3),
            state=torch.zeros(2, 3),
            action_mask=torch.tensor([[1, 0, 0], [1, 1, 0]]),
            rtc_delay=torch.tensor([0, 2]),
        )
        self.assertEqual(output.loss.item(), 2.0)

    def test_model_sample_action_forwards_rtc_inputs_and_keeps_fixed_shape(self):
        model = Tau0VLAModel.__new__(Tau0VLAModel)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(action_dim=2)
        model.flow_matching = _SampleFlow()
        prefix = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        delay = torch.tensor([1, 2])
        result = model.sample_action(
            {
                "state": torch.zeros(2, 3),
                "input_ids": torch.zeros(2, 1, dtype=torch.long),
                "attention_mask": torch.ones(2, 1, dtype=torch.bool),
                "action_prefix": prefix,
                "rtc_delay": delay,
            }
        )
        self.assertIs(model.flow_matching.received_prefix, prefix)
        self.assertIs(model.flow_matching.received_delay, delay)
        self.assertEqual(result.shape, (2, 4, 2))


if __name__ == "__main__":
    unittest.main()
