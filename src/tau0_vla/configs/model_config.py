"""``ModelArguments`` — architecture, which parameter groups train, LoRA, and
the FAST vocabulary extension.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Model architecture and parameter-control settings.

    Holds every option governing VLM / VLA model loading, multimodal module
    freezing, LoRA finetuning, and FAST token extension. ``HfArgumentParser``
    parses these out of a YAML file or the command line, and they are then handed
    to :class:`~tau0_vla.models.model_builder.ModelBuilder`.

    Attributes:
        model_name_or_path: Path to a pretrained model, or a Hugging Face Hub name.
        vlm_model_type: VLM backbone. ``"qwen3.5"`` is what the shipped VLA configs
            use — it selects the Qwen3.5 M-RoPE index function. The plain-VLM path
            (``vla_model_type=None``) instead takes one of ``"qwen3vl"`` |
            ``"qwen3vl_moe"`` | ``"qwen2vl"`` | ``"qwen2.5vl"``.
        vla_model_type: VLA model type. ``None`` for a plain VLM, or ``"tau0_vla"``.
        tune_mm_llm: Whether to train the language-model layers. Frozen when ``False``.
        tune_mm_mlp: Whether to train the vision-language projector (the merger).
        tune_mm_vision: Whether to train the vision encoder.
        tune_vla_dit: Whether to train the VLA's DiT action head. Only takes effect
            when ``vla_model_type`` is not ``None``.
        add_fast_special_tokens: Whether to add FAST discrete-action tokens to the
            vocabulary.
        fast_vocab_size: How many FAST tokens to add. Defaults to 2048.

    Example:
        >>> from tau0_vla.configs.model_config import ModelArguments
        >>> args = ModelArguments(
        ...     model_name_or_path="/path/to/Qwen3-VL-2B",
        ...     vlm_model_type="qwen3vl",
        ...     tune_mm_llm=True,
        ...     lora_enable=False,
        ... )
    """

    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    vlm_model_type: Optional[str] = field(
        default="qwen3vl",
        metadata={
            "help": "VLM Model type. `qwen3.5` for the shipped VLA route; "
            "[`qwen3vl`, `qwen3vl_moe`, `qwen2vl`, `qwen2.5vl`] for a plain VLM."
        },
    )
    vla_model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "VLA Model type, chosen from [`None`, `tau0_vla`]. `None` for only VLM. "
            "The pre-rename spelling `tau0_vla` is accepted as an alias so released "
            "checkpoints load."
        },
    )

    # vlm backbone
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # vla dit
    tune_vla_dit: bool = field(default=False)

    # Enable lm_head for VLM loss computation (required for VQA / FAST co-training)
    use_lm_head: bool = field(
        default=False,
        metadata={"help": "Keep lm_head in the VLM for computing cross-entropy loss during co-training."},
    )

    # FAST
    add_fast_special_tokens: Optional[bool] = field(default=True)
    fast_vocab_size: Optional[int] = field(default=2048)

    # flow matching
    loss_type: str = field(
        default="fm", metadata={"help": "Loss type for VLA action prediction, e.g. 'fm' for flow matching."}
    )
    num_steps: int = field(default=10, metadata={"help": "Number of denoising steps for inference in flow matching."})
    training_time_rtc: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable training-time real-time chunking (RTC): condition the action head on a "
                "uniformly sampled clean action prefix. Adds no learnable parameters."
            )
        },
    )
    rtc_max_delay: int = field(
        default=0,
        metadata={
            "help": (
                "Largest action-prefix delay sampled by training-time RTC, in action timesteps. "
                "When RTC is enabled this must satisfy 1 <= rtc_max_delay < action_horizon."
            )
        },
    )
    vlm_causal: bool = field(
        default=True, metadata={"help": "Use causal (True) or bidirectional (False) prefix mask for action path."}
    )
    tau_vla_prefix_flash_backend: bool = field(
        default=False,
        metadata={
            "help": (
                "Route Tau0VLA VLM-prefix self-attention through SDPA's Flash backend "
                "(no explicit mask, is_causal=True). Requires vlm_causal=True and "
                "right-padded prefix masks. Action/suffix attention still uses explicit masks."
            )
        },
    )
    zero_state_emb: bool = field(
        default=False, metadata={"help": "Zero out state embedding in flow matching action head (e.g. for LIBERO)."}
    )
    use_action_mask_loss: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, use batch['action_mask'] to exclude inactive/padded action dimensions from VLA loss. "
                "If False, average VLA loss over all action dimensions."
            )
        },
    )
    vla_inactive_input_zero: bool = field(
        default=False,
        metadata={
            "help": (
                "Scheme-C inactive-dim input convention. If True, the flow-matching NETWORK INPUT x_t is "
                "zeroed on inactive action dims (mask from batch['action_mask']) at training time — the "
                "exact-0 pattern becomes an explicit per-sample mask signal — and sampling re-zeroes those "
                "dims every integration step. If False (default, scheme A), training input keeps t*noise on "
                "inactive dims and sampling resets them to t*noise per step when action_mask is provided. "
                "MUST stay consistent between a checkpoint's training and its eval/deploy sampling."
            )
        },
    )
