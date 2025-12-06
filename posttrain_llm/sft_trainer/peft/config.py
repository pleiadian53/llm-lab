"""PEFT Configuration Module."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Union


class PEFTMethod(str, Enum):
    """Supported PEFT methods."""
    LORA = "lora"
    VERA = "vera"
    DORA = "dora"
    QLORA = "qlora"
    ADALORA = "adalora"
    IA3 = "ia3"
    PROMPT_TUNING = "prompt_tuning"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"


@dataclass
class PEFTConfig:
    """Configuration for PEFT methods.
    
    This class provides a unified interface for configuring various PEFT methods.
    
    Attributes:
        method: PEFT method to use
        r: Rank for low-rank adaptation methods
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply PEFT to
        bias: Bias type ('none', 'all', or 'lora_only')
        task_type: Task type for the model
        use_rslora: Use Rank-Stabilized LoRA
        use_dora: Enable DoRA
        init_lora_weights: Initialization method
        load_in_4bit: Load model in 4-bit (QLoRA)
        load_in_8bit: Load model in 8-bit
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_quant_type: Quantization type
        bnb_4bit_use_double_quant: Use double quantization
        num_virtual_tokens: For prompt/prefix tuning
        modules_to_save: Additional modules to save
    """
    
    # Common parameters
    method: PEFTMethod = PEFTMethod.LORA
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    
    # LoRA variants
    use_rslora: bool = False
    use_dora: bool = False
    init_lora_weights: Union[bool, str] = True
    
    # VeRA specific
    vera_dropout: float = 0.0
    d_initial: float = 0.1
    projection_prng_key: int = 0
    save_projection: bool = True
    
    # Quantization (QLoRA)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # AdaLoRA specific
    target_r: Optional[int] = None
    init_r: Optional[int] = None
    
    # Prompt tuning
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "random"
    prompt_tuning_init_text: Optional[str] = None
    
    # IA3 specific
    feedforward_modules: Optional[List[str]] = None
    
    # General
    modules_to_save: Optional[List[str]] = None
    layers_to_transform: Optional[List[int]] = None
    
    def to_peft_config(self):
        """Convert to appropriate PEFT library config object."""
        from peft import TaskType
        
        task_type_enum = getattr(TaskType, self.task_type)
        target = self.target_modules or self._get_default_target_modules()
        
        if self.method in (PEFTMethod.LORA, PEFTMethod.DORA, PEFTMethod.QLORA):
            from peft import LoraConfig
            return LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target,
                bias=self.bias,
                task_type=task_type_enum,
                use_rslora=self.use_rslora,
                use_dora=(self.method == PEFTMethod.DORA) or self.use_dora,
                init_lora_weights=self.init_lora_weights,
                modules_to_save=self.modules_to_save,
            )
        
        elif self.method == PEFTMethod.VERA:
            from peft import VeraConfig
            return VeraConfig(
                r=self.r,
                target_modules=target,
                projection_prng_key=self.projection_prng_key,
                save_projection=self.save_projection,
                vera_dropout=self.vera_dropout,
                d_initial=self.d_initial,
                bias=self.bias,
                modules_to_save=self.modules_to_save,
            )
        
        elif self.method == PEFTMethod.ADALORA:
            from peft import AdaLoraConfig
            return AdaLoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target,
                bias=self.bias,
                task_type=task_type_enum,
                target_r=self.target_r or self.r,
                init_r=self.init_r or self.r,
                modules_to_save=self.modules_to_save,
            )
        
        elif self.method == PEFTMethod.IA3:
            from peft import IA3Config
            return IA3Config(
                target_modules=target,
                feedforward_modules=self.feedforward_modules or [],
                task_type=task_type_enum,
                modules_to_save=self.modules_to_save,
            )
        
        elif self.method == PEFTMethod.PROMPT_TUNING:
            from peft import PromptTuningConfig, PromptTuningInit
            init_method = (
                PromptTuningInit.TEXT 
                if self.prompt_tuning_init == "text" 
                else PromptTuningInit.RANDOM
            )
            return PromptTuningConfig(
                task_type=task_type_enum,
                num_virtual_tokens=self.num_virtual_tokens,
                prompt_tuning_init=init_method,
                prompt_tuning_init_text=self.prompt_tuning_init_text,
            )
        
        elif self.method == PEFTMethod.PREFIX_TUNING:
            from peft import PrefixTuningConfig
            return PrefixTuningConfig(
                task_type=task_type_enum,
                num_virtual_tokens=self.num_virtual_tokens,
            )
        
        elif self.method == PEFTMethod.P_TUNING:
            from peft import PromptEncoderConfig
            return PromptEncoderConfig(
                task_type=task_type_enum,
                num_virtual_tokens=self.num_virtual_tokens,
            )
        
        else:
            raise ValueError(f"Unsupported PEFT method: {self.method}")
    
    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules for common architectures."""
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def get_quantization_config(self):
        """Get quantization config for QLoRA."""
        if not (self.load_in_4bit or self.load_in_8bit):
            return None
        
        import torch
        from transformers import BitsAndBytesConfig
        
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )
    
    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "PEFTConfig":
        """Create config from preset.
        
        Available presets:
            - lora_default: Standard LoRA (r=16, alpha=32)
            - lora_high_rank: High-rank LoRA (r=64, alpha=128)
            - dora: DoRA configuration
            - olora: OLoRA (orthogonal initialization)
            - qlora_4bit: QLoRA with 4-bit quantization
            - qlora_8bit: QLoRA with 8-bit quantization
            - vera: VeRA configuration
            - adalora: AdaLoRA configuration
        
        Args:
            preset: Preset name
            **kwargs: Override specific parameters
            
        Returns:
            PEFTConfig instance
        """
        presets = {
            "lora_default": {
                "method": PEFTMethod.LORA,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            },
            "lora_high_rank": {
                "method": PEFTMethod.LORA,
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.1,
            },
            "dora": {
                "method": PEFTMethod.DORA,
                "r": 16,
                "lora_alpha": 32,
                "use_dora": True,
            },
            "olora": {
                "method": PEFTMethod.LORA,
                "r": 16,
                "lora_alpha": 32,
                "init_lora_weights": "olora",
            },
            "qlora_4bit": {
                "method": PEFTMethod.QLORA,
                "r": 16,
                "lora_alpha": 32,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
            },
            "qlora_8bit": {
                "method": PEFTMethod.QLORA,
                "r": 16,
                "lora_alpha": 32,
                "load_in_8bit": True,
            },
            "vera": {
                "method": PEFTMethod.VERA,
                "r": 256,
                "vera_dropout": 0.0,
                "d_initial": 0.1,
            },
            "adalora": {
                "method": PEFTMethod.ADALORA,
                "r": 16,
                "lora_alpha": 32,
                "target_r": 8,
                "init_r": 12,
            },
            "ia3": {
                "method": PEFTMethod.IA3,
            },
            "prompt_tuning": {
                "method": PEFTMethod.PROMPT_TUNING,
                "num_virtual_tokens": 20,
            },
            "prefix_tuning": {
                "method": PEFTMethod.PREFIX_TUNING,
                "num_virtual_tokens": 20,
            },
        }
        
        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset: {preset}. Available: {available}")
        
        config_dict = presets[preset].copy()
        config_dict.update(kwargs)
        
        return cls(**config_dict)
