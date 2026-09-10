# SPDX-License-Identifier: Apache-2.0
# LINQ: int4 group-scaled experts x dynamic per-token int8 activations on CUDA.
"""vLLM's CompressedTensorsW4A8Int8MoEMethod is CPU-only (Arm dynamic 4-bit kernels) and its GPU
W4A8 MoE kernel takes fp8 activations. The llm-compressor W4A8 scheme (int4 g128 symmetric
weights, dynamic per-token symmetric int8 activations -- the dense models' rows) therefore had
no GPU expert kernel. This method reuses the WNA16 MoE loader (packed weights + group scales)
and runs the Triton wna16 kernel's int4_w4a8 branch: int8 x int8 dot, fp32 group and token
scales, the same arithmetic Marlin-QQQ does for the dense W4A8 layers."""
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig, int4_w4a8_moe_quant_config

from .compressed_tensors_moe_wna16 import CompressedTensorsWNA16MoEMethod

logger = init_logger(__name__)


def is_linq_w4a8_int(quant_config, weight_quant: QuantizationArgs, input_quant: QuantizationArgs | None) -> bool:
    """int4 group/channel weights x dynamic per-token INT8 activations on a CUDA device. vLLM's
    _is_dynamic_token_w4a8_int does not look at the activation type (it also matches fp8 tokens,
    which the SM90 fp8 branch catches first); the int check keeps this method off fp8 configs."""
    from vllm.platforms import current_platform

    return (
        input_quant is not None
        and quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant)
        and input_quant.type == QuantizationType.INT.value
        and current_platform.is_cuda()
    )


class LinqW4A8IntMoEMethod(CompressedTensorsWNA16MoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(weight_quant, input_quant, moe, layer_name)
        assert self.num_bits == 4, f"int4 weights only, got {self.num_bits}"
        assert weight_quant.type == QuantizationType.INT.value and weight_quant.symmetric
        assert input_quant is not None and input_quant.num_bits == 8
        assert input_quant.type == QuantizationType.INT.value, "int8 activations (fp8 has its own method)"
        assert input_quant.strategy == QuantizationStrategy.TOKEN.value and input_quant.dynamic
        assert input_quant.symmetric, "int4 x int8 path: symmetric activations only"
        logger.info_once("LINQ: int4 x int8 experts on the Triton wna16 kernel (group %d)", self.group_size)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return int4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale, group_size=self.group_size
        )
