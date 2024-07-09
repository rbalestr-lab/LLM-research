"""LoRA Experiment Library.

Prerequisite:
  * pip install peft=0.10.0

To use:

from loraexp_lib import LoraConfigExp, get_peft_model_exp

config = LoraConfigExp(
    r=args.lora_rank,
    lora_alpha=args.lora_rank,
    target_modules=[...],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    use_lora0=args.lora0,
    m=args.mixture if args.mixture != 0 else None,
    re_lora=args.re_lora,
    using_scaling_beta=args.use_scaling_beta,
)
print(config)
model.backbone.requires_grad_(False)
model = get_peft_model_exp(model, config)

Each class/method in this library is a copy of the corresponding class/method in
peft library. The only changed part is enclosed by the comments
"# ===== EDIT START ===== " and "# ===== EDIT END ===== ". This makes it easier
to merge it into the peft library in the future.
"""

import dataclasses
from enum import Enum
from itertools import chain
import math
import re
from typing import Any, Optional, Union
import warnings

from peft.config import PeftConfig
from peft.mapping import _prepare_prompt_learning_config
from peft.peft_model import PeftModel
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.model import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import PeftType, get_quantization_config
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.utils import PushToHubMixin

_DEBUG = False


@dataclasses.dataclass
class LoraConfigExp(LoraConfig):
  """LoraConfig for LoRA Experiment."""
  m: int | None = dataclasses.field(
      default=None,
      metadata={"help": "Mixture of LoRA # of mixtures"}
  )
  use_lora0: bool = dataclasses.field(
      default=False,
      metadata={"help": "Single matrix adaption, a.k.a. no LoRA"}
  )
  re_lora: str = dataclasses.field(
      default='x',
      metadata={"help": "ReLoRA type of recurrence"}
  )
  use_scaling_beta: bool = dataclasses.field(
      default=False,
      metadata={"help": "Learnable scaling beta"}
  )


class LoraExperimentType(Enum):
  NO_EXPERIMENT = 1
  LORA0 = 2
  MIXTURE_OF_LORA = 3
  RE_LORA = 4

_SUPPORTED_RE_LORA_EXPERIMENTS = [
    "x",
    "x^2",        # lora_B(lora_A(x * x * sign(x))))
    "sqrt(x)",    # lora_B(lora_A(sqrt(x) * sign(x))))
    "baabba",     # BAA^TB^TBAx, the idea is BA(BAx), but BAx and x may not be
                  # in the same shape. Applying A^TB^T to restore to the shape
                  # of the input.
    "mask",       # mask out values < a given threshold.
    "ba+baabba",  # BAx + x, to simulate residual connection.
]

class LoraLayerExp(LoraLayer):
  """LoraLayer for LoRA Experiment."""
  # All names of layers that may contain (trainable) adapter weights
  # ===== EDIT START =====
  adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_A1", "lora_B1", "lora_A2", "lora_B2", "lora_A3", "lora_B3", "scaling_beta")
  # All names of other parameters that may contain adapter-related parameters
  other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "experiment_type", "m", "re_lora", "use_scaling_beta")
  # ===== EDIT END =====

  def __init__(self, base_layer: nn.Module, **kwargs) -> None:
    if _DEBUG:
      print("  --> LoraLayerExp:__init__")

    self.base_layer = base_layer
    self.r = {}
    # ===== EDIT START =====
    self.m = {}
    self.experiment_type = {}
    self.re_lora = {}
    self.use_scaling_beta = {}
    # ===== EDIT END =====
    self.lora_alpha = {}
    self.scaling = {}
    self.lora_dropout = nn.ModuleDict({})
    self.lora_A = nn.ModuleDict({})
    self.lora_B = nn.ModuleDict({})
    # ===== EDIT START =====
    self.lora_A1 = nn.ModuleDict({})
    self.lora_B1 = nn.ModuleDict({})
    self.lora_A2 = nn.ModuleDict({})
    self.lora_B2 = nn.ModuleDict({})
    self.lora_A3 = nn.ModuleDict({})
    self.lora_B3 = nn.ModuleDict({})
    self.A = {0: self.lora_A, 1: self.lora_A1, 2: self.lora_A2, 3: self.lora_A3}
    self.B = {0: self.lora_B, 1: self.lora_B1, 2: self.lora_B2, 3: self.lora_B3}
    self.scaling_beta = nn.ParameterDict({})
    # ===== EDIT END =====
    # For Embedding layer
    self.lora_embedding_A = nn.ParameterDict({})
    self.lora_embedding_B = nn.ParameterDict({})
    # Mark the weight as unmerged
    self._disable_adapters = False
    self.merged_adapters = []
    self.use_dora: dict[str, bool] = {}
    self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
    self._caches: dict[str, Any] = {}
    self.kwargs = kwargs

    base_layer = self.get_base_layer()
    if isinstance(base_layer, nn.Linear):
      in_features, out_features = base_layer.in_features, base_layer.out_features
    elif isinstance(base_layer, nn.Conv2d):
      in_features, out_features = base_layer.in_channels, base_layer.out_channels
    elif isinstance(base_layer, nn.Embedding):
      in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
    elif isinstance(base_layer, Conv1D):
      in_features, out_features = (
          base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
      )
    elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
      # QuantLinear
      in_features, out_features = base_layer.infeatures, base_layer.outfeatures
    elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
      # Megatron ColumnParallelLinear,RowParallelLinear
      in_features, out_features = base_layer.input_size, base_layer.output_size
    elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
      # AQLM QuantLinear
      in_features, out_features = base_layer.in_features, base_layer.out_features
    elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
      # Awq layers
      in_features, out_features = base_layer.in_features, base_layer.out_features
    else:
      raise ValueError(f"Unsupported layer type {type(base_layer)}")

    self.in_features = in_features
    self.out_features = out_features

  def update_layer(
      self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False,
      # ===== EDIT START =====
      m: int | None = None, use_lora0: bool = False, re_lora: str | None = None,
      use_scaling_beta: bool = False,
      # ===== EDIT END =====
  ):
    # ===== EDIT START =====
    if _DEBUG:
      print("  --> LoraLayerExp:update_layer")
    # ===== EDIT END =====

    # This code works for linear layers, override for other layer types
    if r <= 0:
      raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

    # ===== EDIT START =====
    if m is not None and m <= 1:
      raise ValueError(f"`m` should be a positive integer value greater than 1 if defined, but got {m}")

    if m is not None and m > 4:
      raise ValueError(f"`m` should be a positive integer value less than 5, but got {m}")

    if re_lora is not None and re_lora not in _SUPPORTED_RE_LORA_EXPERIMENTS:
      raise ValueError(f"`re_lora` is set but is not supported {re_lora}")

    if m is not None and m > 1 and use_lora0:
      raise ValueError(f"`m` is defined but `use_lora0` is also set")

    if re_lora is not None and use_lora0:
      raise ValueError(f"`re_lora` is defined but `use_lora0` is also set")
    # ===== EDIT END =====

    self.r[adapter_name] = r
    # ===== EDIT START =====
    self.m[adapter_name] = m
    self.re_lora[adapter_name] = re_lora
    # ===== EDIT END =====
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
      lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
      lora_dropout_layer = nn.Identity()

    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
    # Actual trainable parameters
    # ===== EDIT START =====
    self.use_scaling_beta[adapter_name] = use_scaling_beta
    # (0.002 + 0.001) x 100.0 (later at forward()) is effectively a scaling
    # factor 3.0 to start with.
    self.scaling_beta[adapter_name] = nn.Parameter(torch.tensor([[0.002]]))
    if use_lora0:
      print(f" ********** LoRA0({r}) for layer: {adapter_name} **********")
      self.experiment_type[adapter_name] = LoraExperimentType.LORA0
      self.lora_A[adapter_name] = nn.Linear(
          self.in_features, self.out_features, bias=False)
    elif m is not None and m > 1:
      print(f" ********** MoLoRA({r}x{m}){re_lora} for layer: {adapter_name} **********")
      self.experiment_type[adapter_name] = LoraExperimentType.MIXTURE_OF_LORA
      for i in range(m):
        self.A[i][adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.B[i][adapter_name] = nn.Linear(r, self.out_features, bias=False)
    elif re_lora is not None:
      print(f" ********** ReLoRA({r}){re_lora} for layer: {adapter_name} **********")
      self.experiment_type[adapter_name] = LoraExperimentType.RE_LORA
      self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
      self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
    else:
      print(f" ********** LoRA({r}) for layer: {adapter_name} **********")
      self.experiment_type[adapter_name] = LoraExperimentType.NO_EXPERIMENT
      # ===== EDIT END =====
      self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
      self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
    if use_rslora:
      self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    else:
      self.scaling[adapter_name] = lora_alpha / r

    if init_lora_weights == "loftq":
      self.loftq_init(adapter_name)
    elif init_lora_weights:
      self.reset_lora_parameters(adapter_name, init_lora_weights)

    # check weight and qweight (for GPTQ)
    for weight_name in ("weight", "qweight"):
      weight = getattr(self.get_base_layer(), weight_name, None)
      if weight is not None:
        # the layer is already completely initialized, this is an update
        if weight.dtype.is_floating_point or weight.dtype.is_complex:
          self.to(weight.device, dtype=weight.dtype)
        else:
          self.to(weight.device)
        break

    if use_dora:
      self.dora_init(adapter_name)
      self.use_dora[adapter_name] = True
    else:
      self.use_dora[adapter_name] = False

    self.set_adapter(self.active_adapters)

  def reset_lora_parameters(self, adapter_name, init_lora_weights):
    # ===== EDIT START =====
    if _DEBUG:
      print("  --> LoraLayerExp:reset_lora_parameters")
    # ===== EDIT END =====

    if init_lora_weights is False:
      return

    if adapter_name in self.lora_A.keys():
      if init_lora_weights is True:
        # initialize A the same way as the default for nn.Linear and B to zero
        # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
        # ===== EDIT START =====
        if self.experiment_type[adapter_name] == LoraExperimentType.MIXTURE_OF_LORA:
          for i in range(self.m[adapter_name]):
            nn.init.kaiming_uniform_(self.A[i][adapter_name].weight, a=math.sqrt(5))
        else:
          # ===== EDIT END =====
          nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
      elif init_lora_weights.lower() == "gaussian":
        # ===== EDIT START =====
        if self.experiment_type[adapter_name] == LoraExperimentType.MIXTURE_OF_LORA:
          for i in range(self.m[adapter_name]):
              nn.init.normal_(self.A[i][adapter_name].weight, std=1 / self.r[adapter_name])
        else:
          # ===== EDIT END =====
          nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
      else:
        raise ValueError(f"Unknown initialization {init_lora_weights=}")
      # ===== EDIT START =====
      if self.re_lora[adapter_name] == "baabba":
        branches = self.m[adapter_name] if self.m[adapter_name] else 1
        for i in range(branches):
          nn.init.kaiming_uniform_(
              self.B[i][adapter_name].weight, a=math.sqrt(5))
      elif self.experiment_type[adapter_name] == LoraExperimentType.MIXTURE_OF_LORA:
        for i in range(self.m[adapter_name]):
          nn.init.zeros_(self.B[i][adapter_name].weight)
      elif self.experiment_type[adapter_name] == LoraExperimentType.LORA0:
        nn.init.zeros_(self.lora_A[adapter_name].weight)
      else:
        # ===== EDIT END =====
        nn.init.zeros_(self.lora_B[adapter_name].weight)
    if adapter_name in self.lora_embedding_A.keys():
      # initialize a the same way as the default for nn.linear and b to zero
      nn.init.zeros_(self.lora_embedding_A[adapter_name])
      nn.init.normal_(self.lora_embedding_B[adapter_name])

  def loftq_init(self, adapter_name):
    # ===== EDIT START =====
    if self.experiment_type[adapter_name] != LoraExperimentType.NO_EXPERIMENT:
      raise ValueError(f"loftq_init is not supported at adapter: {adapter_name} which is of experiment type: {self.experiment_type[adapter_name]}")
    # ===== EDIT END =====

  def dora_init(self, adapter_name: str) -> None:
    # ===== EDIT START =====
    if self.experiment_type[adapter_name] != LoraExperimentType.NO_EXPERIMENT:
      raise ValueError(f"dora_init is not supported at adapter: {adapter_name} which is of experiment type: {self.experiment_type[adapter_name]}")
    # ===== EDIT END =====

  def _mixed_batch_forward(
      self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
  ) -> torch.Tensor:
    # ===== EDIT START =====
    if _DEBUG:
      print("  --> LoraLayerExp:_mixed_batch_forward")
    # ===== EDIT END =====

    # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
    # extra argument that allows mixing different adapters in the same batch at inference time.
    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype

    unique_adapters = set(adapter_names)
    sub_batch_indices_list = []
    for adapter in unique_adapters:
      sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

    for i, active_adapter in enumerate(unique_adapters):
      if active_adapter == "__base__":
        continue
      if active_adapter not in self.lora_A.keys():
        continue

      # ===== EDIT START =====
      if self.experiment_type[active_adapter] != LoraExperimentType.NO_EXPERIMENT:
        raise ValueError(f"_mixed_batch_forward is not supported at adapter: {active_adapter} which is of experiment type: {self.experiment_type[active_adapter]}")
      else:
        # ===== EDIT END =====
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
        # layer output
        sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
        lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
      result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

    return result


class LinearExp(nn.Module, LoraLayerExp):
  """Linear for LoRA Experiment."""
  def __init__(
      self,
      base_layer,
      adapter_name: str,
      r: int = 0,
      lora_alpha: int = 1,
      lora_dropout: float = 0.0,
      fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
      is_target_conv_1d_layer: bool = False,
      init_lora_weights: Union[bool, str] = True,
      use_rslora: bool = False,
      use_dora: bool = False,
      **kwargs,
  ) -> None:
    # ===== EDIT START =====
    if _DEBUG:
      print("  --> LinearExp:__init__")
    # ===== EDIT END =====

    super().__init__()
    # ===== EDIT START =====
    LoraLayerExp.__init__(self, base_layer, **kwargs)
    # ===== EDIT END =====
    self.fan_in_fan_out = fan_in_fan_out

    self._active_adapter = adapter_name
    self.update_layer(
        adapter_name,
        r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights=init_lora_weights,
        use_rslora=use_rslora,
        use_dora=use_dora,
        # ===== EDIT START =====
        m=kwargs["m"],
        use_lora0=kwargs["use_lora0"],
        re_lora=kwargs["re_lora"],
        use_scaling_beta=kwargs["use_scaling_beta"],
        # ===== EDIT END =====
    )
    self.is_target_conv_1d_layer = is_target_conv_1d_layer

  def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge the active adapter weights into the base weights

    Args:
        safe_merge (`bool`, *optional*):
            If True, the merge operation will be performed in a copy of the original weights and check for NaNs
            before merging the weights. This is useful if you want to check if the merge operation will produce
            NaNs. Defaults to `False`.
        adapter_names (`list[str]`, *optional*):
            The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
            to `None`.
    """
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
      # no adapter to merge
      return

    for active_adapter in adapter_names:
      # ===== EDIT START =====
      if self.experiment_type[active_adapter] != LoraExperimentType.NO_EXPERIMENT:
        raise ValueError(f"merge is not supported at adapter: {active_adapter} which is of experiment type: {self.experiment_type[active_adapter]}")
      # ===== EDIT END =====

      if active_adapter in self.lora_A.keys():
        base_layer = self.get_base_layer()
        if safe_merge:
          # Note that safe_merge will be slower than the normal merge
          # because of the copy operation.
          orig_weights = base_layer.weight.data.clone()
          delta_weight = self.get_delta_weight(active_adapter)
          if not self.use_dora[active_adapter]:
            orig_weights = orig_weights + delta_weight
          else:
            # handle dora
            # since delta_weight already includes scaling, set it to 1 here
            weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
            # We need to cache weight_norm because it has to be based on the original weights. We
            # cannot calculate it on the fly based on the merged weights when unmerging because its a
            # different value
            self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
            dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
            orig_weights = dora_factor.view(-1, 1) * (orig_weights + delta_weight)

          if not torch.isfinite(orig_weights).all():
            raise ValueError(
                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
            )

          base_layer.weight.data = orig_weights
        else:
          delta_weight = self.get_delta_weight(active_adapter)
          if not self.use_dora[active_adapter]:
            base_layer.weight.data = base_layer.weight.data + delta_weight
          else:
            # handle dora
            # since delta_weight already includes scaling, set it to 1 here
            weight_norm = self._get_weight_norm(base_layer.weight, delta_weight, scaling=1).detach()
            # We need to cache weight_norm because it has to be based on the original weights. We
            # cannot calculate it on the fly based on the merged weights when unmerging because its a
            # different value
            self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
            dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
            new_weight = dora_factor.view(-1, 1) * (base_layer.weight.data + delta_weight)
            base_layer.weight.data = new_weight

        self.merged_adapters.append(active_adapter)

  def unmerge(self) -> None:
    """
    This method unmerges all merged adapter layers from the base weights.
    """
    if not self.merged:
      warnings.warn("Already unmerged. Nothing to do.")
      return
    while len(self.merged_adapters) > 0:
      active_adapter = self.merged_adapters.pop()
      # ===== EDIT START =====
      if self.experiment_type[active_adapter] != LoraExperimentType.NO_EXPERIMENT:
        raise ValueError(f"merge is not supported at adapter: {active_adapter} which is of experiment type: {self.experiment_type[active_adapter]}")
      # ===== EDIT END =====

      if active_adapter in self.lora_A.keys():
        weight = self.get_base_layer().weight
        delta_weight = self.get_delta_weight(active_adapter)
        if not self.use_dora[active_adapter]:
          weight.data -= delta_weight
        else:
          weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
          dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
          weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
          weight.data = weight_orig

  def get_delta_weight(self, adapter) -> torch.Tensor:
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    # ===== EDIT START =====
    if self.experiment_type[adapter] != LoraExperimentType.NO_EXPERIMENT:
      raise ValueError(f"merge is not supported at adapter: {adapter} which is of experiment type: {self.experiment_type[adapter]}")
    # ===== EDIT END =====

    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
      weight_A = weight_A.float()
      weight_B = weight_B.float()

    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

    if cast_to_fp32:
      output_tensor = output_tensor.to(dtype=dtype)

      # cast back the weights
      self.lora_A[adapter].weight.data = weight_A.to(dtype)
      self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor

  def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    # ===== EDIT START =====
    if _DEBUG:
      print("  --> LinearExp:forward")
    # ===== EDIT END =====

    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
      if self.merged:
        self.unmerge()
      result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
      result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
      result = self.base_layer(x, *args, **kwargs)
    else:
      result = self.base_layer(x, *args, **kwargs)
      torch_result_dtype = result.dtype
      for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue

        # ===== EDIT START =====
        def apply_scale(active_adapter, x):
          if self.use_scaling_beta[active_adapter]:
            beta = self.scaling_beta[active_adapter]
            # Minium is 1e-3, so the scaling factor is effectively capped at 10.
            return x / 100. / (nn.functional.relu(beta) + 1e-3)
          else:
            return x

        def apply_layers(active_adapter, lora_A, lora_B, dropout, x):
          if self.re_lora[active_adapter] == "x":
            return lora_B(lora_A(dropout(x)))
          elif self.re_lora[active_adapter] == "x^2":
            return lora_B(lora_A(dropout(x * x * torch.sign(x))))
          elif self.re_lora[active_adapter] == "sqrt(x)":
            return lora_B(lora_A(dropout(torch.sqrt(torch.abs(x) + 1e-8) * torch.sign(x))))
          elif self.re_lora[active_adapter] == "baabba":
            z = lora_B(lora_A(dropout(x)))
            z = z @ lora_B.weight
            z = z @ lora_A.weight
            z = lora_B(lora_A(z))
            return z
          elif self.re_lora[active_adapter] == "mask":
            mask = torch.abs(x) > torch.quantile(
                torch.abs(x), float(self.r[active_adapter]) / self.in_features)
            return lora_B(lora_A(dropout(x * mask)))
          elif self.re_lora[active_adapter] == "ba+baabba":
            z1 = lora_B(lora_A(dropout(x)))
            z2 = z1 @ lora_B.weight @ lora_A.weight
            z2 = lora_B(lora_A(z2))
            scaling = torch.norm(z1, dim=(1, 2), p=2) / torch.maximum(
                torch.norm(z2, dim=(1, 2), p=2),
                torch.ones(1).to(x.device) * 1e-5
            )
            scaling = scaling.reshape(-1, 1, 1)
            z2 = z2 * scaling
            return z1 + z2
            # Uncomment below for the MoS(+Re).
            # mixture_list = [z1, z2]
            # mixture = torch.stack(mixture_list)
            # std = max(torch.max(mixture) / 32., 1e-5)
            # return torch.log(torch.mean(torch.exp(mixture / std), dim=0)) * std
          else:
            raise ValueError(f"Unknown re_lora type: {self.re_lora[active_adapter]}")

        if self.experiment_type[active_adapter] == LoraExperimentType.MIXTURE_OF_LORA:
          if self.use_dora[active_adapter]:
            raise ValueError(f"dora is not supported at adapter: {active_adapter} which is of experiment type: {self.experiment_type[active_adapter]}")

          mixture_list = []
          dropout = self.lora_dropout[active_adapter]
          scaling = self.scaling[active_adapter]

          x = x.to(self.lora_A[active_adapter].weight.dtype)
          for j in range(self.m[active_adapter]):
            lora_A = self.A[j][active_adapter]
            lora_B = self.B[j][active_adapter]

          mixture = torch.stack(mixture_list)
          std = max(torch.max(mixture) / 32., 1e-5)
          # Uncomment below for various std.
          # std = max(torch.max(mixture) / 1., 1e-5)
          # std = 1.
          lora_result = torch.log(torch.mean(torch.exp(mixture / std), dim=0)) * std
          result = result + apply_scale(active_adapter, lora_result * scaling)
        elif self.experiment_type[active_adapter] == LoraExperimentType.RE_LORA:
          if self.use_dora[active_adapter]:
            raise ValueError(f"dora is not supported at adapter: {active_adapter} which is of experiment type: {self.experiment_type[active_adapter]}")

          lora_A = self.lora_A[active_adapter]
          lora_B = self.lora_B[active_adapter]
          dropout = self.lora_dropout[active_adapter]
          scaling = self.scaling[active_adapter]

          lora_result = apply_layers(active_adapter, lora_A, lora_B, dropout, x)

          result = result + apply_scale(active_adapter, lora_result * scaling)
        elif self.experiment_type[active_adapter] == LoraExperimentType.LORA0:
          if self.use_dora[active_adapter]:
            raise ValueError(f"dora is not supported at adapter: {active_adapter} which is of experiment type: {self.experiment_type[active_adapter]}")

          lora_A = self.lora_A[active_adapter]
          dropout = self.lora_dropout[active_adapter]
          scaling = self.scaling[active_adapter]

          x = x.to(lora_A.weight.dtype)

          result = result + lora_A(dropout(x)) * scaling
        else:
          # ===== EDIT END =====
          lora_A = self.lora_A[active_adapter]
          lora_B = self.lora_B[active_adapter]
          dropout = self.lora_dropout[active_adapter]
          scaling = self.scaling[active_adapter]
          x = x.to(lora_A.weight.dtype)

          if not self.use_dora[active_adapter]:
            result = result + lora_B(lora_A(dropout(x))) * scaling
          else:
            x = dropout(x)
            result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

      result = result.to(torch_result_dtype)

    return result

  def __repr__(self) -> str:
    rep = super().__repr__()
    return "lora." + rep


def dispatch_default_exp(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfigExp,
    **kwargs,
) -> Optional[torch.nn.Module]:
  # ===== EDIT START =====
  if _DEBUG:
    print("  --> dispatch_default_exp")
  # ===== EDIT END =====

  new_module = None

  if isinstance(target, BaseTunerLayer):
    target_base_layer = target.get_base_layer()
  else:
    target_base_layer = target

  # ===== EDIT START =====
  assert isinstance(target_base_layer, torch.nn.Linear)
  # ===== EDIT END =====
  if kwargs["fan_in_fan_out"]:
    warnings.warn(
        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
        "Setting fan_in_fan_out to False."
    )
    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
  kwargs.update(lora_config.loftq_config)
  # ===== EDIT START =====
  new_module = LinearExp(target, adapter_name, **kwargs)
  # ===== EDIT END =====

  return new_module


class LoraModelExp(LoraModel):
  """LoraModel for LoRA Experiment."""

  def _create_and_replace(
      self,
      lora_config,
      adapter_name,
      target,
      target_name,
      parent,
      current_key,
  ):
    if _DEBUG:
      print("  --> LoraModelExp:_create_and_replace")

    if current_key is None:
      raise ValueError("Current Key shouldn't be `None`")

    # Regexp matching - Find key which matches current target_name in patterns provided
    pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
    target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
    r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
    alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

    kwargs = {
        "r": r,
        "m": lora_config.m,
        "use_lora0": lora_config.use_lora0,
        "re_lora": lora_config.re_lora,
        "use_scaling_beta": lora_config.use_scaling_beta,
        "lora_alpha": alpha,
        "lora_dropout": lora_config.lora_dropout,
        "fan_in_fan_out": lora_config.fan_in_fan_out,
        "init_lora_weights": lora_config.init_lora_weights,
        "use_rslora": lora_config.use_rslora,
        "use_dora": lora_config.use_dora,
        "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
        "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
    }

    quant_methods = ["gptq", "aqlm", "awq"]
    for quant_method in quant_methods:
      quantization_config = get_quantization_config(self.model, method=quant_method)
      assert quantization_config is None

    # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
    from peft.tuners.adalora import AdaLoraLayer

    if isinstance(target, LoraLayerExp) and not isinstance(target, AdaLoraLayer):
      target.update_layer(
          adapter_name,
          r,
          lora_alpha=alpha,
          lora_dropout=lora_config.lora_dropout,
          init_lora_weights=lora_config.init_lora_weights,
          use_rslora=lora_config.use_rslora,
          use_dora=lora_config.use_dora,
      )
    else:
      new_module = self._create_new_module(lora_config, adapter_name, target,
                                           **kwargs)
      if adapter_name != self.active_adapter:
        # adding an additional adapter: it is not automatically trainable
        new_module.requires_grad_(False)
      self._replace_module(parent, target_name, new_module, target)

  @staticmethod
  def _create_new_module(lora_config, adapter_name, target, **kwargs):
    if _DEBUG:
      print("  --> LoraModelExp:_create_new_module")

    # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
    # because the first match is always used. Therefore, the default layers should be checked last.
    dispatchers = [dispatch_default_exp]

    new_module = None
    for dispatcher in dispatchers:
      new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
      if new_module is not None:  # first match wins
        break

    if new_module is None:
      # no module could be matched
      raise ValueError(
          f"Target module {target} is not supported. Currently, only the following modules are supported: "
          "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
      )

    return new_module


class PeftModelExp(PeftModel):
  """PeftModel for LoRA Experiment."""

  def __init__(self, model: PreTrainedModel, peft_config: PeftConfig,
               adapter_name: str = "default") -> None:
    """Initializes the PerfModelExp.
    """
    # ===== EDIT START =====
    if _DEBUG:
      print("  --> PerfModelExp:__init__")
    # ===== EDIT END =====

    PushToHubMixin.__init__(self)
    torch.nn.Module.__init__(self)
    self.modules_to_save = None
    self.active_adapter = adapter_name
    self.peft_type = peft_config.peft_type
    # These args are special PEFT arguments that users can pass. They need to be
    # removed before passing them to forward.
    self.special_peft_forward_args = {"adapter_names"}

    self._is_prompt_learning = peft_config.is_prompt_learning
    if self._is_prompt_learning:
      self._peft_config = {adapter_name: peft_config}
      self.base_model = model
      self.add_adapter(adapter_name, peft_config)
    else:
      self._peft_config = None
      # ===== EDIT START =====
      assert peft_config.peft_type == PeftType.LORA
      cls = LoraModelExp
      # ===== EDIT END =====
      self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
      self.set_additional_trainable_modules(peft_config, adapter_name)

    if getattr(model, "is_gradient_checkpointing", True):
      model = self._prepare_model_for_gradient_checkpointing(model)

    # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
    # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
    # behavior we disable that in this line.
    if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
      self.base_model.config.pretraining_tp = 1


class PeftModelForCausalLMExp(PeftModelExp):
  """
  Peft model for causal language modeling.

  Args:
      model ([`~transformers.PreTrainedModel`]): Base transformer model.
      peft_config ([`PeftConfig`]): Peft config.


  Example:

      ```py
      >>> from transformers import AutoModelForCausalLM
      >>> from peft import PeftModelForCausalLM, get_peft_config

      >>> config = {
      ...     "peft_type": "PREFIX_TUNING",
      ...     "task_type": "CAUSAL_LM",
      ...     "inference_mode": False,
      ...     "num_virtual_tokens": 20,
      ...     "token_dim": 1280,
      ...     "num_transformer_submodules": 1,
      ...     "num_attention_heads": 20,
      ...     "num_layers": 36,
      ...     "encoder_hidden_size": 1280,
      ...     "prefix_projection": False,
      ...     "postprocess_past_key_value_function": None,
      ... }

      >>> peft_config = get_peft_config(config)
      >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
      >>> peft_model = PeftModelForCausalLM(model, peft_config)
      >>> peft_model.print_trainable_parameters()
      trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
      ```
  """

  def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
    super().__init__(model, peft_config, adapter_name)
    self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      task_ids=None,
      **kwargs,
  ):
    peft_config = self.active_peft_config
    if not peft_config.is_prompt_learning:
      if self.base_model.config.model_type == "mpt":
        if inputs_embeds is not None:
          raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

      if peft_config.peft_type == PeftType.POLY:
        kwargs["task_ids"] = task_ids

      with self._enable_peft_forward_hooks(**kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    batch_size = _get_batch_size(input_ids, inputs_embeds)
    if attention_mask is not None:
      # concat prompt attention mask
      prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
      attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    if kwargs.get("position_ids", None) is not None:
      warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
      kwargs["position_ids"] = None
    if kwargs.get("token_type_ids", None) is not None:
      warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
      kwargs["token_type_ids"] = None
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )

    if peft_config.peft_type == PeftType.PREFIX_TUNING:
      past_key_values = self.get_prompt(batch_size)
      return self.base_model(
          input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
      )
    else:
      if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
      # concat prompt labels
      if labels is not None:
        prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
        kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
      prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
      prompts = prompts.to(inputs_embeds.dtype)
      inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
      return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

  def generate(self, *args, **kwargs):
    peft_config = self.active_peft_config
    self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
    if hasattr(self.base_model, "model"):
      self.base_model.model.generation_config = self.generation_config
    else:
      self.base_model.generation_config = self.generation_config
    try:
      if not peft_config.is_prompt_learning:
        with self._enable_peft_forward_hooks(*args, **kwargs):
          kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
          outputs = self.base_model.generate(*args, **kwargs)
      else:
        outputs = self.base_model.generate(**kwargs)
    except:
      self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
      raise
    else:
      self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
      return outputs

  def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
    peft_config = self.active_peft_config
    model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

    # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
    # for some architectures which requires a special fix for prompt tuning etc.
    # TODO: starting with transformers 4.38, all architectures should support caching.
    uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
    uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
    transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
    uses_cache = uses_transformers_4_38 or (
        uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
    )

    if peft_config.peft_type == PeftType.POLY:
      model_kwargs["task_ids"] = task_ids
    if peft_config.is_prompt_learning:
      if uses_cache and (model_kwargs["past_key_values"] is not None):
        # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
        # In prompt learning methods, past key values are longer when compared to the `input_ids`.
        # As such only consider the last input ids in the autogressive generation phase.
        if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
          model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

      if model_kwargs.get("attention_mask", None) is not None:
        size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
        prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
        model_kwargs["attention_mask"] = torch.cat(
            (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
        )

      if model_kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        model_kwargs["position_ids"] = None

      if kwargs.get("token_type_ids", None) is not None:
        warnings.warn(
            "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
        )
        kwargs["token_type_ids"] = None

      if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
        past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
        model_kwargs["past_key_values"] = past_key_values
      else:
        if model_kwargs["past_key_values"] is None:
          inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
          prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
          prompts = prompts.to(inputs_embeds.dtype)
          model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
          model_kwargs["input_ids"] = None

    # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
    # passed in the forward pass to keep track of the position ids of the cache. We have to
    # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
    # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
    _ = model_kwargs.pop("cache_position", None)

    return model_kwargs


MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, PeftModel] = {
    "CAUSAL_LM": PeftModelForCausalLMExp,
}


def get_peft_model_exp(
    model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", mixed: bool = False
) -> PeftModel:
  """
  Returns a Peft model object from a model and a config.

  Args:
      model ([`transformers.PreTrainedModel`]):
          Model to be wrapped.
      peft_config ([`PeftConfig`]):
          Configuration object containing the parameters of the Peft model.
      adapter_name (`str`, `optional`, defaults to `"default"`):
          The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
      mixed (`bool`, `optional`, defaults to `False`):
          Whether to allow mixing different (compatible) adapter types.
  """
  model_config = getattr(model, "config", {"model_type": "custom"})
  if hasattr(model_config, "to_dict"):
    model_config = model_config.to_dict()

  peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

  # ===== EDIT START =====
  assert not mixed
  assert peft_config.task_type in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys()
  # ===== EDIT END =====

  if peft_config.is_prompt_learning:
    peft_config = _prepare_prompt_learning_config(peft_config, model_config)
  return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
