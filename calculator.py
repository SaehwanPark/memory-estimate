"""
Memory and Unified RAM calculator for LLMs in agentic coding workflows.
Indent: 2 spaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from model_fetcher import ModelArchitecture, QuantizationInfo


class KVCacheType(Enum):
  F16 = "f16"
  BF16 = "bf16"
  FP8_E4M3 = "fp8_e4m3"
  FP8_E5M2 = "fp8_e5m2"
  Q8_0 = "q8_0"
  Q5_0 = "q5_0"
  Q4_0 = "q4_0"

  @property
  def bytes_per_element(self) -> float:
    # GGML block quantization layouts:
    # F16 / BF16: 2 bytes
    # FP8 (e4m3/e5m2): 1 byte (unblocked float)
    # Q8_0: block of 32 has 2-byte scale + 32 int8 = 34 bytes -> 34/32 = 1.0625 bytes
    # Q5_0: block of 32 has 2-byte scale + 4-byte high + 16-byte low = 22 bytes -> 22/32 = 0.6875 bytes
    # Q4_0: block of 32 has 2-byte scale + 16-byte nibbles = 18 bytes -> 18/32 = 0.5625 bytes
    if self in (KVCacheType.F16, KVCacheType.BF16):
      return 2.0
    elif self in (KVCacheType.FP8_E4M3, KVCacheType.FP8_E5M2):
      return 1.0
    elif self == KVCacheType.Q8_0:
      return 1.0625
    elif self == KVCacheType.Q5_0:
      return 0.6875
    elif self == KVCacheType.Q4_0:
      return 0.5625
    return 2.0

  @property
  def label(self) -> str:
    if self == KVCacheType.F16:
      return "f16 (16-bit float / 2.00 B)"
    elif self == KVCacheType.BF16:
      return "bf16 (16-bit bfloat / 2.00 B)"
    elif self == KVCacheType.FP8_E4M3:
      return "fp8_e4m3 (8-bit float / 1.00 B)"
    elif self == KVCacheType.FP8_E5M2:
      return "fp8_e5m2 (8-bit float / 1.00 B)"
    elif self == KVCacheType.Q8_0:
      return "q8_0 (8.5 bpw / 1.06 B - Recommended)"
    elif self == KVCacheType.Q5_0:
      return "q5_0 (5.5 bpw / 0.69 B)"
    elif self == KVCacheType.Q4_0:
      return "q4_0 (4.5 bpw / 0.56 B)"
    return self.value


class KVCachePrecision(Enum):
  FP16 = "FP16 (16-bit / 2.0 bytes)"
  FP8 = "FP8 (8-bit float / 1.0 byte)"
  Q8_0 = "Q8_0 (8.5 bpw / 1.0625 bytes - Recommended)"
  Q4_0 = "Q4_0 (4.5 bpw / 0.5625 bytes)"
  Q5_0 = "Q5_0 (5.5 bpw / 0.6875 bytes)"

  @property
  def bytes_per_element(self) -> float:
    if self == KVCachePrecision.FP16:
      return 2.0
    elif self == KVCachePrecision.FP8:
      return 1.0
    elif self == KVCachePrecision.Q8_0:
      return 1.0625
    elif self == KVCachePrecision.Q4_0:
      return 0.5625
    elif self == KVCachePrecision.Q5_0:
      return 0.6875
    return 2.0

  @property
  def short_name(self) -> str:
    if self == KVCachePrecision.FP16:
      return "FP16"
    elif self == KVCachePrecision.FP8:
      return "FP8"
    elif self == KVCachePrecision.Q8_0:
      return "Q8_0"
    elif self == KVCachePrecision.Q4_0:
      return "Q4_0"
    elif self == KVCachePrecision.Q5_0:
      return "Q5_0"
    return "FP16"

  def to_cache_type(self) -> KVCacheType:
    if self == KVCachePrecision.FP16:
      return KVCacheType.F16
    elif self == KVCachePrecision.FP8:
      return KVCacheType.FP8_E4M3
    elif self == KVCachePrecision.Q8_0:
      return KVCacheType.Q8_0
    elif self == KVCachePrecision.Q4_0:
      return KVCacheType.Q4_0
    elif self == KVCachePrecision.Q5_0:
      return KVCacheType.Q5_0
    return KVCacheType.F16


@dataclass
class MemoryBreakdown:
  weights_bytes: int
  weights_gb: float
  kv_cache_bytes: int
  kv_cache_gb: float
  activation_scratch_bytes: int
  activation_scratch_gb: float
  headroom_bytes: int
  headroom_gb: float
  total_ram_required_bytes: int
  total_ram_required_gb: float
  runtime_process_bytes: int
  runtime_process_gb: float

  # Context metrics
  context_window: int
  kv_bytes_per_token: float
  kv_cache_precision: str
  cache_type_k: str
  cache_type_v: str

  # Flags & Architectural notes
  is_mla: bool
  is_hybrid_linear: bool
  is_moe: bool
  insights: List[str]
  component_confidences: Dict[str, str] = field(default_factory=dict)


@dataclass
class MacHardwareEvaluation:
  tier_name: str
  ram_gb: int
  status: str  # "OPTIMAL", "TIGHT", "LIKELY_SWAP_OR_OOM"
  badge_color: str  # "green", "yellow", "red"
  free_ram_gb: float
  utilization_pct: float
  os_ram_status: str  # "COMFORTABLE", "TIGHT", "EXCEEDED"
  metal_status: str  # "WITHIN_DEFAULT_LIMIT", "EXCEEDS_DEFAULT_LIMIT"
  default_metal_limit_gb: float
  notes: str


def calculate_kv_cache_bytes(
  arch: ModelArchitecture,
  context_tokens: int,
  precision: Optional[KVCachePrecision] = None,
  cache_type_k: Optional[KVCacheType] = None,
  cache_type_v: Optional[KVCacheType] = None,
) -> Tuple[int, float]:
  """
  Calculates KV cache memory in bytes and bytes-per-token with compositional attention modeling.

  Supports:
  1. Standard GQA / MHA: full attention layers * kv_heads * head_dim * (k_bpe + v_bpe)
  2. Multi-Head Latent Attention (MLA): full attention layers * latent_dim * avg(k_bpe, v_bpe)
  3. Hybrid Linear Attention: dynamic context cache on full-attention layers only,
     plus constant recurrent state for linear-attention layers (e.g. Qwen3.x, GLM-5.3-Flash).
  4. Separate K and V cache quantization types (e.g. --cache-type-k q8_0 --cache-type-v q4_0).
  5. Precise GGML block quantization overhead (e.g. Q8_0=1.0625, Q4_0=0.5625, Q5_0=0.6875 bytes).
  """
  # Determine K and V bytes per element
  if cache_type_k is not None:
    k_bpe = cache_type_k.bytes_per_element
  elif precision is not None:
    k_bpe = precision.bytes_per_element
  else:
    k_bpe = KVCacheType.Q8_0.bytes_per_element

  if cache_type_v is not None:
    v_bpe = cache_type_v.bytes_per_element
  elif precision is not None:
    v_bpe = precision.bytes_per_element
  else:
    v_bpe = KVCacheType.Q8_0.bytes_per_element

  num_layers = arch.num_hidden_layers

  # Compositional layer distribution: full attention vs linear attention
  if arch.is_hybrid_linear:
    full_layers = (
      arch.num_full_attention_layers
      if arch.num_full_attention_layers is not None
      else num_layers
    )
    linear_layers = (
      arch.num_linear_attention_layers
      if arch.num_linear_attention_layers is not None
      else max(0, num_layers - full_layers)
    )
  else:
    full_layers = num_layers
    linear_layers = 0

  # Context length for full attention layers (sliding window handling)
  if arch.sliding_window and arch.sliding_window < context_tokens:
    effective_ctx = arch.sliding_window
  else:
    effective_ctx = context_tokens

  # 1. Dynamic context cache for full attention layers
  if arch.is_mla and arch.kv_lora_rank:
    # Multi-Head Latent Attention (DeepSeek V2/V3, GLM-5 MLA layers)
    latent_dim = arch.kv_lora_rank + (arch.qk_rope_head_dim or 0)
    avg_bpe = (k_bpe + v_bpe) / 2.0
    full_attn_bpt = full_layers * latent_dim * avg_bpe
    full_attn_total_bytes = int(full_attn_bpt * effective_ctx)
  else:
    # Standard Grouped-Query Attention (GQA) or Multi-Head Attention (MHA)
    kv_heads = arch.num_key_value_heads if arch.num_key_value_heads > 0 else arch.num_attention_heads
    head_dim = arch.head_dim if arch.head_dim > 0 else (arch.hidden_size // arch.num_attention_heads)
    full_attn_bpt = full_layers * kv_heads * head_dim * (k_bpe + v_bpe)
    full_attn_total_bytes = int(full_attn_bpt * effective_ctx)

  # 2. Fixed recurrent state for linear attention layers (constant O(1) memory per layer)
  # Standard state size ~ heads * head_dim * conv_kernel * 4 bytes per layer (~128 KB)
  linear_state_per_layer = getattr(arch, "linear_state_bytes_per_layer", None) or (64 * 128 * 4 * 4)
  linear_total_bytes = int(linear_layers * linear_state_per_layer)

  total_bytes = full_attn_total_bytes + linear_total_bytes
  bytes_per_token = full_attn_bpt + (linear_total_bytes / max(1, context_tokens))
  return total_bytes, bytes_per_token


def calculate_activation_scratch_bytes(
  arch: ModelArchitecture,
  context_tokens: int,
  ubatch_chunk_size: int = 1024,
) -> int:
  """
  Calculates intermediate activation and scratchpad memory used during prompt prefill and decoding.
  In agentic coding, prefill chunks (512 to 2048 tokens) inject large codebases/tools,
  requiring substantial activation scratch buffers.
  """
  hidden_size = arch.hidden_size
  chunk = min(ubatch_chunk_size, context_tokens)

  # Base runtime & context graph overhead in Metal / llama.cpp / vLLM
  ctx_buffer_bytes = int((context_tokens * arch.num_attention_heads * 2 * 16))

  # Forward pass prefill chunk activation buffer
  chunk_activation_bytes = int(chunk * hidden_size * 4 * 6)

  # MoE router buffer if applicable
  moe_overhead = 0
  if arch.is_moe and arch.num_routed_experts:
    moe_overhead = int(chunk * arch.num_routed_experts * 4 * 2)

  # llama.cpp Metal context scratch buffer minimum is typically ~800MB, scaling up with context
  total_scratch = max(800 * 1024 * 1024, ctx_buffer_bytes + chunk_activation_bytes + moe_overhead)

  # Long context scaling clamp: for 128k context, scratchpad is typically ~2.5 - 3.5 GB
  if context_tokens >= 65536:
    total_scratch += int((context_tokens / 65536) * 1024 * 1024 * 1024)

  return total_scratch


def calculate_unified_ram(
  arch: ModelArchitecture,
  selected_quant: QuantizationInfo,
  context_tokens: int,
  precision: Optional[KVCachePrecision] = None,
  cache_type_k: Optional[KVCacheType] = None,
  cache_type_v: Optional[KVCacheType] = None,
  headroom_gb: float = 6.0,
  ubatch_chunk_size: int = 1024,
  kv_precision: Optional[KVCachePrecision] = None,
) -> MemoryBreakdown:
  """
  Calculates Unified RAM requirements for local LLM inference.
  Total RAM = Weights + KV Cache + Activation Scratch + Headroom.
  """
  # Resolve precision arguments for backwards compatibility
  if kv_precision is not None and precision is None:
    precision = kv_precision

  if cache_type_k is None:
    if precision is not None:
      cache_type_k = precision.to_cache_type()
    else:
      cache_type_k = KVCacheType.Q8_0

  if cache_type_v is None:
    if precision is not None:
      cache_type_v = precision.to_cache_type()
    else:
      cache_type_v = KVCacheType.Q8_0

  # 1. Weights Memory (Exact repository bytes or parameter heuristic)
  weights_bytes = selected_quant.total_size_bytes
  weights_gb = weights_bytes / (1024**3)

  # 2. KV Cache Memory (Architecture-derived)
  kv_bytes, kv_per_token = calculate_kv_cache_bytes(
    arch=arch,
    context_tokens=context_tokens,
    cache_type_k=cache_type_k,
    cache_type_v=cache_type_v,
  )
  kv_gb = kv_bytes / (1024**3)

  # 3. Activation & Scratch Memory (Backend heuristic)
  scratch_bytes = calculate_activation_scratch_bytes(
    arch=arch,
    context_tokens=context_tokens,
    ubatch_chunk_size=ubatch_chunk_size,
  )
  scratch_gb = scratch_bytes / (1024**3)

  # 4. Required Headroom (User policy)
  headroom_bytes = int(headroom_gb * (1024**3))

  # 5. Runtime Process Memory (Metal allocation working set)
  runtime_bytes = weights_bytes + kv_bytes + scratch_bytes
  runtime_gb = runtime_bytes / (1024**3)

  # 6. Total Unified RAM
  total_bytes = runtime_bytes + headroom_bytes
  total_gb = total_bytes / (1024**3)

  # Epistemic confidence breakdown
  weights_conf = (
    "Estimated parameter heuristic" if selected_quant.is_estimated else "Exact repository bytes"
  )
  component_confidences = {
    "weights": weights_conf,
    "kv_cache": "Architecture-derived",
    "activation_scratch": "Backend heuristic",
    "headroom": "User policy",
  }

  # Generate insights
  insights = []
  if arch.is_mla:
    insights.append(
      f"💎 MLA Efficiency: At {context_tokens:,} tokens, KV cache is {kv_gb:.2f} GB "
      f"({kv_per_token:.1f} bytes/token) using compressed latent vector storage."
    )
  if arch.is_hybrid_linear:
    insights.append(
      f"⚡ Hybrid Linear Attention: Fixed recurrent states on {arch.num_linear_attention_layers} "
      f"linear layers; only {arch.num_full_attention_layers} full-attention layers expand with context depth."
    )
  if context_tokens >= 65536:
    insights.append(
      f"🧠 Deep Context ({context_tokens // 1024}k): Sized for multi-turn agent loops, "
      f"large repositories, and extended tool interactions."
    )

  if (cache_type_k in (KVCacheType.F16, KVCacheType.BF16) or cache_type_v in (KVCacheType.F16, KVCacheType.BF16)) and kv_gb > 8.0:
    insights.append(
      f"💡 Recommendation: Switching KV Cache to q8_0 reduces cache memory by ~47%. "
      "Recommended memory-efficient default; quality impact is generally small but model/backend dependent."
    )

  precision_label = (
    f"K: {cache_type_k.value} / V: {cache_type_v.value}"
    if cache_type_k != cache_type_v
    else cache_type_k.value
  )

  return MemoryBreakdown(
    weights_bytes=weights_bytes,
    weights_gb=round(weights_gb, 2),
    kv_cache_bytes=kv_bytes,
    kv_cache_gb=round(kv_gb, 2),
    activation_scratch_bytes=scratch_bytes,
    activation_scratch_gb=round(scratch_gb, 2),
    headroom_bytes=headroom_bytes,
    headroom_gb=round(headroom_gb, 2),
    total_ram_required_bytes=total_bytes,
    total_ram_required_gb=round(total_gb, 2),
    runtime_process_bytes=runtime_bytes,
    runtime_process_gb=round(runtime_gb, 2),
    context_window=context_tokens,
    kv_bytes_per_token=round(kv_per_token, 1),
    kv_cache_precision=precision_label,
    cache_type_k=cache_type_k.value,
    cache_type_v=cache_type_v.value,
    is_mla=arch.is_mla,
    is_hybrid_linear=arch.is_hybrid_linear,
    is_moe=arch.is_moe,
    insights=insights,
    component_confidences=component_confidences,
  )


def evaluate_all_mac_tiers(
  total_ram_needed_gb: float,
  runtime_process_gb: Optional[float] = None,
) -> List[MacHardwareEvaluation]:
  """
  Evaluates Apple Silicon Unified RAM configurations against required memory.
  Separates physical OS RAM fit from the macOS default ~75% Metal working-set limit.
  """
  standard_tiers = [
    ("Mac (16 GB Unified)", 16),
    ("Mac (24 GB Unified)", 24),
    ("Mac (32 GB Unified)", 32),
    ("Mac (36 GB Unified)", 36),
    ("Mac (48 GB Unified)", 48),
    ("Mac (64 GB Unified)", 64),
    ("Mac (96 GB Unified)", 96),
    ("Mac (128 GB Unified)", 128),
    ("Mac (192 GB Unified)", 192),
    ("Mac (256 GB Unified)", 256),
    ("Mac (512 GB Unified)", 512),
  ]

  evaluations = []
  runtime_gb = (
    runtime_process_gb
    if runtime_process_gb is not None
    else max(0.0, total_ram_needed_gb - 6.0)
  )

  for name, ram in standard_tiers:
    free_ram = ram - total_ram_needed_gb
    utilization = (total_ram_needed_gb / ram) * 100.0
    default_metal_limit = ram * 0.75

    # Check OS RAM fit and Metal working-set fit independently
    exceeds_os_ram = total_ram_needed_gb > ram
    exceeds_metal_limit = runtime_gb > default_metal_limit

    if exceeds_os_ram:
      status = "LIKELY_SWAP_OR_OOM"
      badge_color = "red"
      os_ram_status = "EXCEEDED"
      metal_status = "EXCEEDS_DEFAULT_LIMIT" if exceeds_metal_limit else "WITHIN_DEFAULT_LIMIT"
      deficit = total_ram_needed_gb - ram
      notes = (
        f"Exceeds physical RAM by {deficit:.1f} GB. "
        "Likely memory pressure, severe SSD swap thrashing, or process termination."
      )
    elif exceeds_metal_limit:
      status = "TIGHT"
      badge_color = "yellow"
      os_ram_status = "TIGHT" if total_ram_needed_gb > ram * 0.85 else "COMFORTABLE"
      metal_status = "EXCEEDS_DEFAULT_LIMIT"
      notes = (
        f"Fits in physical RAM ({free_ram:.1f} GB free), but runtime footprint ({runtime_gb:.1f} GB) "
        f"exceeds default ~75% Metal working-set limit ({default_metal_limit:.1f} GB). "
        "Requires 'sudo sysctl iogpu.wired_mem_limit' tweak to allocate safely without Metal failure."
      )
    elif total_ram_needed_gb > ram * 0.85:
      status = "TIGHT"
      badge_color = "yellow"
      os_ram_status = "TIGHT"
      metal_status = "WITHIN_DEFAULT_LIMIT"
      notes = (
        f"Fits within default Metal limit, but total RAM utilization is high ({utilization:.0f}% with "
        f"{free_ram:.1f} GB free). Recommended to close heavy background applications."
      )
    else:
      status = "OPTIMAL"
      badge_color = "green"
      os_ram_status = "COMFORTABLE"
      metal_status = "WITHIN_DEFAULT_LIMIT"
      notes = (
        f"Comfortably fits with {free_ram:.1f} GB remaining. "
        f"Within default Metal working-set limit ({default_metal_limit:.1f} GB)."
      )

    evaluations.append(
      MacHardwareEvaluation(
        tier_name=name,
        ram_gb=ram,
        status=status,
        badge_color=badge_color,
        free_ram_gb=round(free_ram, 1),
        utilization_pct=round(utilization, 1),
        os_ram_status=os_ram_status,
        metal_status=metal_status,
        default_metal_limit_gb=round(default_metal_limit, 1),
        notes=notes,
      )
    )

  return evaluations
