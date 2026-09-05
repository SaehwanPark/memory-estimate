"""
Memory and Unified RAM calculator for LLMs in agentic coding workflows.
Indent: 2 spaces.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from model_fetcher import ModelArchitecture, QuantizationInfo


class KVCachePrecision(Enum):
  FP16 = "FP16 (16-bit / 2.0 bytes)"
  FP8 = "FP8 / Q8_0 (8-bit / 1.0 byte - Recommended)"
  Q4_0 = "Q4_0 (4-bit / 0.5 bytes)"
  Q5_0 = "Q5_0 (~5-bit / 0.625 bytes)"

  @property
  def bytes_per_element(self) -> float:
    if self == KVCachePrecision.FP16:
      return 2.0
    elif self == KVCachePrecision.FP8:
      return 1.0
    elif self == KVCachePrecision.Q4_0:
      return 0.5
    elif self == KVCachePrecision.Q5_0:
      return 0.625
    return 2.0

  @property
  def short_name(self) -> str:
    if self == KVCachePrecision.FP16:
      return "FP16"
    elif self == KVCachePrecision.FP8:
      return "FP8"
    elif self == KVCachePrecision.Q4_0:
      return "Q4_0"
    elif self == KVCachePrecision.Q5_0:
      return "Q5_0"
    return "FP16"


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
  
  # Context metrics
  context_window: int
  kv_bytes_per_token: float
  kv_cache_precision: str
  
  # Flags & Architectural notes
  is_mla: bool
  is_hybrid_linear: bool
  is_moe: bool
  insights: List[str]


@dataclass
class MacHardwareEvaluation:
  tier_name: str
  ram_gb: int
  status: str  # "OPTIMAL", "TIGHT", "WILL_SWAP_OR_OOM"
  badge_color: str  # "green", "yellow", "red"
  free_ram_gb: float
  utilization_pct: float
  notes: str


def calculate_kv_cache_bytes(
  arch: ModelArchitecture,
  context_tokens: int,
  precision: KVCachePrecision = KVCachePrecision.FP8,
) -> Tuple[int, float]:
  """
  Calculates KV cache memory in bytes and bytes-per-token.
  Handles:
  1. Standard MHA / GQA (2 * layers * kv_heads * head_dim * context * bytes)
  2. Multi-Head Latent Attention / MLA (layers * (kv_lora_rank + qk_rope_head_dim) * context * bytes)
  3. Hybrid Linear Attention (e.g. GLM-5.3-Flash, applies MLA only to full attention layers,
    plus constant recurrent state for linear attention layers)
  """

  bpe = precision.bytes_per_element
  num_layers = arch.num_hidden_layers

  # Multi-Head Latent Attention (DeepSeek V2/V3 and GLM-5)
  if arch.is_mla and arch.kv_lora_rank:
    latent_dim = arch.kv_lora_rank + (arch.qk_rope_head_dim or 0)
    
    if arch.is_hybrid_linear and arch.num_full_attention_layers is not None:
      # Only full/sparse attention layers maintain dynamic KV cache
      attn_layers = arch.num_full_attention_layers
      linear_layers = arch.num_linear_attention_layers or (num_layers - attn_layers)
      
      # MLA layers dynamic context cache
      mla_bytes_per_token = attn_layers * latent_dim * bpe
      mla_total_bytes = int(mla_bytes_per_token * context_tokens)
      
      # Fixed recurrent state for linear attention layers (constant O(1) memory)
      # e.g. state size ~ heads * head_dim * conv_kernel * 4 bytes per layer
      linear_state_per_layer = 64 * 128 * 4 * 4  # ~128 KB
      linear_total_bytes = int(linear_layers * linear_state_per_layer)
      
      total_bytes = mla_total_bytes + linear_total_bytes
      bytes_per_token = total_bytes / max(1, context_tokens)
      return total_bytes, bytes_per_token
    else:
      # Standard full MLA across all layers
      bytes_per_token = num_layers * latent_dim * bpe
      total_bytes = int(bytes_per_token * context_tokens)
      return total_bytes, bytes_per_token

  # Standard Grouped-Query Attention (GQA) or Multi-Head Attention (MHA)
  kv_heads = arch.num_key_value_heads if arch.num_key_value_heads > 0 else arch.num_attention_heads
  head_dim = arch.head_dim if arch.head_dim > 0 else (arch.hidden_size // arch.num_attention_heads)

  # Check sliding window
  if arch.sliding_window and arch.sliding_window < context_tokens:
    # If model uses sliding window on all layers (e.g. Mistral 7B standard)
    effective_ctx = arch.sliding_window
  else:
    effective_ctx = context_tokens

  # 2 for Key and Value
  bytes_per_token = 2 * num_layers * kv_heads * head_dim * bpe
  total_bytes = int(bytes_per_token * effective_ctx)
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
  # Scales with context length: attention workspace & position index buffers
  ctx_buffer_bytes = int((context_tokens * arch.num_attention_heads * 2 * 16))
  
  # Forward pass prefill chunk activation buffer
  chunk_activation_bytes = int(chunk * hidden_size * 4 * 6)
  
  # MoE router buffer if applicable
  moe_overhead = 0
  if arch.is_moe and arch.num_routed_experts:
    moe_overhead = int(chunk * arch.num_routed_experts * 4 * 2)

  # llama.cpp Metal context scratch buffer minimum is typically ~800MB, scaling up with context
  total_scratch = max(800 * 1024 * 1024, ctx_buffer_bytes + chunk_activation_bytes + moe_overhead)
  
  # Long context scaling clamp
  # For 128k context, scratchpad is typically ~2.5 - 3.5 GB
  if context_tokens >= 65536:
    total_scratch += int((context_tokens / 65536) * 1024 * 1024 * 1024)

  return total_scratch


def calculate_unified_ram(
  arch: ModelArchitecture,
  selected_quant: QuantizationInfo,
  context_tokens: int,
  kv_precision: KVCachePrecision = KVCachePrecision.FP8,
  headroom_gb: float = 6.0,
  ubatch_chunk_size: int = 1024,
) -> MemoryBreakdown:
  """
  Calculates complete Unified RAM requirements for agentic coding.
  Total RAM = Weights + KV Cache + Activation Scratch + Headroom.
  """
  # 1. Weights Memory
  weights_bytes = selected_quant.total_size_bytes
  weights_gb = weights_bytes / (1024**3)

  # 2. KV Cache Memory
  kv_bytes, kv_per_token = calculate_kv_cache_bytes(
    arch=arch,
    context_tokens=context_tokens,
    precision=kv_precision,
  )
  kv_gb = kv_bytes / (1024**3)

  # 3. Activation & Scratch Memory
  scratch_bytes = calculate_activation_scratch_bytes(
    arch=arch,
    context_tokens=context_tokens,
    ubatch_chunk_size=ubatch_chunk_size,
  )
  scratch_gb = scratch_bytes / (1024**3)

  # 4. Required Headroom (macOS + display + IDE + browser)
  headroom_bytes = int(headroom_gb * (1024**3))

  # 5. Total Unified RAM
  total_bytes = weights_bytes + kv_bytes + scratch_bytes + headroom_bytes
  total_gb = total_bytes / (1024**3)

  # Generate agentic insights
  insights = []
  if arch.is_mla:
    insights.append(
      f"💎 MLA Efficiency: At {context_tokens:,} tokens, KV cache is only {kv_gb:.2f} GB "
      f"({kv_per_token:.1f} bytes/token). Standard GQA would require ~{kv_gb * 4.5:.1f} GB!"
    )
  if arch.is_hybrid_linear:
    insights.append(
      f"⚡ Hybrid Linear: Linear attention layers maintain constant state; only {arch.num_full_attention_layers} "
      f"layers expand with context depth."
    )
  if context_tokens >= 65536:
    insights.append(
      f"🧠 Deep Agentic Context ({context_tokens // 1024}k): Ideal for multi-turn repo navigation, "
      f"large git diffs, and iterative tool calls."
    )
  if kv_precision == KVCachePrecision.FP16 and kv_gb > 8.0:
    fp8_saved_gb = kv_gb * 0.5
    insights.append(
      f"💡 Recommendation: Switching KV Cache from FP16 to FP8 (Q8_0) will save ~{fp8_saved_gb:.1f} GB of RAM "
      f"with near-zero coding benchmark degradation."
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
    context_window=context_tokens,
    kv_bytes_per_token=round(kv_per_token, 1),
    kv_cache_precision=kv_precision.short_name,
    is_mla=arch.is_mla,
    is_hybrid_linear=arch.is_hybrid_linear,
    is_moe=arch.is_moe,
    insights=insights,
  )


def evaluate_all_mac_tiers(total_ram_needed_gb: float) -> List[MacHardwareEvaluation]:
  """
  Evaluates standard Apple Silicon Unified RAM configurations against the needed memory.
  Considers macOS default ~75% single-process allocation threshold and sysctl adjustments.
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
  for name, ram in standard_tiers:
    free_ram = ram - total_ram_needed_gb
    utilization = (total_ram_needed_gb / ram) * 100.0

    if total_ram_needed_gb <= ram * 0.82:
      status = "OPTIMAL"
      badge_color = "green"
      notes = f"Comfortably fits with {free_ram:.1f} GB remaining. Zero memory pressure or swapping."
    elif total_ram_needed_gb <= ram:
      status = "TIGHT"
      badge_color = "yellow"
      notes = (
        f"Fits with only {free_ram:.1f} GB remaining. Recommended to close heavy background apps. "
        "May require 'sudo sysctl iogpu.wired_mem_limit' tweak."
      )
    else:
      deficit = total_ram_needed_gb - ram
      status = "WILL_SWAP_OR_OOM"
      badge_color = "red"
      notes = (
        f"Insufficient unified RAM (short by {deficit:.1f} GB). "
        "Will cause severe SSD swap thrashing (10-50x speed drop) or crash with Metal allocation failure."
      )

    evaluations.append(
      MacHardwareEvaluation(
        tier_name=name,
        ram_gb=ram,
        status=status,
        badge_color=badge_color,
        free_ram_gb=round(free_ram, 1),
        utilization_pct=round(utilization, 1),
        notes=notes,
      )
    )

  return evaluations
