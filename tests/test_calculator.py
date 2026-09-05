"""
Unit tests for calculator.py
Indent: 2 spaces.
"""

import pytest
from calculator import (
  KVCachePrecision,
  calculate_kv_cache_bytes,
  calculate_activation_scratch_bytes,
  calculate_unified_ram,
  evaluate_all_mac_tiers,
)
from model_fetcher import ModelArchitecture, QuantizationInfo


@pytest.fixture
def qwen_coder_arch():
  # Qwen2.5-Coder-32B: 64 layers, 40 heads, 8 kv heads, head_dim 128
  return ModelArchitecture(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    model_name="Qwen2.5-Coder-32B-Instruct",
    architecture_type="qwen2",
    is_gguf=False,
    num_hidden_layers=64,
    hidden_size=5120,
    num_attention_heads=40,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=32768,
  )


@pytest.fixture
def glm5_flash_arch():
  # GLM-5.3-Flash: 46 layers, 11 full MLA attention layers, kv_lora_rank 512, MoE 288 experts
  return ModelArchitecture(
    repo_id="unsloth/GLM-5.3-Flash-GGUF",
    model_name="GLM-5.3-Flash",
    architecture_type="glm5next",
    is_gguf=True,
    num_hidden_layers=46,
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=1,
    head_dim=128,
    max_position_embeddings=1048576,
    is_moe=True,
    num_routed_experts=288,
    num_experts_per_tok=8,
    is_mla=True,
    kv_lora_rank=512,
    qk_rope_head_dim=0,
    is_hybrid_linear=True,
    num_full_attention_layers=11,
    num_linear_attention_layers=35,
  )


def test_standard_gqa_kv_cache(qwen_coder_arch):
  # 64 layers, 8 kv heads, 128 head_dim -> 2 * 64 * 8 * 128 = 131,072 elements per token
  # At FP16 (2 bytes/elem): 262,144 bytes/token
  # At 32,768 tokens: 262,144 * 32,768 = 8,589,934,592 bytes = exactly 8.0 GB!
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=qwen_coder_arch,
    context_tokens=32768,
    precision=KVCachePrecision.FP16,
  )
  assert bpt == 262144.0
  assert total_bytes == 8 * 1024**3

  # At FP8 (1 byte/elem): exactly 4.0 GB!
  total_bytes_fp8, bpt_fp8 = calculate_kv_cache_bytes(
    arch=qwen_coder_arch,
    context_tokens=32768,
    precision=KVCachePrecision.FP8,
  )
  assert bpt_fp8 == 131072.0
  assert total_bytes_fp8 == 4 * 1024**3


def test_mla_hybrid_kv_cache(glm5_flash_arch):
  # GLM-5.3-Flash has 11 MLA layers with latent rank 512
  # Dynamic elements per token = 11 * 512 = 5,632 elements/token
  # At FP8 (1 byte): ~5,632 bytes/token (plus small constant linear state)
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=glm5_flash_arch,
    context_tokens=65536,
    precision=KVCachePrecision.FP8,
  )
  # At 64k tokens, KV cache is less than 0.5 GB!
  gb = total_bytes / (1024**3)
  assert gb < 0.5
  assert gb > 0.3


def test_unified_ram_calculation(qwen_coder_arch):
  quant = QuantizationInfo(
    name="Q4_K_M",
    total_size_bytes=int(18.5 * 1024**3),
    total_size_gb=18.5,
    file_count=1,
    files=["qwen2.5-coder-32b-q4_k_m.gguf"],
  )

  breakdown = calculate_unified_ram(
    arch=qwen_coder_arch,
    selected_quant=quant,
    context_tokens=65536,  # 64k agentic context
    kv_precision=KVCachePrecision.FP8,
    headroom_gb=6.0,
  )

  assert breakdown.weights_gb == 18.5
  assert breakdown.headroom_gb == 6.0
  assert breakdown.kv_cache_gb > 7.0  # 64k at FP8 is 8.0 GB
  assert breakdown.activation_scratch_gb >= 0.8
  # Total should be around 18.5 + 8.0 + ~2.0 + 6.0 = ~34.5 GB
  assert 33.0 <= breakdown.total_ram_required_gb <= 37.0


def test_evaluate_mac_hardware():
  # Test with 35 GB required memory
  evals = evaluate_all_mac_tiers(35.0)
  tier_dict = {e.ram_gb: e for e in evals}

  # 32 GB Mac should WILL_SWAP_OR_OOM
  assert tier_dict[32].status == "WILL_SWAP_OR_OOM"
  assert tier_dict[32].badge_color == "red"

  # 36 GB Mac should be TIGHT
  assert tier_dict[36].status == "TIGHT"
  assert tier_dict[36].badge_color == "yellow"

def test_sliding_window_attention():
  arch = ModelArchitecture(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_name="Mistral-7B",
    architecture_type="mistral",
    is_gguf=False,
    num_hidden_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=32768,
    sliding_window=4096,
  )

  # When context is 32k, effective sliding window caps cache at 4096
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=arch,
    context_tokens=32768,
    precision=KVCachePrecision.FP16,
  )
  expected_bytes = int(bpt * 4096)
  assert total_bytes == expected_bytes


def test_kv_precisions():
  arch = ModelArchitecture(
    repo_id="test/llama",
    model_name="test-llama",
    architecture_type="llama",
    is_gguf=False,
    num_hidden_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
    head_dim=128,
    max_position_embeddings=32768,
  )

  b_fp16, _ = calculate_kv_cache_bytes(arch, 8192, KVCachePrecision.FP16)
  b_fp8, _ = calculate_kv_cache_bytes(arch, 8192, KVCachePrecision.FP8)
  b_q4, _ = calculate_kv_cache_bytes(arch, 8192, KVCachePrecision.Q4_0)

  assert b_fp8 == b_fp16 / 2
  assert b_q4 == b_fp16 / 4

