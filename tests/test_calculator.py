"""
Unit tests for calculator.py
Indent: 2 spaces.
"""

import pytest
from calculator import (
  KVCachePrecision,
  KVCacheType,
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


@pytest.fixture
def qwen3_hybrid_arch():
  # Qwen3.6-35B-A3B: 40 total layers (10 full GQA attention layers, 30 linear attention layers)
  # num_key_value_heads = 2, head_dim = 256
  return ModelArchitecture(
    repo_id="unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
    model_name="Qwen3.6-35B-A3B",
    architecture_type="qwen3_5_moe_text",
    is_gguf=False,
    num_hidden_layers=40,
    hidden_size=2048,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=256,
    max_position_embeddings=262144,
    is_moe=True,
    num_routed_experts=256,
    num_experts_per_tok=8,
    is_mla=False,
    is_hybrid_linear=True,
    num_full_attention_layers=10,
    num_linear_attention_layers=30,
    full_attention_kv_scheme="gqa",
  )


def test_standard_gqa_kv_cache(qwen_coder_arch):
  # 64 layers, 8 kv heads, 128 head_dim -> 2 * 64 * 8 * 128 = 131,072 elements per token
  # At FP16 (2 bytes/elem): 262,144 bytes/token
  # At 32,768 tokens: 262,144 * 32,768 = 8,589,934,592 bytes = exactly 8.0 GB!
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=qwen_coder_arch,
    context_tokens=32768,
    cache_type_k=KVCacheType.F16,
    cache_type_v=KVCacheType.F16,
  )
  assert bpt == 262144.0
  assert total_bytes == 8 * 1024**3

  # At FP8 (1 byte/elem): exactly 4.0 GB!
  total_bytes_fp8, bpt_fp8 = calculate_kv_cache_bytes(
    arch=qwen_coder_arch,
    context_tokens=32768,
    cache_type_k=KVCacheType.FP8_E4M3,
    cache_type_v=KVCacheType.FP8_E4M3,
  )
  assert bpt_fp8 == 131072.0
  assert total_bytes_fp8 == 4 * 1024**3


def test_mla_hybrid_kv_cache(glm5_flash_arch):
  # GLM-5.3-Flash has 11 MLA layers with latent rank 512
  # Dynamic elements per token = 11 * 512 = 5,632 elements/token
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=glm5_flash_arch,
    context_tokens=65536,
    cache_type_k=KVCacheType.FP8_E4M3,
    cache_type_v=KVCacheType.FP8_E4M3,
  )
  # At 64k tokens, KV cache is less than 0.5 GB!
  gb = total_bytes / (1024**3)
  assert gb < 0.5
  assert gb > 0.3


def test_non_mla_hybrid_linear_attention_p0(qwen3_hybrid_arch):
  # P0 correctness test:
  # Qwen 3.6 MoE has 10 full attention layers (GQA) and 30 linear attention layers.
  # It must NOT compute standard KV cache across all 40 layers!
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=qwen3_hybrid_arch,
    context_tokens=65536,
    cache_type_k=KVCacheType.F16,
    cache_type_v=KVCacheType.F16,
  )
  # 10 full layers, 2 kv heads, 256 head_dim -> 2 * 10 * 2 * 256 * 2.0 = 20,480 bytes/token for full attention
  # For 65,536 tokens: 20,480 * 65536 = 1,342,177,280 bytes (~1.25 GB)
  # 30 linear layers * 131,072 bytes recurrent state = 3,932,160 bytes (~3.75 MB)
  expected_full_bytes = 10 * 2 * 256 * 4.0 * 65536
  expected_linear_bytes = 30 * 131072
  expected_total = int(expected_full_bytes + expected_linear_bytes)
  assert total_bytes == expected_total

  # If a buggy implementation calculated 40 layers of conventional GQA, it would be ~5.0 GB
  wrong_40_layer_bytes = 40 * 2 * 256 * 4.0 * 65536
  assert total_bytes < wrong_40_layer_bytes / 3


def test_asymmetric_kv_quantization(qwen_coder_arch):
  # Asymmetric: K is Q8_0 (1.0625 bpe), V is Q4_0 (0.5625 bpe)
  # Total per element pair = 1.0625 + 0.5625 = 1.625 bytes
  total_bytes, bpt = calculate_kv_cache_bytes(
    arch=qwen_coder_arch,
    context_tokens=8192,
    cache_type_k=KVCacheType.Q8_0,
    cache_type_v=KVCacheType.Q4_0,
  )
  elements_per_token_half = 64 * 8 * 128
  expected_bpt = elements_per_token_half * (1.0625 + 0.5625)
  assert bpt == expected_bpt
  assert total_bytes == int(expected_bpt * 8192)


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
    cache_type_k=KVCacheType.Q8_0,
    cache_type_v=KVCacheType.Q8_0,
    headroom_gb=6.0,
  )

  assert breakdown.weights_gb == 18.5
  assert breakdown.headroom_gb == 6.0
  assert breakdown.kv_cache_gb > 7.0  # 64k at Q8_0 (1.0625 B) is ~8.5 GB
  assert breakdown.activation_scratch_gb >= 0.8
  # Total should be around 18.5 + 8.5 + ~2.0 + 6.0 = ~35 GB
  assert 33.0 <= breakdown.total_ram_required_gb <= 37.0
  assert "Exact repository bytes" in breakdown.component_confidences["weights"]
  assert breakdown.component_confidences["kv_cache"] == "Architecture-derived"


def test_evaluate_mac_hardware():
  # Test with 35 GB required memory, 29 GB runtime process memory
  evals = evaluate_all_mac_tiers(total_ram_needed_gb=35.0, runtime_process_gb=29.0)
  tier_dict = {e.ram_gb: e for e in evals}

  # 32 GB Mac should LIKELY_SWAP_OR_OOM
  assert tier_dict[32].status == "LIKELY_SWAP_OR_OOM"
  assert tier_dict[32].badge_color == "red"
  assert tier_dict[32].os_ram_status == "EXCEEDED"

  # 36 GB Mac: 36 * 0.75 = 27 GB default Metal limit.
  # 29 GB runtime exceeds 27 GB default limit!
  assert tier_dict[36].status == "TIGHT"
  assert tier_dict[36].badge_color == "yellow"
  assert tier_dict[36].metal_status == "EXCEEDS_DEFAULT_LIMIT"
  assert "sudo sysctl iogpu.wired_mem_limit" in tier_dict[36].notes

  # 64 GB Mac: 64 * 0.75 = 48 GB default Metal limit. Fits comfortably.
  assert tier_dict[64].status == "OPTIMAL"
  assert tier_dict[64].badge_color == "green"
  assert tier_dict[64].metal_status == "WITHIN_DEFAULT_LIMIT"


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
    cache_type_k=KVCacheType.F16,
    cache_type_v=KVCacheType.F16,
  )
  expected_bytes = int(bpt * 4096)
  assert total_bytes == expected_bytes


def test_ggml_block_quant_sizes():
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

  b_f16, _ = calculate_kv_cache_bytes(arch, 8192, cache_type_k=KVCacheType.F16, cache_type_v=KVCacheType.F16)
  b_fp8, _ = calculate_kv_cache_bytes(arch, 8192, cache_type_k=KVCacheType.FP8_E4M3, cache_type_v=KVCacheType.FP8_E4M3)
  b_q8, _ = calculate_kv_cache_bytes(arch, 8192, cache_type_k=KVCacheType.Q8_0, cache_type_v=KVCacheType.Q8_0)
  b_q4, _ = calculate_kv_cache_bytes(arch, 8192, cache_type_k=KVCacheType.Q4_0, cache_type_v=KVCacheType.Q4_0)
  b_q5, _ = calculate_kv_cache_bytes(arch, 8192, cache_type_k=KVCacheType.Q5_0, cache_type_v=KVCacheType.Q5_0)

  # F16 = 2.0 B, FP8 = 1.0 B
  assert b_fp8 == b_f16 / 2
  # Q8_0 has 34 bytes / 32 values = 1.0625 B/val
  assert b_q8 == int(b_f16 * (1.0625 / 2.0))
  # Q4_0 has 18 bytes / 32 values = 0.5625 B/val
  assert b_q4 == int(b_f16 * (0.5625 / 2.0))
  # Q5_0 has 22 bytes / 32 values = 0.6875 B/val
  assert b_q5 == int(b_f16 * (0.6875 / 2.0))
