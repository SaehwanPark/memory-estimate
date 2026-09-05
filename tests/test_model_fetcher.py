"""
Unit tests for model_fetcher.py
Indent: 2 spaces.
"""

import pytest
from model_fetcher import (
  parse_hf_url,
  extract_quant_name_from_filename,
  ModelArchitecture,
  QuantizationInfo,
  ModelMetadataFetcher,
)


def test_parse_hf_url_standard():
  assert parse_hf_url("https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF") == "unsloth/GLM-5.3-Flash-GGUF"
  assert parse_hf_url("https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct") == "Qwen/Qwen2.5-Coder-32B-Instruct"
  assert parse_hf_url("https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main") == "deepseek-ai/DeepSeek-V3"
  assert parse_hf_url("unsloth/GLM-5.3-Flash-GGUF") == "unsloth/GLM-5.3-Flash-GGUF"
  assert parse_hf_url("   'https://huggingface.co/owner/model'  ") == "owner/model"
  assert parse_hf_url("") == ""


def test_extract_quant_name_from_filename():
  # Subdirectory format
  assert extract_quant_name_from_filename("BF16/GLM-5.3-Flash-BF16-00001-of-00014.gguf") == "BF16"
  assert extract_quant_name_from_filename("Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf") == "Q8_0"
  
  # Filename format with sharding
  assert extract_quant_name_from_filename("GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf") == "UD-Q3_K_XL"
  assert extract_quant_name_from_filename("GLM-5.3-Flash-UD-QD3_XXS.gguf") == "UD-QD3_XXS"
  assert extract_quant_name_from_filename("Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
  assert extract_quant_name_from_filename("Meta-Llama-3.1-8B-Instruct-IQ2_XS.gguf") == "IQ2_XS"
  assert extract_quant_name_from_filename("model-Q4_0_4_4.gguf") == "Q4_0_4_4"


def test_model_architecture_defaults():
  arch = ModelArchitecture(
    repo_id="test/model",
    model_name="test-model",
    architecture_type="llama",
    is_gguf=True,
    num_hidden_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=131072,
  )
  assert arch.num_hidden_layers == 32
  assert arch.num_key_value_heads == 8
  assert not arch.is_mla
  assert not arch.is_moe
  assert not arch.is_hybrid_linear


def test_model_metadata_fetcher_quantization_grouping():
  fetcher = ModelMetadataFetcher()
  
  class DummySibling:
    def __init__(self, rfilename, size):
      self.rfilename = rfilename
      self.size = size

  class DummyRepoInfo:
    siblings = [
      DummySibling("BF16/GLM-5.3-Flash-BF16-00001-of-00002.gguf", 10 * 1024**3),
      DummySibling("BF16/GLM-5.3-Flash-BF16-00002-of-00002.gguf", 10 * 1024**3),
      DummySibling("GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00001.gguf", 15 * 1024**3),
      DummySibling("mmproj-BF16.gguf", 1 * 1024**3),  # Should be filtered out
    ]

  quants = fetcher.get_available_quantizations("test/repo", DummyRepoInfo())
  quant_dict = {q.name: q for q in quants}

  assert "BF16" in quant_dict
  assert quant_dict["BF16"].file_count == 2
  assert quant_dict["BF16"].total_size_gb == 20.0

  assert "UD-Q3_K_XL" in quant_dict
  assert quant_dict["UD-Q3_K_XL"].file_count == 1
  assert quant_dict["UD-Q3_K_XL"].total_size_gb == 15.0

  assert "mmproj-BF16" not in quant_dict


def test_base_model_quantization_presets():
  fetcher = ModelMetadataFetcher()
  
  class DummySibling:
    def __init__(self, rfilename, size):
      self.rfilename = rfilename
      self.size = size

  class DummyRepoInfo:
    siblings = [
      DummySibling("model-00001-of-00002.safetensors", 16 * 1024**3),
      DummySibling("model-00002-of-00002.safetensors", 16 * 1024**3),
    ]

  quants = fetcher.get_available_quantizations("test/base-model", DummyRepoInfo())
  names = [q.name for q in quants]
  assert any("Q4_K_M" in n for n in names)
  assert any("FP8" in n for n in names)
  assert any("BF16" in n for n in names)


def test_build_architecture_obj_notes():
  fetcher = ModelMetadataFetcher()
  config = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "kv_lora_rank": 512,
  }
  arch = fetcher._build_architecture_obj(
    repo_id="test/deepseek",
    is_gguf=False,
    base_model=None,
    gguf_meta={},
    config=config,
  )
  assert arch.is_moe is True
  assert arch.num_routed_experts == 64
  assert arch.is_mla is True
  assert len(arch.special_notes) >= 2


def test_detect_prequantized_mlx_model():
  fetcher = ModelMetadataFetcher()

  class DummySibling:
    def __init__(self, rfilename, size):
      self.rfilename = rfilename
      self.size = size

  class DummyRepoInfo:
    tags = ["mlx", "safetensors", "3-bit", "base_model:quantized:Qwen/Qwen3.6-35B-A3B"]
    siblings = [
      DummySibling("model-00001-of-00004.safetensors", int(4.98 * 1024**3)),
      DummySibling("model-00002-of-00004.safetensors", int(4.96 * 1024**3)),
      DummySibling("model-00003-of-00004.safetensors", int(4.93 * 1024**3)),
      DummySibling("model-00004-of-00004.safetensors", int(1.31 * 1024**3)),
    ]

  config = {
    "quantization": {
      "group_size": 64,
      "bits": 3,
      "mode": "affine",
    }
  }

  quants = fetcher.get_available_quantizations(
    "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
    DummyRepoInfo(),
    config=config,
  )

  assert len(quants) >= 1
  exact_quant = quants[0]
  assert "UD-MLX-3bit" in exact_quant.name or "3-bit" in exact_quant.name
  assert "[Repo Exact]" in exact_quant.name
  assert exact_quant.file_count == 4
  assert 16.0 <= exact_quant.total_size_gb <= 16.3
  assert exact_quant.is_estimated is False
  assert exact_quant.bits_per_weight == 3.0


def test_qwen_moe_architecture_parsing():
  fetcher = ModelMetadataFetcher()
  config = {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "full_attention_interval": 4,
  }

  arch = fetcher._build_architecture_obj(
    repo_id="unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
    is_gguf=False,
    base_model="Qwen/Qwen3.6-35B-A3B",
    gguf_meta={},
    config=config,
  )

  assert arch.is_moe is True
  assert arch.num_routed_experts == 256
  assert arch.num_experts_per_tok == 8
  assert arch.is_hybrid_linear is True
  assert arch.num_full_attention_layers == 10
  assert arch.num_linear_attention_layers == 30
  assert arch.full_attention_kv_scheme == "gqa"


def test_gguf_precedence_over_config():
  fetcher = ModelMetadataFetcher()
  # Upstream config says 32 layers and 4096 hidden, but GGUF has 28 layers and 3584 hidden
  config = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
  }
  gguf_meta = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.embedding_length": 3584,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.context_length": 131072,
  }

  arch = fetcher._build_architecture_obj(
    repo_id="test/qwen-gguf",
    is_gguf=True,
    base_model="test/upstream-base",
    gguf_meta=gguf_meta,
    config=config,
  )

  # Artifact metadata takes priority
  assert arch.num_hidden_layers == 28
  assert arch.hidden_size == 3584
  assert arch.num_attention_heads == 28
  assert arch.num_key_value_heads == 4
  assert arch.max_position_embeddings == 131072
  assert arch.full_attention_kv_scheme == "gqa"


def test_inferred_defaults_warning():
  fetcher = ModelMetadataFetcher()
  # Empty config and empty GGUF metadata
  arch = fetcher._build_architecture_obj(
    repo_id="test/unknown-model",
    is_gguf=False,
    base_model=None,
    gguf_meta={},
    config={},
  )

  assert arch.is_inferred_default is True
  assert len(arch.inferred_fields) > 0
  assert any("num_hidden_layers" in f for f in arch.inferred_fields)
  assert any("⚠️ Architecture fields unavailable" in note for note in arch.special_notes)


def test_per_layer_kv_heads_list():
  fetcher = ModelMetadataFetcher()
  # GGUF metadata for hybrid linear model with per-layer head_count_kv list (e.g. GLM-5.3-Flash)
  gguf_meta = {
    "general.architecture": "glm5next",
    "general.name": "GLM 5.3 Flash",
    "glm5next.block_count": 46,
    "glm5next.embedding_length": 4096,
    "glm5next.attention.head_count": 64,
    "glm5next.attention.head_count_kv": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    "glm5next.attention.kv_lora_rank": 512,
    "glm5next.context_length": 1048576,
  }

  arch = fetcher._build_architecture_obj(
    repo_id="unsloth/GLM-5.3-Flash-GGUF",
    is_gguf=True,
    base_model=None,
    gguf_meta=gguf_meta,
    config={},
  )

  assert arch.num_hidden_layers == 46
  assert arch.num_attention_heads == 64
  assert arch.num_key_value_heads == 1
  assert arch.is_hybrid_linear is True
  assert arch.num_full_attention_layers == 12
  assert arch.num_linear_attention_layers == 34
  assert arch.is_mla is True
  assert arch.is_inferred_default is False




