"""
Hugging Face model metadata and architecture fetcher.
Supports standard Transformers/Safetensors repos and GGUF repositories.
Indent: 2 spaces.
"""

from dataclasses import dataclass, field
import io
import json
import re
import struct
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from huggingface_hub import HfApi


@dataclass
class QuantizationInfo:
  name: str
  total_size_bytes: int
  total_size_gb: float
  file_count: int
  files: List[str]
  is_estimated: bool = False
  bits_per_weight: Optional[float] = None


@dataclass
class ModelArchitecture:
  repo_id: str
  model_name: str
  architecture_type: str  # e.g., "llama", "qwen2", "glm5next", "deepseek_v2", "deepseek_v3"
  is_gguf: bool
  num_hidden_layers: int
  hidden_size: int
  num_attention_heads: int
  num_key_value_heads: int
  head_dim: int
  max_position_embeddings: int
  vocab_size: Optional[int] = None
  
  # MoE considerations
  is_moe: bool = False
  num_routed_experts: Optional[int] = None
  num_experts_per_tok: Optional[int] = None
  num_shared_experts: Optional[int] = None
  total_parameters_est: Optional[float] = None  # in billions
  active_parameters_est: Optional[float] = None  # in billions
  
  # MLA (Multi-Head Latent Attention) considerations
  is_mla: bool = False
  kv_lora_rank: Optional[int] = None
  qk_rope_head_dim: Optional[int] = None
  
  # Hybrid / Linear Attention (e.g. GLM-5.3-Flash)
  is_hybrid_linear: bool = False
  num_full_attention_layers: Optional[int] = None
  num_linear_attention_layers: Optional[int] = None
  
  # Sliding window attention
  sliding_window: Optional[int] = None
  
  # Raw config or extra notes
  base_model: Optional[str] = None
  special_notes: List[str] = field(default_factory=list)


def parse_hf_url(url_or_repo_id: str) -> str:
  """
  Extracts the clean repository ID (e.g., 'unsloth/GLM-5.3-Flash-GGUF')
  from any Hugging Face URL or bare repo ID string.
  """
  clean = url_or_repo_id.strip()
  if not clean:
    return ""

  # Remove leading/trailing quotes or whitespace
  clean = clean.strip("\"'")

  # Handle URLs like https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF
  # or https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/main
  url_pattern = r"(?:https?://)?(?:www\.)?huggingface\.co/([^/\s]+/[^/\s#?]+)"
  m = re.match(url_pattern, clean)
  if m:
    return m.group(1)

  # Check if it's already a repo_id like 'owner/model'
  parts = clean.split("/")
  if len(parts) == 2 and parts[0] and parts[1]:
    return f"{parts[0]}/{parts[1]}"

  # In case of single-word or custom name
  return clean


def extract_quant_name_from_filename(filename: str) -> str:
  """
  Extracts a standardized quantization label from a GGUF filename or relative path.
  e.g. 'BF16/GLM-5.3-Flash-BF16-00001-of-00014.gguf' -> 'BF16'
  e.g. 'GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf' -> 'UD-Q3_K_XL'
  e.g. 'Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf' -> 'Q4_K_M'
  e.g. 'model.UD-QD3_XXS.gguf' -> 'UD-QD3_XXS'
  """
  parts = filename.split("/")
  if len(parts) > 1 and not parts[0].startswith("."):
    # Subdirectory often holds the quantization name (e.g. BF16/...)
    folder = parts[0].strip()
    if folder and folder.lower() != "shard_rewrite":
      return folder.upper()

  base = parts[-1]
  # Strip sharding pattern like -00001-of-00004.gguf
  clean = re.sub(r"-\d+-of-\d+\.gguf$", "", base, flags=re.IGNORECASE)
  clean = re.sub(r"\.gguf$", "", clean, flags=re.IGNORECASE)

  # Look for known quant naming after delimiter (-, ., _)
  m = re.search(
    r"[-._](UD-[A-Za-z0-9_]+|UD_QD[A-Za-z0-9_]+|QD[A-Za-z0-9_]+|IQ[0-9]_[A-Za-z0-9_]+|Q[0-9]_[A-Za-z0-9_]+|Q[0-9]_[0-9](?:_\d+)*|BF16|F16|F32|Q[0-9]_[A-Z])$",
    clean,
    re.IGNORECASE,
  )
  if m:
    return m.group(1).upper()

  # Search anywhere in string
  m2 = re.search(
    r"(UD-[A-Za-z0-9_]+|UD_QD[A-Za-z0-9_]+|QD[A-Za-z0-9_]+|IQ[0-9]_[A-Za-z0-9_]+|Q[0-9]_[A-Za-z0-9_]+|BF16|F16|F32)",
    clean,
    re.IGNORECASE,
  )
  if m2:
    return m2.group(1).upper()

  return clean


def parse_gguf_header_range(url: str, max_bytes: int = 8 * 1024 * 1024) -> Dict[str, Any]:
  """
  Reads GGUF metadata from the beginning of a remote GGUF file using an HTTP Range request.
  This allows extracting exact architecture, context length, heads, and layers without downloading the model.
  """
  headers = {"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "MemoryEstimate/1.0"}
  req = urllib.request.Request(url, headers=headers)
  try:
    with urllib.request.urlopen(req, timeout=10) as resp:
      data = resp.read()
  except Exception:
    return {}

  if len(data) < 24 or data[:4] != b"GGUF":
    return {}

  # Parse GGUF v2/v3 header
  try:
    version = struct.unpack("<I", data[4:8])[0]
    tensor_count, kv_count = struct.unpack("<QQ", data[8:24])
    
    # Simple binary reader for GGUF key-value metadata
    offs = 24
    fields: Dict[str, Any] = {}
    
    for _ in range(kv_count):
      if offs + 8 > len(data):
        break
      key_len = struct.unpack("<Q", data[offs : offs + 8])[0]
      offs += 8
      if offs + key_len > len(data):
        break
      key = data[offs : offs + key_len].decode("utf-8", errors="ignore")
      offs += key_len
      if offs + 4 > len(data):
        break
      val_type = struct.unpack("<I", data[offs : offs + 4])[0]
      offs += 4
      
      val, new_offs = _read_gguf_value(data, offs, val_type)
      if new_offs is None:
        break
      offs = new_offs
      fields[key] = val
      
    return fields
  except Exception:
    return {}


def _read_gguf_value(data: bytes, offs: int, val_type: int) -> Tuple[Any, Optional[int]]:
  """
  Helper to unpack GGUF value types.
  Type IDs:
  0: UINT8, 1: INT8, 2: UINT16, 3: INT16, 4: UINT32, 5: INT32, 6: FLOAT32,
  7: BOOL, 8: STRING, 9: ARRAY, 10: UINT64, 11: INT64, 12: FLOAT64
  """
  try:
    if val_type == 0:  # UINT8
      return data[offs], offs + 1
    elif val_type == 1:  # INT8
      return struct.unpack("<b", data[offs : offs + 1])[0], offs + 1
    elif val_type == 2:  # UINT16
      return struct.unpack("<H", data[offs : offs + 2])[0], offs + 2
    elif val_type == 3:  # INT16
      return struct.unpack("<h", data[offs : offs + 2])[0], offs + 2
    elif val_type == 4:  # UINT32
      return struct.unpack("<I", data[offs : offs + 4])[0], offs + 4
    elif val_type == 5:  # INT32
      return struct.unpack("<i", data[offs : offs + 4])[0], offs + 4
    elif val_type == 6:  # FLOAT32
      return struct.unpack("<f", data[offs : offs + 4])[0], offs + 4
    elif val_type == 7:  # BOOL
      return bool(data[offs]), offs + 1
    elif val_type == 8:  # STRING
      slen = struct.unpack("<Q", data[offs : offs + 8])[0]
      offs += 8
      sval = data[offs : offs + slen].decode("utf-8", errors="ignore")
      return sval, offs + slen
    elif val_type == 9:  # ARRAY
      arr_type = struct.unpack("<I", data[offs : offs + 4])[0]
      offs += 4
      arr_len = struct.unpack("<Q", data[offs : offs + 8])[0]
      offs += 8
      # For large arrays (like tokenizer tokens), skip to save memory/speed
      if arr_type == 8 and arr_len > 1000:
        for _ in range(arr_len):
          if offs + 8 > len(data):
            return None, None
          sl = struct.unpack("<Q", data[offs : offs + 8])[0]
          offs += 8 + sl
        return f"<Array of {arr_len} strings>", offs
      items = []
      for _ in range(arr_len):
        v, next_offs = _read_gguf_value(data, offs, arr_type)
        if next_offs is None:
          return None, None
        offs = next_offs
        items.append(v)
      return items, offs

    elif val_type == 10:  # UINT64
      return struct.unpack("<Q", data[offs : offs + 8])[0], offs + 8
    elif val_type == 11:  # INT64
      return struct.unpack("<q", data[offs : offs + 8])[0], offs + 8
    elif val_type == 12:  # FLOAT64
      return struct.unpack("<d", data[offs : offs + 8])[0], offs + 8
    else:
      return None, None
  except Exception:
    return None, None


def detect_prequantized_metadata(
  repo_id: str,
  repo_info: Any,
  config: Dict[str, Any],
) -> Optional[QuantizationInfo]:
  """
  Detects if the repository contains pre-quantized weights (e.g. MLX, AWQ, GPTQ, EXL2)
  and calculates the exact weight memory from the repository files.
  """
  cfg_quant = config.get("quantization") or config.get("quantization_config")
  tags = getattr(repo_info, "tags", [])
  repo_name = repo_id.split("/")[-1]

  bits: Optional[float] = None
  method: Optional[str] = None
  details: List[str] = []

  if isinstance(cfg_quant, dict):
    b = cfg_quant.get("bits") or cfg_quant.get("bits_per_weight")
    if b is not None:
      bits = float(b)
    m = cfg_quant.get("quant_method") or cfg_quant.get("mode")
    if m:
      method = str(m)
    if "group_size" in cfg_quant:
      details.append(f"g{cfg_quant['group_size']}")

  # Check tags
  for t in tags:
    tl = t.lower()
    if tl in ["mlx", "awq", "gptq", "exl2", "bitsandbytes"]:
      if not method:
        method = tl.upper()
    m_bit = re.match(r"^([0-9.]+)[-_ ]?bit$", tl)
    if m_bit and bits is None:
      bits = float(m_bit.group(1))

  # Check repo name for patterns like UD-MLX-3bit, 3bit, 4bit, AWQ, GPTQ, EXL2
  m_repo = re.search(
    r"(UD[-_]MLX[-_][0-9.]+bit|[0-9.]+bit|AWQ|GPTQ|EXL2)", repo_name, re.IGNORECASE
  )
  quant_label_from_name = m_repo.group(0) if m_repo else None

  if "mlx" in [t.lower() for t in tags] or "mlx" in repo_name.lower():
    if not method:
      method = "MLX"

  is_prequantized = bool(
    cfg_quant
    or method
    or (bits is not None)
    or ("quantized" in tags)
    or any("base_model:quantized" in t for t in tags)
  )

  if not is_prequantized:
    return None

  siblings = getattr(repo_info, "siblings", [])
  weight_files = [
    s for s in siblings
    if s.rfilename.endswith(".safetensors") or s.rfilename.endswith(".bin")
  ]

  if not weight_files:
    return None

  total_bytes = sum(getattr(s, "size", 0) or 0 for s in weight_files)
  total_gb = round(total_bytes / (1024**3), 2)

  # Formulate clean quantization name
  # e.g. "UD-MLX-3bit (3-bit MLX affine g64)"
  name_parts = []
  if quant_label_from_name:
    name_parts.append(quant_label_from_name)
  elif bits is not None:
    bit_str = f"{int(bits) if bits.is_integer() else bits}-bit"
    name_parts.append(f"{method or 'Quant'} {bit_str}")
  else:
    name_parts.append(method or "Pre-Quantized")

  sub_info = []
  if bits is not None and not any(f"{int(bits) if bits.is_integer() else bits}-bit" in p for p in name_parts):
    sub_info.append(f"{int(bits) if bits.is_integer() else bits}-bit")
  if method and method not in " ".join(name_parts):
    sub_info.append(method)
  sub_info.extend(details)

  if sub_info:
    display_name = f"{name_parts[0]} ({', '.join(sub_info)})"
  else:
    display_name = name_parts[0]

  return QuantizationInfo(
    name=f"{display_name} [Repo Exact]",
    total_size_bytes=total_bytes,
    total_size_gb=total_gb,
    file_count=len(weight_files),
    files=[f.rfilename for f in weight_files],
    is_estimated=False,
    bits_per_weight=bits,
  )


class ModelMetadataFetcher:
  def __init__(self, hf_token: Optional[str] = None):
    self.api = HfApi(token=hf_token)

  def fetch_repo_info(self, repo_id: str) -> Any:
    """
    Fetches raw repository info from Hugging Face API including file metadata.
    """
    info = self.api.repo_info(repo_id=repo_id, files_metadata=True)
    return info

  def fetch_config_json(self, repo_id: str) -> Dict[str, Any]:
    """
    Fetches config.json from repository if present.
    """
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
      req = urllib.request.Request(url, headers={"User-Agent": "MemoryEstimate/1.0"})
      with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
      return {}

  def get_available_quantizations(
    self, repo_id: str, repo_info: Any, config: Optional[Dict[str, Any]] = None
  ) -> List[QuantizationInfo]:
    """
    If GGUF repo: groups files by quantization variant and calculates exact file size.
    If pre-quantized repo (MLX, AWQ, GPTQ): extracts exact quantization and file sizes.
    If unquantized Transformers repo: generates standard quantization options based on model parameters.
    """
    siblings = getattr(repo_info, "siblings", [])
    gguf_files = [s for s in siblings if s.rfilename.endswith(".gguf")]

    if gguf_files:
      # GGUF Repository
      groups: Dict[str, List[Any]] = {}
      for s in gguf_files:
        fn = s.rfilename
        # Ignore multimodal projectors for primary LLM weight selection
        if "mmproj" in fn.lower():
          continue
        qname = extract_quant_name_from_filename(fn)
        if qname not in groups:
          groups[qname] = []
        groups[qname].append(s)

      quants: List[QuantizationInfo] = []
      for qname, files in sorted(groups.items()):
        total_bytes = sum(getattr(f, "size", 0) or 0 for f in files)
        quants.append(
          QuantizationInfo(
            name=qname,
            total_size_bytes=total_bytes,
            total_size_gb=round(total_bytes / (1024**3), 2),
            file_count=len(files),
            files=[f.rfilename for f in files],
            is_estimated=False,
          )
        )
      return quants

    # Check if pre-quantized repo (MLX, AWQ, GPTQ, EXL2)
    if config is None:
      config = self.fetch_config_json(repo_id)

    exact_quant = detect_prequantized_metadata(repo_id, repo_info, config)
    if exact_quant:
      quants = [exact_quant]
      
      # Also provide base unquantized reference for comparison
      bits = exact_quant.bits_per_weight or 4.0
      est_base_gb = round(exact_quant.total_size_gb * (16.0 / max(1.0, bits)), 1)
      quants.append(
        QuantizationInfo(
          name="BF16 / FP16 (Base Model reference)",
          total_size_bytes=int(est_base_gb * (1024**3)),
          total_size_gb=est_base_gb,
          file_count=1,
          files=[],
          is_estimated=True,
          bits_per_weight=16.0,
        )
      )
      return quants

    # Standard Unquantized Base Hugging Face model (safetensors / pytorch)
    total_safetensors_bytes = sum(
      getattr(s, "size", 0) or 0
      for s in siblings
      if s.rfilename.endswith(".safetensors") or s.rfilename.endswith(".bin")
    )
    
    base_gb = total_safetensors_bytes / (1024**3) if total_safetensors_bytes > 0 else 32.0

    quant_presets = [
      ("BF16 / FP16 (Unquantized)", 16.0, base_gb if total_safetensors_bytes > 0 else 64.0),
      ("FP8 / INT8 (8-bit)", 8.0, base_gb * 0.52 if total_safetensors_bytes > 0 else 32.0),
      ("Q8_0 (8.5 bpw)", 8.5, base_gb * 0.55 if total_safetensors_bytes > 0 else 34.0),
      ("Q6_K (6.6 bpw)", 6.6, base_gb * 0.44 if total_safetensors_bytes > 0 else 27.0),
      ("Q5_K_M (5.5 bpw)", 5.5, base_gb * 0.37 if total_safetensors_bytes > 0 else 23.0),
      ("Q4_K_M (4.5 bpw - Recommended)", 4.5, base_gb * 0.31 if total_safetensors_bytes > 0 else 19.0),
      ("Q3_K_M (3.5 bpw)", 3.5, base_gb * 0.25 if total_safetensors_bytes > 0 else 15.0),
      ("Q2_K (2.7 bpw)", 2.7, base_gb * 0.20 if total_safetensors_bytes > 0 else 12.0),
      ("IQ1_S / IQ1_M (1.7 bpw)", 1.7, base_gb * 0.14 if total_safetensors_bytes > 0 else 8.5),
    ]

    quants = []
    for qname, bpw, gb in quant_presets:
      bytes_val = int(gb * (1024**3))
      quants.append(
        QuantizationInfo(
          name=qname,
          total_size_bytes=bytes_val,
          total_size_gb=round(gb, 2),
          file_count=1,
          files=[],
          is_estimated=True,
          bits_per_weight=bpw,
        )
      )
    return quants


  def fetch_architecture(
    self, repo_id: str, repo_info: Any, selected_quant_files: Optional[List[str]] = None
  ) -> ModelArchitecture:
    """
    Extracts structural information and model-specific configurations.
    Priority:
    1. GGUF header range request (if GGUF repo)
    2. config.json in base_model (if linked)
    3. config.json in repo directly
    4. Heuristic / fallback from repo card tags
    """
    siblings = getattr(repo_info, "siblings", [])
    gguf_files = [s for s in siblings if s.rfilename.endswith(".gguf")]
    is_gguf = len(gguf_files) > 0

    base_model = None
    if hasattr(repo_info, "card_data") and repo_info.card_data:
      bm = getattr(repo_info.card_data, "base_model", None)
      if isinstance(bm, list) and len(bm) > 0:
        base_model = bm[0]
      elif isinstance(bm, str):
        base_model = bm

    if not base_model and hasattr(repo_info, "tags"):
      for tag in repo_info.tags:
        if tag.startswith("base_model:"):
          base_model = tag.split("base_model:")[-1].replace("quantized:", "").strip()
          break

    gguf_metadata: Dict[str, Any] = {}
    if is_gguf:
      target_file = None
      if selected_quant_files and len(selected_quant_files) > 0:
        target_file = selected_quant_files[0]
      else:
        for gf in gguf_files:
          if "mmproj" not in gf.rfilename.lower():
            target_file = gf.rfilename
            break
      
      if target_file:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{target_file}"
        gguf_metadata = parse_gguf_header_range(url)

    config_json: Dict[str, Any] = {}
    config_sources = [base_model, repo_id] if base_model else [repo_id]
    for src in config_sources:
      if not src:
        continue
      url = f"https://huggingface.co/{src}/raw/main/config.json"
      try:
        req = urllib.request.Request(url, headers={"User-Agent": "MemoryEstimate/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
          config_json = json.loads(resp.read().decode("utf-8"))
          if config_json:
            break
      except Exception:
        continue

    return self._build_architecture_obj(
      repo_id=repo_id,
      is_gguf=is_gguf,
      base_model=base_model,
      gguf_meta=gguf_metadata,
      config=config_json,
    )

  def _build_architecture_obj(
    self,
    repo_id: str,
    is_gguf: bool,
    base_model: Optional[str],
    gguf_meta: Dict[str, Any],
    config: Dict[str, Any],
  ) -> ModelArchitecture:
    """
    Consolidates config.json and GGUF header into a clean ModelArchitecture dataclass.
    """
    cfg = config.get("text_config", config)

    arch_type = "transformer"
    if "general.architecture" in gguf_meta:
      arch_type = str(gguf_meta["general.architecture"]).lower()
    elif "architectures" in cfg and cfg["architectures"]:
      arch_type = cfg["architectures"][0].lower()
    elif "model_type" in cfg:
      arch_type = str(cfg["model_type"]).lower()

    model_name = repo_id.split("/")[-1]
    if "general.name" in gguf_meta:
      model_name = str(gguf_meta["general.name"])

    arch_prefix = f"{arch_type}." if arch_type else ""

    num_layers = (
      cfg.get("num_hidden_layers")
      or cfg.get("n_layer")
      or cfg.get("num_layers")
      or gguf_meta.get(f"{arch_prefix}block_count")
      or gguf_meta.get("llama.block_count")
      or 32
    )

    hidden_size = (
      cfg.get("hidden_size")
      or cfg.get("n_embd")
      or gguf_meta.get(f"{arch_prefix}embedding_length")
      or gguf_meta.get("llama.embedding_length")
      or 4096
    )

    num_heads = (
      cfg.get("num_attention_heads")
      or cfg.get("n_head")
      or gguf_meta.get(f"{arch_prefix}attention.head_count")
      or gguf_meta.get("llama.attention.head_count")
      or 32
    )

    num_kv_heads = (
      cfg.get("num_key_value_heads")
      or cfg.get("n_head_kv")
      or gguf_meta.get(f"{arch_prefix}attention.head_count_kv")
      or gguf_meta.get("llama.attention.head_count_kv")
      or num_heads
    )

    explicit_head_dim = (
      cfg.get("head_dim")
      or gguf_meta.get(f"{arch_prefix}attention.key_length")
      or gguf_meta.get("llama.attention.key_length")
    )
    head_dim = explicit_head_dim if explicit_head_dim else (hidden_size // num_heads if num_heads else 128)

    max_context = (
      cfg.get("max_position_embeddings")
      or cfg.get("max_sequence_length")
      or cfg.get("seq_length")
      or gguf_meta.get(f"{arch_prefix}context_length")
      or gguf_meta.get("llama.context_length")
      or 32768
    )

    vocab_size = cfg.get("vocab_size") or gguf_meta.get(f"{arch_prefix}vocab_size")

    num_routed_experts = (
      cfg.get("n_routed_experts")
      or cfg.get("num_local_experts")
      or cfg.get("num_experts")
      or cfg.get("n_experts")
      or cfg.get("expert_count")
      or gguf_meta.get(f"{arch_prefix}expert_count")
    )
    num_experts_per_tok = (
      cfg.get("num_experts_per_tok")
      or cfg.get("num_experts_per_token")
      or cfg.get("n_routed_experts_per_tok")
      or cfg.get("expert_used_count")
      or gguf_meta.get(f"{arch_prefix}expert_used_count")
    )
    num_shared_experts = cfg.get("n_shared_experts") or gguf_meta.get(f"{arch_prefix}expert_shared_count")

    is_moe = bool(num_routed_experts and num_routed_experts > 1)

    kv_lora_rank = (
      cfg.get("kv_lora_rank")
      or gguf_meta.get(f"{arch_prefix}attention.kv_lora_rank")
      or gguf_meta.get("deepseek2.attention.kv_lora_rank")
    )
    qk_rope_head_dim = (
      cfg.get("qk_rope_head_dim")
      or gguf_meta.get(f"{arch_prefix}rope.dimension_count")
    )
    is_mla = bool(kv_lora_rank is not None and kv_lora_rank > 0)

    layer_types = cfg.get("layer_types", [])
    full_attn_interval = cfg.get("full_attention_interval")
    is_hybrid_linear = False
    num_full_attn = num_layers
    num_linear_attn = 0

    if layer_types:
      num_linear_attn = sum(1 for lt in layer_types if "linear" in lt)
      num_full_attn = sum(1 for lt in layer_types if "linear" not in lt and ("attention" in lt or "sparse" in lt))
      if num_linear_attn > 0:
        is_hybrid_linear = True
    elif full_attn_interval and full_attn_interval > 1:
      # e.g. Qwen3.5/3.6 MoE full_attention_interval = 4
      is_hybrid_linear = True
      num_full_attn = max(1, num_layers // full_attn_interval)
      num_linear_attn = num_layers - num_full_attn
    elif "glm5next" in arch_type.lower():
      is_hybrid_linear = True
      num_full_attn = 11
      num_linear_attn = num_layers - num_full_attn

    sliding_window = cfg.get("sliding_window") or gguf_meta.get(f"{arch_prefix}attention.sliding_window_size")

    notes = []
    cfg_quant = cfg.get("quantization") or cfg.get("quantization_config")
    if isinstance(cfg_quant, dict):
      q_bits = cfg_quant.get("bits") or cfg_quant.get("bits_per_weight")
      q_mode = cfg_quant.get("mode") or cfg_quant.get("quant_method") or ""
      notes.append(
        f"🎯 Pre-Quantized Model Detected ({q_bits}-bit {q_mode}). "
        "Weights are extracted with exact precision directly from repository shards."
      )

    if is_mla:
      notes.append(
        f"Multi-Head Latent Attention (MLA) detected (kv_lora_rank={kv_lora_rank}). "
        "KV cache memory is compressed ~5x compared to standard GQA, vastly reducing RAM at deep contexts!"
      )
    if is_hybrid_linear:
      notes.append(
        f"Hybrid Linear Attention architecture detected ({num_full_attn} full attention layers, "
        f"{num_linear_attn} linear attention layers). Linear attention layers store fixed-size recurrent states, "
        "preventing linear KV cache memory growth across all layers!"
      )
    if is_moe:
      notes.append(
        f"Mixture-of-Experts (MoE) detected: {num_routed_experts} total experts, "
        f"{num_experts_per_tok} active per token. All expert weights must reside in Unified RAM, "
        f"while activation memory is bounded by active experts."
      )
    if num_kv_heads < num_heads and not is_mla:
      gqa_ratio = num_heads // num_kv_heads if num_kv_heads else 1
      notes.append(f"Grouped-Query Attention (GQA {gqa_ratio}:1) reduces KV cache by {gqa_ratio}x relative to MHA.")

    return ModelArchitecture(
      repo_id=repo_id,
      model_name=model_name,
      architecture_type=arch_type,
      is_gguf=is_gguf,
      num_hidden_layers=int(num_layers),
      hidden_size=int(hidden_size),
      num_attention_heads=int(num_heads),
      num_key_value_heads=int(num_kv_heads),
      head_dim=int(head_dim),
      max_position_embeddings=int(max_context),
      vocab_size=int(vocab_size) if vocab_size else None,
      is_moe=is_moe,
      num_routed_experts=int(num_routed_experts) if num_routed_experts else None,
      num_experts_per_tok=int(num_experts_per_tok) if num_experts_per_tok else None,
      num_shared_experts=int(num_shared_experts) if num_shared_experts else None,
      is_mla=is_mla,
      kv_lora_rank=int(kv_lora_rank) if kv_lora_rank else None,
      qk_rope_head_dim=int(qk_rope_head_dim) if qk_rope_head_dim else 0,
      is_hybrid_linear=is_hybrid_linear,
      num_full_attention_layers=num_full_attn,
      num_linear_attention_layers=num_linear_attn,
      sliding_window=int(sliding_window) if sliding_window else None,
      base_model=base_model,
      special_notes=notes,
    )
