"""
Streamlit App: Unified RAM Estimator for LLMs in Agentic Coding.
Calculates memory required on Unified Memory (e.g. Apple Silicon Mac)
based on exact HuggingFace metadata, GGUF quantizations, and model architectures.
Indent: 2 spaces.
"""

import math
from typing import Optional, List, Dict, Any
import pandas as pd
import streamlit as st

from calculator import (
  KVCachePrecision,
  calculate_unified_ram,
  calculate_kv_cache_bytes,
  evaluate_all_mac_tiers,
  MemoryBreakdown,
)
from model_fetcher import (
  ModelMetadataFetcher,
  ModelArchitecture,
  QuantizationInfo,
  parse_hf_url,
)

# Page configuration
st.set_page_config(
  page_title="LLM Unified RAM Estimator | Agentic Coding",
  page_icon="🧠",
  layout="wide",
  initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
  """
  <style>
    /* Base Variables (Light Mode Default) */
    :root {
      --bg-card: #f8fafc;
      --border-card: #e2e8f0;
      --text-title: #0f172a;
      --text-notes: #475569;
      --text-sub: #64748b;

      --card-primary-bg: #eff6ff;
      --card-primary-border: #3b82f6;
      --text-primary-title: #1d4ed8;
      --text-primary-val: #1e3a8a;
      --text-primary-sub: #3b82f6;

      --opt-bg: rgba(240, 253, 244, 0.95);
      --opt-border: #86efac;
      --opt-badge-bg: #dcfce7;
      --opt-badge-text: #15803d;
      --opt-badge-border: #bbf7d0;

      --tight-bg: rgba(254, 252, 232, 0.95);
      --tight-border: #fde047;
      --tight-badge-bg: #fef9c3;
      --tight-badge-text: #a16207;
      --tight-badge-border: #fef08a;

      --oom-bg: rgba(254, 242, 242, 0.95);
      --oom-border: #fca5a5;
      --oom-badge-bg: #fee2e2;
      --oom-badge-text: #b91c1c;
      --oom-badge-border: #fecaca;
    }

    /* Dark Mode (OS level prefers-color-scheme) */
    @media (prefers-color-scheme: dark) {
      :root {
        --bg-card: rgba(30, 41, 59, 0.75);
        --border-card: rgba(148, 163, 184, 0.2);
        --text-title: #f8fafc;
        --text-notes: #cbd5e1;
        --text-sub: #94a3b8;

        --card-primary-bg: rgba(30, 58, 138, 0.35);
        --card-primary-border: #60a5fa;
        --text-primary-title: #93c5fd;
        --text-primary-val: #bfdbfe;
        --text-primary-sub: #60a5fa;

        --opt-bg: rgba(20, 83, 45, 0.25);
        --opt-border: rgba(74, 222, 128, 0.35);
        --opt-badge-bg: rgba(34, 197, 94, 0.2);
        --opt-badge-text: #4ade80;
        --opt-badge-border: rgba(74, 222, 128, 0.4);

        --tight-bg: rgba(113, 63, 18, 0.25);
        --tight-border: rgba(250, 204, 21, 0.35);
        --tight-badge-bg: rgba(234, 179, 8, 0.2);
        --tight-badge-text: #fde047;
        --tight-badge-border: rgba(250, 204, 21, 0.4);

        --oom-bg: rgba(127, 29, 29, 0.25);
        --oom-border: rgba(248, 113, 113, 0.35);
        --oom-badge-bg: rgba(239, 68, 68, 0.2);
        --oom-badge-text: #f87171;
        --oom-badge-border: rgba(248, 113, 113, 0.4);
      }
    }

    /* Support Streamlit Manual Dark Theme Attribute */
    [data-theme="dark"], [data-base-theme="dark"], .stApp[data-theme="dark"],
    [data-testid="stAppViewContainer"][data-theme="dark"] {
      --bg-card: rgba(30, 41, 59, 0.75);
      --border-card: rgba(148, 163, 184, 0.2);
      --text-title: #f8fafc;
      --text-notes: #cbd5e1;
      --text-sub: #94a3b8;

      --card-primary-bg: rgba(30, 58, 138, 0.35);
      --card-primary-border: #60a5fa;
      --text-primary-title: #93c5fd;
      --text-primary-val: #bfdbfe;
      --text-primary-sub: #60a5fa;

      --opt-bg: rgba(20, 83, 45, 0.25);
      --opt-border: rgba(74, 222, 128, 0.35);
      --opt-badge-bg: rgba(34, 197, 94, 0.2);
      --opt-badge-text: #4ade80;
      --opt-badge-border: rgba(74, 222, 128, 0.4);

      --tight-bg: rgba(113, 63, 18, 0.25);
      --tight-border: rgba(250, 204, 21, 0.35);
      --tight-badge-bg: rgba(234, 179, 8, 0.2);
      --tight-badge-text: #fde047;
      --tight-badge-border: rgba(250, 204, 21, 0.4);

      --oom-bg: rgba(127, 29, 29, 0.25);
      --oom-border: rgba(248, 113, 113, 0.35);
      --oom-badge-bg: rgba(239, 68, 68, 0.2);
      --oom-badge-text: #f87171;
      --oom-badge-border: rgba(248, 113, 113, 0.4);
    }

    .metric-card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 10px;
      padding: 16px;
      text-align: center;
      margin-bottom: 10px;
      box-sizing: border-box;
      transition: all 0.2s ease;
    }
    .metric-primary {
      background: var(--card-primary-bg) !important;
      border: 2px solid var(--card-primary-border) !important;
    }
    .metric-label-primary {
      font-size: 0.8rem;
      color: var(--text-primary-title);
      font-weight: 700;
      letter-spacing: 0.03em;
    }
    .metric-value-primary {
      font-size: 1.9rem;
      font-weight: 800;
      color: var(--text-primary-val);
      margin: 4px 0;
    }
    .metric-sub-primary {
      font-size: 0.72rem;
      color: var(--text-primary-sub);
    }
    .metric-label {
      font-size: 0.8rem;
      color: var(--text-sub);
      font-weight: 600;
      letter-spacing: 0.03em;
    }
    .metric-value {
      font-size: 1.7rem;
      font-weight: 700;
      color: var(--text-title);
      margin: 4px 0;
    }
    .metric-sub {
      font-size: 0.72rem;
      color: var(--text-sub);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* Hardware Matrix Cards with Dark Mode Accent Colors */
    .hw-card {
      border-radius: 10px;
      padding: 16px;
      text-align: left;
      height: 168px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      margin-bottom: 12px;
      box-sizing: border-box;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .hw-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    .hw-card-optimal {
      background: var(--opt-bg);
      border: 1px solid var(--opt-border);
    }
    .hw-card-tight {
      background: var(--tight-bg);
      border: 1px solid var(--tight-border);
    }
    .hw-card-oom {
      background: var(--oom-bg);
      border: 1px solid var(--oom-border);
    }
    .hw-tier-title {
      font-weight: 700;
      font-size: 1.05rem;
      margin-bottom: 6px;
      color: var(--text-title);
    }
    .hw-tier-notes {
      font-size: 0.78rem;
      color: var(--text-notes);
      line-height: 1.4;
      margin-top: 6px;
    }

    /* Status Badges */
    .badge-optimal {
      background-color: var(--opt-badge-bg);
      color: var(--opt-badge-text);
      border: 1px solid var(--opt-badge-border);
      padding: 4px 10px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.82rem;
      display: inline-block;
    }
    .badge-tight {
      background-color: var(--tight-badge-bg);
      color: var(--tight-badge-text);
      border: 1px solid var(--tight-badge-border);
      padding: 4px 10px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.82rem;
      display: inline-block;
    }
    .badge-oom {
      background-color: var(--oom-badge-bg);
      color: var(--oom-badge-text);
      border: 1px solid var(--oom-badge-border);
      padding: 4px 10px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.82rem;
      display: inline-block;
    }
    .stAlert {
      border-radius: 8px;
    }
  </style>
  """,
  unsafe_allow_html=True,
)



@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_repo_metadata(repo_id: str, hf_token: Optional[str] = None):
  fetcher = ModelMetadataFetcher(hf_token=hf_token)
  info = fetcher.fetch_repo_info(repo_id)
  quants = fetcher.get_available_quantizations(repo_id, info)
  return info, quants


@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_architecture(
  repo_id: str,
  _repo_info: Any,
  selected_quant_files: Optional[List[str]] = None,
  hf_token: Optional[str] = None,
):
  fetcher = ModelMetadataFetcher(hf_token=hf_token)
  return fetcher.fetch_architecture(repo_id, _repo_info, selected_quant_files)


def render_header():
  st.title("🧠 LLM Unified RAM Estimator for Agentic Coding")
  st.markdown(
    """
    Accurately estimate unified RAM requirements (e.g. Apple Silicon Mac) for deep-context agentic coding workflows.
    Fetches real metadata, sharded GGUF file sizes, and architecture specifications directly from Hugging Face.
    """
  )


def main():
  render_header()

  # Sidebar: Quick Presets & Token configuration
  with st.sidebar:
    st.header("⚙️ Model Selection")
    
    preset_choice = st.selectbox(
      "Quick Model Presets:",
      [
        "Custom URL / Repo ID",
        "unsloth/GLM-5.3-Flash-GGUF (MLA + Hybrid MoE)",
        "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit (MLX 3-bit MoE)",
        "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF (GQA 32B)",
        "Qwen/Qwen2.5-Coder-32B-Instruct (Base Repo)",
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF (Llama 3.1)",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (MLA MoE)",
      ],
      index=1,
    )

    preset_map = {
      "unsloth/GLM-5.3-Flash-GGUF (MLA + Hybrid MoE)": "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF",
      "unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit (MLX 3-bit MoE)": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit",
      "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF (GQA 32B)": "https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
      "Qwen/Qwen2.5-Coder-32B-Instruct (Base Repo)": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
      "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF (Llama 3.1)": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
      "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (MLA MoE)": "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    }

    initial_url = preset_map.get(preset_choice, "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF")

    hf_url_input = st.text_input(
      "Hugging Face Model URL or Repo ID:",
      value=initial_url,
      placeholder="e.g. https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF",
      help="Paste any Hugging Face model URL, tree URL, or repo ID (e.g. unsloth/GLM-5.3-Flash-GGUF)",
    )

    hf_token = st.text_input(
      "Hugging Face Token (Optional):",
      type="password",
      help="Required only for private or gated models (e.g., official meta-llama/Llama-3.1). Not needed for GGUFs.",
    )

    st.divider()
    st.markdown("### 🛠️ Agentic Coding Parameters")
    st.caption("Realistic settings for multi-turn agent loops, test runs, and repo browsing.")

  # Extract clean repo ID
  clean_repo_id = parse_hf_url(hf_url_input)
  if not clean_repo_id:
    st.info("👈 Enter a valid Hugging Face URL or choose a preset in the sidebar to begin.")
    return

  # Fetch metadata
  with st.spinner(f"Fetching metadata for `{clean_repo_id}` from Hugging Face..."):
    try:
      repo_info, quants = cached_fetch_repo_metadata(clean_repo_id, hf_token if hf_token else None)
    except Exception as e:
      st.error(f"Failed to fetch repository information for `{clean_repo_id}`: {e}")
      st.caption("Please check the repository URL or provide a Hugging Face token if the repo is gated.")
      return

  if not quants:
    st.warning("No model files or quantization formats found in this repository.")
    return

  # Quantization selection in main view
  col_quant_select, col_quant_info = st.columns([2, 1])

  with col_quant_select:
    # Prepare labels for quantizations
    quant_options = []
    quant_map: Dict[str, QuantizationInfo] = {}

    for q in quants:
      label = f"{q.name} ({q.total_size_gb:.2f} GB"
      if q.file_count > 1:
        label += f", {q.file_count} shards"
      if q.is_estimated:
        label += ", estimated"
      label += ")"
      quant_options.append(label)
      quant_map[label] = q

    default_idx = 0
    for idx, opt in enumerate(quant_options):
      if any(k in opt for k in ["Repo Exact", "UD-Q3_K_XL", "Q4_K_M", "Q4_K", "FP8"]):
        default_idx = idx
        break

    selected_opt = st.selectbox(
      "Select Quantization Variant:",
      options=quant_options,
      index=default_idx,
      help="Choose the quantization level. Sharded GGUFs sum all parts to calculate exact weight memory.",
    )
    if selected_opt and selected_opt in quant_map:
      selected_quant = quant_map[selected_opt]
    else:
      selected_quant = quants[0]


  with col_quant_info:
    st.markdown(
      f"""
      <div class="metric-card">
        <div class="metric-label">MODEL WEIGHTS (VRAM)</div>
        <div class="metric-value">{selected_quant.total_size_gb:.2f} GB</div>
        <div class="metric-sub">{selected_quant.file_count} file(s) | {selected_quant.name}</div>
      </div>
      """,
      unsafe_allow_html=True,
    )


  # Fetch Architecture
  with st.spinner("Extracting model architecture & attention structure..."):
    try:
      arch = cached_fetch_architecture(
        clean_repo_id,
        repo_info,
        selected_quant.files,
        hf_token if hf_token else None,
      )
    except Exception as e:
      st.warning(f"Could not automatically parse architecture: {e}. Using default transformer architecture.")
      arch = ModelArchitecture(
        repo_id=clean_repo_id,
        model_name=clean_repo_id.split("/")[-1],
        architecture_type="transformer",
        is_gguf=len(selected_quant.files) > 0,
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=131072,
      )

  # Sidebar Controls: Agentic Parameters
  with st.sidebar:
    # Context window setting
    max_native = arch.max_position_embeddings or 131072
    # Ensure slider range has sane minimum and maximum
    slider_max = max(131072, min(max_native, 262144))
    
    # Realistic default for agentic coding: 64k tokens (or max native if smaller)
    default_context = min(65536, max_native)
    if default_context < 8192:
      default_context = max_native

    context_choice = st.select_slider(
      "Context Window Depth (tokens):",
      options=[8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144],
      value=default_context if default_context in [8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144] else 65536,
      help="Agentic coding typically operates at 32k to 64k+ tokens to retain tool calls, test runs, and repo context.",
    )

    # KV Cache Precision
    kv_prec_choice = st.selectbox(
      "KV Cache Quantization:",
      options=list(KVCachePrecision),
      format_func=lambda x: x.value,
      index=1,  # Default to FP8 (Q8_0)
      help="FP8 / Q8_0 is the recommended modern default for agentic coding: saves 50% KV cache memory with zero loss in code quality.",
    )

    # Headroom Slider
    headroom_val = st.slider(
      "Required Headroom (GB):",
      min_value=2.0,
      max_value=20.0,
      value=6.0,
      step=0.5,
      help="Memory reserved for macOS, display framebuffers (4K/5K monitors), VSCode/Cursor, browser, and background developer apps.",
    )

    # Advanced Prefill Batch Size
    with st.expander("Advanced Inference Settings"):
      ubatch_size = st.select_slider(
        "Prefill Chunk Size (ubatch):",
        options=[256, 512, 1024, 2048],
        value=1024,
        help="In llama.cpp / Ollama, chunk size evaluated per forward pass during prompt prefill.",
      )

  # Calculation
  calc = calculate_unified_ram(
    arch=arch,
    selected_quant=selected_quant,
    context_tokens=context_choice,
    kv_precision=kv_prec_choice,
    headroom_gb=headroom_val,
    ubatch_chunk_size=ubatch_size,
  )

  # Display Main Metrics Row
  m1, m2, m3, m4, m5 = st.columns(5)
  with m1:
    st.markdown(
      f"""
      <div class="metric-card metric-primary">
        <div class="metric-label-primary">TOTAL UNIFIED RAM NEEDED</div>
        <div class="metric-value-primary">{calc.total_ram_required_gb:.1f} GB</div>
        <div class="metric-sub-primary">At {context_choice // 1024}k context ({calc.kv_cache_precision})</div>
      </div>
      """,
      unsafe_allow_html=True,
    )
  with m2:
    st.markdown(
      f"""
      <div class="metric-card">
        <div class="metric-label">MODEL WEIGHTS</div>
        <div class="metric-value">{calc.weights_gb:.1f} GB</div>
        <div class="metric-sub">{selected_quant.name}</div>
      </div>
      """,
      unsafe_allow_html=True,
    )
  with m3:
    st.markdown(
      f"""
      <div class="metric-card">
        <div class="metric-label">KV CACHE MEMORY</div>
        <div class="metric-value">{calc.kv_cache_gb:.1f} GB</div>
        <div class="metric-sub">{calc.kv_bytes_per_token:.1f} bytes / token</div>
      </div>
      """,
      unsafe_allow_html=True,
    )
  with m4:
    st.markdown(
      f"""
      <div class="metric-card">
        <div class="metric-label">ACTIVATION SCRATCH</div>
        <div class="metric-value">{calc.activation_scratch_gb:.1f} GB</div>
        <div class="metric-sub">Prefill chunk: {ubatch_size}</div>
      </div>
      """,
      unsafe_allow_html=True,
    )
  with m5:
    st.markdown(
      f"""
      <div class="metric-card">
        <div class="metric-label">SYSTEM HEADROOM</div>
        <div class="metric-value">{calc.headroom_gb:.1f} GB</div>
        <div class="metric-sub">macOS + Display + IDE</div>
      </div>
      """,
      unsafe_allow_html=True,
    )

  st.write("")

  # Visual Breakdown Progress Bar
  st.subheader("📊 Memory Allocation Breakdown")
  col_bar, col_legend = st.columns([3, 1])

  with col_bar:
    # Prepare breakdown data
    df_breakdown = pd.DataFrame(
      {
        "Component": ["Model Weights", "KV Cache", "Prefill Scratchpad", "System Headroom"],
        "Memory (GB)": [
          calc.weights_gb,
          calc.kv_cache_gb,
          calc.activation_scratch_gb,
          calc.headroom_gb,
        ],
      }
    )
    df_breakdown["Percentage"] = (
      df_breakdown["Memory (GB)"] / calc.total_ram_required_gb * 100.0
    ).round(1)

    st.bar_chart(
      df_breakdown.set_index("Component")["Memory (GB)"],
      horizontal=True,
    )

  with col_legend:
    st.markdown("##### Allocation Share:")
    for _, row in df_breakdown.iterrows():
      st.markdown(f"- **{row['Component']}**: `{row['Memory (GB)']} GB` ({row['Percentage']}%)")

  # Model-Specific Considerations Alert Box
  st.write("")
  if arch.special_notes or calc.insights:
    st.subheader("💡 Model-Specific Considerations & Agentic Insights")
    for note in arch.special_notes:
      st.info(note)
    for insight in calc.insights:
      st.success(insight)

  # Mac Hardware Sizing & Compatibility Matrix
  st.write("")
  st.subheader("💻 Apple Silicon Unified RAM Compatibility Matrix")
  st.caption("How this model configuration fits across various Mac unified memory tiers:")

  mac_evals = evaluate_all_mac_tiers(calc.total_ram_required_gb)

  # Display in 4-column responsive grid
  cols = st.columns(4)
  for idx, ev in enumerate(mac_evals):
    col = cols[idx % 4]
    with col:
      if ev.status == "OPTIMAL":
        card_class = "hw-card hw-card-optimal"
        badge_html = f'<span class="badge-optimal">🟢 OPTIMAL ({ev.free_ram_gb:.1f} GB Free)</span>'
      elif ev.status == "TIGHT":
        card_class = "hw-card hw-card-tight"
        badge_html = f'<span class="badge-tight">🟡 TIGHT ({ev.free_ram_gb:.1f} GB Free)</span>'
      else:
        card_class = "hw-card hw-card-oom"
        badge_html = f'<span class="badge-oom">🔴 WILL SWAP / OOM ({abs(ev.free_ram_gb):.1f} GB Short)</span>'

      st.markdown(
        f"""
        <div class="{card_class}">
          <div class="hw-tier-title">{ev.tier_name}</div>
          <div style="margin-bottom: 6px;">{badge_html}</div>
          <div class="hw-tier-notes">{ev.notes}</div>
        </div>
        """,
        unsafe_allow_html=True,
      )


  # Interactive Context Scaling Chart
  st.write("")
  st.subheader("📈 Context Scaling Curve: KV Cache RAM vs Context Depth")
  st.caption("Demonstrating how KV Cache memory grows as agentic coding context expands from 8k to 131k tokens:")

  test_contexts = [8192, 16384, 32768, 49152, 65536, 98304, 131072]
  fp16_kv_vals = []
  fp8_kv_vals = []
  q4_kv_vals = []

  for ctx in test_contexts:
    b_fp16, _ = calculate_kv_cache_bytes(arch, ctx, KVCachePrecision.FP16)
    b_fp8, _ = calculate_kv_cache_bytes(arch, ctx, KVCachePrecision.FP8)
    b_q4, _ = calculate_kv_cache_bytes(arch, ctx, KVCachePrecision.Q4_0)
    fp16_kv_vals.append(round(b_fp16 / (1024**3), 2))
    fp8_kv_vals.append(round(b_fp8 / (1024**3), 2))
    q4_kv_vals.append(round(b_q4 / (1024**3), 2))

  df_scaling = pd.DataFrame(
    {
      "Context (Tokens)": [f"{c // 1024}k" for c in test_contexts],
      "FP16 KV Cache (GB)": fp16_kv_vals,
      "FP8 / Q8_0 KV Cache (GB)": fp8_kv_vals,
      "Q4_0 KV Cache (GB)": q4_kv_vals,
    }
  ).set_index("Context (Tokens)")

  st.line_chart(df_scaling)

  # Model Architecture Technical Inspection
  with st.expander("🔍 Model Architecture & Technical Metadata"):
    c1, c2 = st.columns(2)
    with c1:
      st.markdown("##### Core Hyperparameters")
      st.markdown(f"- **Architecture Type**: `{arch.architecture_type}`")
      st.markdown(f"- **Hidden Layers**: `{arch.num_hidden_layers}`")
      st.markdown(f"- **Hidden Size**: `{arch.hidden_size}`")
      st.markdown(f"- **Attention Heads**: `{arch.num_attention_heads}`")
      st.markdown(f"- **KV Heads (GQA)**: `{arch.num_key_value_heads}`")
      st.markdown(f"- **Head Dimension**: `{arch.head_dim}`")
      st.markdown(f"- **Native Max Context**: `{arch.max_position_embeddings:,}` tokens")
    with c2:
      st.markdown("##### Advanced Architectural Features")
      st.markdown(f"- **MLA (Multi-Head Latent Attention)**: `{'Yes' if arch.is_mla else 'No'}`")
      if arch.is_mla:
        st.markdown(f"  - Latent KV LoRA Rank: `{arch.kv_lora_rank}`")
        st.markdown(f"  - Decoupled RoPE Dim: `{arch.qk_rope_head_dim}`")
      st.markdown(f"- **Hybrid Linear Attention**: `{'Yes' if arch.is_hybrid_linear else 'No'}`")
      if arch.is_hybrid_linear:
        st.markdown(f"  - Full MLA Layers: `{arch.num_full_attention_layers}`")
        st.markdown(f"  - Linear Attention Layers: `{arch.num_linear_attention_layers}`")
      st.markdown(f"- **Mixture-of-Experts (MoE)**: `{'Yes' if arch.is_moe else 'No'}`")
      if arch.is_moe:
        st.markdown(f"  - Total Routed Experts: `{arch.num_routed_experts}`")
        st.markdown(f"  - Active Experts per Token: `{arch.num_experts_per_tok}`")
      if arch.base_model:
        st.markdown(f"- **Linked Base Model**: `{arch.base_model}`")

  # Agentic Coding Practical Deployment Tips
  st.write("")
  with st.expander("🚀 Practical Deployment Tips for Agentic Coding on macOS"):
    st.markdown(
      """
      ### Why Memory Management is Critical for Agentic Coding:
      1. **Context Accumulation**:
        Unlike simple chatbots that handle 1,000-token queries, coding agents (Aider, Claude Code, Cursor, Roo Code)
        accumulate file trees, entire source files, compiler errors, and git diffs over 10-30 turns.
        The context stays near 32k-100k tokens throughout the session.

      2. **Swapping Destroys Inference Speed**:
        When unified memory exceeds physical capacity, macOS starts swapping to NVMe SSD.
        Inference speeds collapse from **25 tokens/sec down to 0.5 tokens/sec**, rendering the agent unusable.
      3. **Recommended llama.cpp / Ollama Flags**:
        - Enable 8-bit KV Cache: `--cache-type-k q8_0 --cache-type-v q8_0`
        - Set context length: `-c 65536`
        - Tune prefill chunk: `-b 1024 -ub 1024`
      4. **Raising macOS Wired Memory Limit**:
        By default, macOS caps Metal unified allocations at ~75% of physical RAM.
        To allow up to 85-90% allocation on high-end Macs (e.g. 128GB or 192GB), run in Terminal:
        ```bash
        sudo sysctl iogpu.wired_mem_limit=<bytes>
        ```
      """

    )


if __name__ == "__main__":
  main()
