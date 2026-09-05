# LLM Unified RAM Estimator for Agentic Coding 🧠

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://memory-estimate.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](pyproject.toml)
[![Documentation](https://img.shields.io/badge/Docs-GitHub%20Pages-38bdf8.svg)](https://saehwanpark.github.io/memory-estimate/)
[![Tests](https://img.shields.io/badge/Tests-pytest-green.svg)](tests/)

An interactive Streamlit application to calculate the exact Unified RAM required to run Large Language Models in **agentic coding** workflows with extensive context depth.

Instead of relying on simple total parameter heuristics, this tool fetches real repository metadata, inspects remote GGUF binary headers via range requests, groups sharded files, and analyzes exact architecture configurations directly from **Hugging Face**.

🚀 **[Launch Live Web Application (Streamlit Cloud)](https://memory-estimate.streamlit.app/)**  
📖 **[Read Full Documentation & Background (English & 한국어)](https://saehwanpark.github.io/memory-estimate/)**

---

## ⚡ Key Highlights & Differentiation

| Capability | This Application | Can-It-Run / Web Tools | Hugging Face Calc |
| :--- | :--- | :--- | :--- |
| **Real-Time HF Ingestion** | **Zero-download** shard byte sums & range headers | Manual parameter entry | Base models only |
| **MLA (Multi-Head Latent Attn)** | **Supported** (DeepSeek V2/V3, GLM-5 ~5x compression) | Unsupported (5-10x error) | Unsupported |
| **Hybrid Linear Attention** | **Supported** (GLM-5.3-Flash, Qwen3.6 MoE) | Unsupported | Unsupported |
| **Pre-Quantized MLX Detection** | **Exact Detection** (3-bit/4-bit MLX safetensors) | Unsupported | Unsupported |
| **Deep Context Focus** | **32k to 128k+ tokens**, FP8 KV cache, prefill chunks | 2k-8k tokens only | Single forward pass |
| **Apple Silicon Sizing Matrix** | **16 GB to 512 GB tiers** with macOS headroom | Generic VRAM | Generic VRAM |

---

## 🧮 Memory Calculation Formula

$$\text{Unified RAM Total} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{scratch}} + M_{\text{headroom}}$$

1. **Model Weights ($M_{\text{weights}}$)**:
  - **GGUF / Pre-quantized MLX**: Exact sum of sharded file bytes from repository metadata.
  - **Base Models**: Estimated via parameter count $\times$ bits per weight / 8.
2. **KV Cache Memory ($M_{\text{KV}}$)**:
  - **Standard GQA / MHA**: $2 \times N_{\text{layers}} \times H_{\text{KV}} \times D_{\text{head}} \times C \times B_{\text{elem}}$
  - **MLA (Multi-Head Latent Attention)**: $N_{\text{layers}} \times (d_c + d_R) \times C \times B_{\text{elem}}$
  - **Hybrid Linear Attention**: $N_{\text{full}} \times (d_c + d_R) \times C \times B_{\text{elem}} + N_{\text{linear}} \times S_{\text{state}}$

3. **Activation & Scratchpad ($M_{\text{scratch}}$)**:
  - Intermediate forward-pass tensor buffers and attention workspace during prompt prefill chunks (`ubatch`).
4. **Required Headroom ($M_{\text{headroom}}$)**:
  - Default **6.0 GB** reserved for macOS kernel, WindowServer, 4K/5K displays, VSCode/Cursor, and background dev processes.

---

## 🚀 Quick Start

### 🌐 Option A: Live Web App (No Installation Required)

Access the live calculator instantly in your browser:

👉 **[https://memory-estimate.streamlit.app/](https://memory-estimate.streamlit.app/)**

---

### 💻 Option B: Run Locally

#### 1. Prerequisites
Requires Python $\ge$ 3.13 and [`uv`](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/SaehwanPark/memory-estimate.git
cd memory-estimate

# Install dependencies
uv sync
```

#### 2. Launch Streamlit Application

```bash
uv run streamlit run app.py
```
Or via the main entrypoint:
```bash
uv run python main.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🧪 Testing & Verification

Run the full unit test suite with `pytest`:

```bash
uv run pytest -v
```

Type checking via `basedpyright`:

```bash
uv run basedpyright
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
