# LLM Unified RAM Estimator for Deep-Context Local Inference 🧠

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://memory-estimate.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](pyproject.toml)
[![Documentation](https://img.shields.io/badge/Docs-GitHub%20Pages-38bdf8.svg)](https://saehwanpark.github.io/memory-estimate/)
[![CI](https://github.com/SaehwanPark/memory-estimate/actions/workflows/ci.yml/badge.svg)](https://github.com/SaehwanPark/memory-estimate/actions/workflows/ci.yml)

An explainable, artifact-aware memory planner to estimate peak Unified RAM requirements for Large Language Models in **deep-context local inference and agentic coding** workflows (32k to 128k+ tokens).

Instead of guessing from nominal parameter counts, this tool fetches real repository metadata, inspects remote GGUF binary headers via range requests, groups sharded files, analyzes attention topologies (GQA, MLA, hybrid linear attention), and models actual runtime buffer heuristics directly from **Hugging Face**.

🚀 **[Launch Live Web Application (Streamlit Cloud)](https://memory-estimate.streamlit.app/)**  
📖 **[Read Full Documentation & Background (English & 한국어)](https://saehwanpark.github.io/memory-estimate/)**

---

## ⚡ Key Highlights & Differentiation

| Capability | This Application | Can-It-Run / Generic Web Calculators | Hugging Face Calc |
| :--- | :--- | :--- | :--- |
| **Real-Time HF Ingestion** | **Zero-download** exact shard byte sums & GGUF range headers | Manual parameter entry / guesses | Base models only |
| **MLA (Multi-Head Latent Attn)** | **Supported** (DeepSeek V2/V3, GLM-5 ~5x compression) | Unsupported (5-10x error) | Unsupported |
| **Compositional Hybrid Attention** | **Supported** (GQA/MLA full layers + constant linear recurrent state, e.g. Qwen3.6 MoE, GLM-5.3-Flash) | Unsupported | Unsupported |
| **Pre-Quantized MLX Detection** | **Exact Detection** (3-bit/4-bit MLX safetensors) | Unsupported | Unsupported |
| **Precise GGML Block Quantization** | **Modeled** (`q8_0` = 1.0625 B, `q4_0` = 0.5625 B, `q5_0` = 0.6875 B, asymmetric K/V types) | Assumes 0.5 / 1.0 B flat | Flat precision only |
| **Deep Context Focus** | **32k to 128k+ tokens**, prefill chunk scaling (`ubatch`), KV cache growth | 2k-8k tokens only | Single forward pass |
| **Apple Silicon Sizing Matrix** | **16 GB to 512 GB tiers** separating OS RAM fit from macOS ~75% Metal working-set limit | Generic VRAM | Generic VRAM |
| **Epistemic Precision** | **Explicitly separates** exact artifact bytes $\to$ architecture-derived $\to$ backend heuristics $\to$ user policy | Opaque single number | Opaque single number |

---

## 🧮 Memory Calculation & Epistemic Decomposition

The peak Unified RAM requirement is estimated transparently across four distinct epistemic tiers:

$$M_{\text{total}} = \underbrace{M_{\text{weights}}}_{\text{Exact / Measured}} + \underbrace{M_{\text{KV}}}_{\text{Architecture-Derived}} + \underbrace{M_{\text{scratch}}}_{\text{Backend Heuristic}} + \underbrace{M_{\text{headroom}}}_{\text{User Policy}}$$

| Component | Basis | Modeling Details |
| :--- | :--- | :--- |
| **Model Weights ($M_{\text{weights}}$)** | **Exact repository bytes** | For GGUF and pre-quantized MLX repos, calculated from exact byte sums of downloadable shards via HF API. For base repos, estimated from nominal parameters and target quant bpw. |
| **KV Cache Memory ($M_{\text{KV}}$)** | **Architecture-derived** | Computed compositionally from extracted attention hyperparameters: full-attention layers (GQA/MHA or MLA) scale with context depth, while linear-attention layers maintain constant recurrent state ($O(1)$ context memory). Supports asymmetric K/V cache formats and GGML block scale metadata overhead. |
| **Activation Scratchpad ($M_{\text{scratch}}$)** | **Backend heuristic** | Attention workspaces, context graph overhead, MoE router buffers, and forward-pass activation buffers during prompt prefill chunks (`ubatch`). |
| **System Headroom ($M_{\text{headroom}}$)** | **User policy** | Configurable memory reserve (default **6.0 GB**) for macOS kernel, WindowServer, 4K/5K framebuffers, IDE (VSCode/Cursor), browser, and background developer tooling. |

### Attention Topology Formulas

1. **Standard GQA / MHA ($N_{\text{full}}$ layers)**:
   $$M_{\text{KV, full}} = N_{\text{full}} \times H_{\text{KV}} \times D_{\text{head}} \times (B_{\text{elem, K}} + B_{\text{elem, V}}) \times C_{\text{effective}}$$

2. **Multi-Head Latent Attention / MLA ($N_{\text{full}}$ layers)**:
   $$M_{\text{KV, full}} = N_{\text{full}} \times (d_c + d_R) \times \frac{B_{\text{elem, K}} + B_{\text{elem, V}}}{2} \times C_{\text{effective}}$$

3. **Hybrid Linear Attention ($N_{\text{full}}$ full layers + $N_{\text{linear}}$ linear layers)**:
   $$M_{\text{KV}} = M_{\text{KV, full}} + N_{\text{linear}} \times S_{\text{state}}$$
   *(Linear attention layers maintain fixed recurrent state $S_{\text{state}} \approx 128\text{ KB}$, preventing KV memory explosion across non-full layers).*

### Mac Hardware & Metal Working-Set Fitting

Memory evaluation evaluates two distinct constraints:
1. **Physical OS RAM Fit**: $M_{\text{total}} \le M_{\text{RAM}}$
2. **Default Metal Working-Set Limit**: $M_{\text{runtime}} \le 0.75 \times M_{\text{RAM}}$

By default, macOS caps Metal allocations for a single process at ~75% of physical RAM. If runtime process memory fits in RAM but exceeds the default limit, the application flags **REQUIRES SYSCTL** (`sudo sysctl iogpu.wired_mem_limit`).

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

Run the test suite with `pytest`:

```bash
uv run pytest -v
```

Type checking with `basedpyright`:

```bash
uv run basedpyright
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
