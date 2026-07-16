# HiPrune Colab Visualizer

A single-cell, Colab-native visualizer for [HiPrune](https://github.com/Danielement321/HiPrune) — paste `hiprune_colab.py` into one Google Colab cell, pick a model and an image, and see exactly which visual tokens the model kept and which it pruned, alongside the baseline vs. pruned responses and full statistics.

## What is HiPrune?

HiPrune ([paper](https://arxiv.org/abs/2508.00553), [original repo](https://github.com/Danielement321/HiPrune)) is a **training-free, model-agnostic visual token pruning method** for vision-language models. VLMs turn an image into hundreds or thousands of visual tokens, and LLM inference cost grows with that count — but most tokens are redundant. HiPrune exploits the *hierarchical attention* inside the vision encoder itself to decide which tokens matter, with no retraining and no extra model:

- **Anchor tokens** — the tokens with the highest attention in a *middle* encoder layer, which is object-centric. These sit on the main subject of the image.
- **Buffer tokens** — the immediate spatial neighbors (left/right/above/below) of each anchor, kept to preserve local context around objects.
- **Register tokens** — the highest-attention tokens in a *deep* encoder layer, which is global. Deep ViT layers repurpose low-information patches (sky, water, padding) as global aggregation slots, so these carry summarized scene-level context even though they often look "empty".

Everything else is pruned before the tokens reach the LLM. The paper reports up to 99.3% of task accuracy with only 33.3% of tokens (and 99.5% with just 11.1%), while cutting inference FLOPs and latency by up to 9x.

## What this cell does

1. Installs the right dependencies and fetches the HiPrune code. For Qwen and LLaVA, pruning runs through the **authors' unmodified code** (pinned to a commit), not a re-implementation. For Gemma 4 — which the paper never tested — the cell contains **our port** of the method (see below).
2. Runs a **baseline** generation (retention 1.0, provably keeps all tokens) and a **pruned** generation at your chosen retention ratio.
3. Shows the original image next to an **overlay** coloring every patch: anchor (orange), buffer (red), register (teal), pruned (dimmed to black).
4. Prints **model metadata** (params, layers, dtype, device), **both responses**, **token statistics** (counts, percentages, and mean attention per category), and **latency** (prefill / decode, measured with CUDA synchronization after a warm-up run).

The token categorization used for the overlay was differential-tested against a verbatim port of the authors' selection code (500 randomized trials per family, exact mask identity), so the overlay shows precisely the tokens the model actually kept.

## Supported models

| Dropdown option | Checkpoint | Visual tokens | transformers | HiPrune code |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | dynamic (native resolution, capped by `MAX_PIXELS`) | 4.52.0 | authors' original |
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` | always 576 (square-pad to 336×336, 24×24 grid) | 4.37.2 | authors' original |
| Gemma-4-E4B | `google/gemma-4-e4b-it` | up to 280 soft tokens (48×48 px each, 3×3-pooled 16 px patches) | 5.14.1 | **our port** |

Note the different image pipelines: Qwen processes the image near its native size; LLaVA always square-pads with the CLIP mean color and resizes to 336×336 — the gray padding bands in LLaVA overlays are real tokens the model sees and prunes; Gemma resizes to multiples of 48 px and average-pools every 3×3 patch window into one "soft token" before the LLM.

### The Gemma 4 branch is our port, not author code

The HiPrune paper covers LLaVA-1.5/NeXT and Qwen2.5-VL only. Our Gemma 4 port applies the same method to Gemma's vision stack: per-layer ViT attention is captured (mean over heads and queries, the same "global attention" variant the authors use for CLS-free encoders), patch scores are aggregated onto soft tokens using the pooler's own 3×3 kernel arithmetic, and the standard anchor/buffer/register selection then prunes soft tokens before the LLM (placeholder ids and vision features are pruned consistently, keeping Gemma's per-layer-embedding path intact). The aggregation and selection were differential-tested against a verbatim port of Gemma's pooler code, and at retention 1.0 the cell verifies token-identical output against fully native generation at runtime. Caveats: results are our extrapolation of the paper, and the default `OBJECT_LAYER = 8` (middle of the 16-layer encoder) is our choice, not a paper-validated hyperparameter — sweep it if anchors don't land on objects.

Gemma weights are **license-gated**: accept the license at [huggingface.co/google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it), then add a Hugging Face token in Colab's Secrets panel (key icon in the left sidebar) under the name `HF_TOKEN`.

## How to use

1. Open [Google Colab](https://colab.research.google.com) and switch to a **GPU runtime** (Runtime > Change runtime type). A T4 is fine for Qwen; LLaVA-1.5-7B (~13.5 GB) and Gemma-4-E4B (~16 GB) want an L4 or A100 (for Gemma on a T4, set `MODEL_ID` to `google/gemma-4-e2b-it`, ~10 GB).
2. Copy the entire contents of `hiprune_colab.py` into **one cell**.
3. Set the parameters in the form panel (or edit the top of the cell):
   - `MODEL_FAMILY` — Qwen2.5-VL, LLaVA-1.5-7B, or Gemma-4-E4B.
   - `IMAGE_PATH` — path to an image you uploaded to `/content` (falls back to the repo's demo image if the path doesn't exist).
   - `PROMPT` — the question to ask about the image.
   - `RETENTION_RATIO` — fraction of visual tokens to keep (paper default ≈ 0.22 for 128/576 tokens on LLaVA).
   - `ALPHA`, `OBJECT_LAYER` — HiPrune hyperparameters; leave `OBJECT_LAYER` at 0 for the per-family default (Qwen: 16, LLaVA: 9 — paper values; Gemma: 8 — our choice).
4. Run the cell. First run downloads the model weights; re-runs with a new image / prompt / ratio are fast because the model stays cached.

### Switching between models

The three families need **incompatible `transformers` versions** (4.52.0 / 4.37.2 / 5.14.1), so switching families mid-session requires one runtime restart. The cell handles it: it installs the correct version, prints a `RESTART REQUIRED` banner, and stops. Do Runtime > Restart session, run the cell again, and it proceeds directly. Re-runs within the same family never need a restart. If you compare models often, the easiest setup is separate Colab notebooks, one per family.

## Reading the output

- **Overlay** — orange/red clusters should sit on the main subject (object-centric middle layer); teal register tokens scatter over background regions by design, since deep layers use low-information patches as global context slots. This is expected, not a bug.
- **Latency table** — prefill includes the vision encoder plus the first token; decode throughput (tok/s) is the fair comparison when the two runs generate different numbers of tokens. Absolute numbers are slower than the paper's because Colab GPUs run eager attention (no FlashAttention-2 wheels).

## Credits

- HiPrune: [Danielement321/HiPrune](https://github.com/Danielement321/HiPrune) — *"HiPrune: Training-Free Visual Token Pruning via Hierarchical Attention in Vision-Language Models"* ([arXiv:2508.00553](https://arxiv.org/abs/2508.00553)). Qwen and LLaVA pruning runs through the authors' original code, pinned to commit `8278100`; the Gemma 4 branch is our port of their method.
- Models: [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) (Alibaba), [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) (Liu et al.), and [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4) (Google DeepMind).
