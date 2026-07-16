# HiPrune Colab Visualizer

A single-cell, Colab-native visualizer for [HiPrune](https://github.com/Danielement321/HiPrune) — paste `hiprune_colab.py` into one Google Colab cell, pick a model and an image, and see exactly which visual tokens the model kept and which it pruned, alongside the baseline vs. pruned responses and full statistics.

## What is HiPrune?

HiPrune ([paper](https://arxiv.org/abs/2508.00553), [original repo](https://github.com/Danielement321/HiPrune)) is a **training-free, model-agnostic visual token pruning method** for vision-language models. VLMs turn an image into hundreds or thousands of visual tokens, and LLM inference cost grows with that count — but most tokens are redundant. HiPrune exploits the *hierarchical attention* inside the vision encoder itself to decide which tokens matter, with no retraining and no extra model:

- **Anchor tokens** — the tokens with the highest attention in a *middle* encoder layer, which is object-centric. These sit on the main subject of the image.
- **Buffer tokens** — the immediate spatial neighbors (left/right/above/below) of each anchor, kept to preserve local context around objects.
- **Register tokens** — the highest-attention tokens in a *deep* encoder layer, which is global. Deep ViT layers repurpose low-information patches (sky, water, padding) as global aggregation slots, so these carry summarized scene-level context even though they often look "empty".

Everything else is pruned before the tokens reach the LLM. The paper reports up to 99.3% of task accuracy with only 33.3% of tokens (and 99.5% with just 11.1%), while cutting inference FLOPs and latency by up to 9x.

## What this cell does

1. Installs the right dependencies and downloads the **authors' unmodified HiPrune code** (pinned to a commit) — pruning runs through the original repo code path, not a re-implementation.
2. Runs a **baseline** generation (retention 1.0, provably keeps all tokens) and a **pruned** generation at your chosen retention ratio.
3. Shows the original image next to an **overlay** coloring every patch: anchor (orange), buffer (red), register (teal), pruned (dimmed to black).
4. Prints **model metadata** (params, layers, dtype, device), **both responses**, **token statistics** (counts, percentages, and mean attention per category), and **latency** (prefill / decode, measured with CUDA synchronization after a warm-up run).

The token categorization used for the overlay was differential-tested against a verbatim port of the authors' selection code (500 randomized trials per family, exact mask identity), so the overlay shows precisely the tokens the model actually kept.

## Supported models

| Dropdown option | Checkpoint | Visual tokens | transformers |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | dynamic (native resolution, capped by `MAX_PIXELS`) | 4.52.0 |
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` | always 576 (square-pad to 336×336, 24×24 grid) | 4.37.2 |

Note the different image pipelines: Qwen processes the image near its native size, while LLaVA always square-pads with the CLIP mean color and resizes to 336×336 — the gray padding bands in LLaVA overlays are real tokens the model sees and prunes.

## How to use

1. Open [Google Colab](https://colab.research.google.com) and switch to a **GPU runtime** (Runtime > Change runtime type). A T4 is fine for Qwen; LLaVA-1.5-7B is tight on a T4 (~13.5 GB of weights) — prefer an L4 or A100 if you hit out-of-memory.
2. Copy the entire contents of `hiprune_colab.py` into **one cell**.
3. Set the parameters in the form panel (or edit the top of the cell):
   - `MODEL_FAMILY` — Qwen2.5-VL or LLaVA-1.5-7B.
   - `IMAGE_PATH` — path to an image you uploaded to `/content` (falls back to the repo's demo image if the path doesn't exist).
   - `PROMPT` — the question to ask about the image.
   - `RETENTION_RATIO` — fraction of visual tokens to keep (paper default ≈ 0.22 for 128/576 tokens on LLaVA).
   - `ALPHA`, `OBJECT_LAYER` — HiPrune hyperparameters; leave `OBJECT_LAYER` at 0 for the paper default per family (Qwen: 16, LLaVA: 9).
4. Run the cell. First run downloads the model weights; re-runs with a new image / prompt / ratio are fast because the model stays cached.

### Switching between models

Qwen and LLaVA need **incompatible `transformers` versions**, so switching families mid-session requires one runtime restart. The cell handles it: it installs the correct version, prints a `RESTART REQUIRED` banner, and stops. Do Runtime > Restart session, run the cell again, and it proceeds directly. Re-runs within the same family never need a restart. If you compare both models often, the easiest setup is two separate Colab notebooks, one per family.

## Reading the output

- **Overlay** — orange/red clusters should sit on the main subject (object-centric middle layer); teal register tokens scatter over background regions by design, since deep layers use low-information patches as global context slots. This is expected, not a bug.
- **Latency table** — prefill includes the vision encoder plus the first token; decode throughput (tok/s) is the fair comparison when the two runs generate different numbers of tokens. Absolute numbers are slower than the paper's because Colab GPUs run eager attention (no FlashAttention-2 wheels).

## Credits

- HiPrune: [Danielement321/HiPrune](https://github.com/Danielement321/HiPrune) — *"HiPrune: Training-Free Visual Token Pruning via Hierarchical Attention in Vision-Language Models"* ([arXiv:2508.00553](https://arxiv.org/abs/2508.00553)). All pruning logic in this cell runs through the authors' original code, pinned to commit `8278100`.
- Models: [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) (Alibaba) and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) (Liu et al.).
