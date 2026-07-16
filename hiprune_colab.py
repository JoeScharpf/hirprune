# ============================================================================
#  HiPrune on Qwen2.5-VL / LLaVA-1.5 -- single-cell Google Colab visualizer
# ============================================================================
#  Paste this entire file into ONE Colab cell and run it (GPU runtime).
#
#  What it does:
#    1. Installs the selected family's dependencies and fetches the authors'
#       unmodified HiPrune code (pinned to a commit) -- pruning runs through
#       the ORIGINAL repo code path for both families.
#    2. Runs a baseline generation (retention 1.0, provably keeps all tokens)
#       and a pruned generation at your chosen retention ratio.
#    3. Shows the original image next to an overlay of anchor / buffer /
#       register / pruned patches.
#    4. Prints model metadata, both responses, token / attention statistics,
#       and prefill / decode latency.
#
#  IMPORTANT -- switching between Qwen and LLaVA requires a runtime restart:
#  they need incompatible transformers versions (4.52.0 vs 4.37.2). The cell
#  detects this, installs the right version, and tells you to restart.
#  Re-running with a new image / prompt / ratio in the SAME family is fast.
# ============================================================================

# ------------------------------- Parameters --------------------------------
MODEL_FAMILY = "Qwen2.5-VL-3B-Instruct"  # @param ["Qwen2.5-VL-3B-Instruct", "LLaVA-1.5-7B"]
IMAGE_PATH = "/content/demo.jpg"  # @param {type:"string"}
PROMPT = "Describe this image in detail."  # @param {type:"string"}
RETENTION_RATIO = 0.223  # @param {type:"slider", min:0.05, max:1.0, step:0.005}
ALPHA = 0.1  # @param {type:"number"}
# 0 = paper default for the family (Qwen: 16, LLaVA: 9, per the README table)
OBJECT_LAYER = 0  # @param {type:"integer"}
MAX_NEW_TOKENS = 256  # @param {type:"integer"}
# Qwen only (LLaVA always uses a fixed 336x336 input = 576 tokens).
# Caps vision tokens; eager attention memory grows quadratically with the
# vision sequence, so keep this modest on a T4 (16 GB). 1280*28*28 pixels
# = at most ~1280 merged visual tokens (~1.7 GB peak attention per block).
MAX_PIXELS = 1280 * 28 * 28  # @param {type:"raw"}

# ------------------------- Family configuration ----------------------------
IS_QWEN = MODEL_FAMILY.startswith("Qwen")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" if IS_QWEN else "liuhaotian/llava-v1.5-7b"
REQUIRED_TRANSFORMERS = "4.52.0" if IS_QWEN else "4.37.2"
if OBJECT_LAYER <= 0:
    OBJECT_LAYER = 16 if IS_QWEN else 9

HIPRUNE_COMMIT = "82781005a7e72a6be9ede58fd77473efa72b5e4f"
HIPRUNE_REPO_URL = "https://github.com/Danielement321/HiPrune.git"
QWEN_FILE_URL = (
    "https://raw.githubusercontent.com/Danielement321/HiPrune/"
    f"{HIPRUNE_COMMIT}/Qwen2_5_VL/qwen2_5_vl_HiPrune.py"
)
DEMO_IMAGE_URL = (
    "https://raw.githubusercontent.com/Danielement321/HiPrune/"
    f"{HIPRUNE_COMMIT}/assets/surf.webp"
)

# ------------------- Version gate + one-time setup -------------------------
import importlib.metadata
import os
import subprocess
import sys
import urllib.request


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    _tf_installed = importlib.metadata.version("transformers")
except importlib.metadata.PackageNotFoundError:
    _tf_installed = None

if _tf_installed != REQUIRED_TRANSFORMERS:
    print(f"Installing dependencies for {MODEL_FAMILY} "
          f"(transformers {_tf_installed} -> {REQUIRED_TRANSFORMERS})...")
    if IS_QWEN:
        _pip(f"transformers=={REQUIRED_TRANSFORMERS}", "accelerate", "qwen_vl_utils")
    else:
        # accelerate 0.27.2 is contemporary with transformers 4.37.2 and has
        # Python 3.12 support; sentencepiece/protobuf for the slow tokenizer.
        _pip(f"transformers=={REQUIRED_TRANSFORMERS}", "accelerate==0.27.2",
             "sentencepiece", "protobuf")
    if "transformers" in sys.modules:
        # The old version is already imported into this Python process and
        # cannot be swapped in place.
        print("\n" + "!" * 78)
        print("!!  RESTART REQUIRED: the correct transformers version is now installed,")
        print("!!  but the old one is already loaded in this session.")
        print("!!  Go to  Runtime > Restart session,  then run this cell again.")
        print("!" * 78)
        raise SystemExit("Restart the runtime, then re-run this cell.")

# Family code fetch (both are pinned to the same commit).
if IS_QWEN:
    if not os.path.exists("/content/qwen2_5_vl_HiPrune.py"):
        urllib.request.urlretrieve(QWEN_FILE_URL, "/content/qwen2_5_vl_HiPrune.py")
    if "/content" not in sys.path:
        sys.path.insert(0, "/content")
    try:
        import qwen_vl_utils  # noqa: F401  (missing if runtime was recycled)
    except ImportError:
        _pip("qwen_vl_utils")
else:
    if not os.path.isdir("/content/HiPrune/LLaVA/llava"):
        print("Cloning HiPrune repo (pinned)...")
        subprocess.run(["git", "clone", "--quiet", HIPRUNE_REPO_URL, "/content/HiPrune"], check=True)
        subprocess.run(["git", "-C", "/content/HiPrune", "checkout", "--quiet", HIPRUNE_COMMIT], check=True)
    # The llava package is pure Python -- importable straight from the source
    # tree, no pip install needed.
    if "/content/HiPrune/LLaVA" not in sys.path:
        sys.path.insert(0, "/content/HiPrune/LLaVA")

# HiPrune hyperparameters. Qwen reads these at IMPORT time (module constants,
# re-applied to the module below); LLaVA reads them at CALL time inside
# encode_images, so the env vars alone are authoritative there.
os.environ["HIPRUNE_OBJECT_LAYER"] = str(OBJECT_LAYER)
os.environ["HIPRUNE_ALPHA"] = str(ALPHA)
os.environ["HIPRUNE_QWEN_RETENTION"] = str(RETENTION_RATIO)
os.environ["HIPRUNE_RETENTION"] = "576"  # LLaVA token budget; overridden per run

# ----------------------------- Common imports ------------------------------
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

assert torch.cuda.is_available(), "No GPU found -- switch to a GPU runtime (Runtime > Change runtime type)."

if not IS_QWEN:
    # Compat shim: transformers 4.37.2 predates modern torch and may reference
    # the removed private pytree API. Alias it back if this torch dropped it.
    import torch.utils._pytree as _pytree

    if not hasattr(_pytree, "_register_pytree_node") and hasattr(_pytree, "register_pytree_node"):
        _pytree._register_pytree_node = _pytree.register_pytree_node

from transformers import StoppingCriteria, StoppingCriteriaList


# ----------------------------- Load model (cached) --------------------------
def _needs_model_reload():
    cached = globals().get("_HIPRUNE_MODEL")
    if cached is None or globals().get("_HIPRUNE_MODEL_ID") != MODEL_ID:
        return True
    # An earlier run may have cached a model whose weights got stranded on
    # CPU (transformers 4.52.x device_map + tied-weights bug). Detect that
    # and force a clean reload instead of reusing the broken one.
    return any(p.device.type != "cuda" for p in cached.parameters())


if _needs_model_reload():
    import gc

    # Drop any stale cached model first so its GPU memory is freed before the
    # new copy loads. "model" from a previous run aliases the same object.
    globals().pop("_HIPRUNE_MODEL", None)
    globals().pop("model", None)
    gc.collect()
    torch.cuda.empty_cache()

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if not IS_QWEN and gpu_gb < 20:
        print(f"WARNING: LLaVA-1.5-7B needs ~13.5 GB for fp16 weights; this GPU has "
              f"{gpu_gb:.1f} GB. It may fit, but if you hit CUDA out-of-memory, switch "
              f"to an L4 or A100 runtime (Runtime > Change runtime type).")

    if IS_QWEN:
        import qwen2_5_vl_HiPrune as hp
        from transformers import AutoProcessor

        print(f"Loading {MODEL_ID} (fp16, eager attention)...")
        # NOTE: no device_map here on purpose. transformers 4.52.x has a known
        # bug where accelerate's device_map dispatch leaves tied weights (the
        # 3B model ties embed_tokens/lm_head) stranded on CPU, crashing the
        # first embedding lookup. A plain .to("cuda") sidesteps accelerate.
        _HIPRUNE_MODEL = hp.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # vision blocks assert against sdpa; flash-attn unavailable on Colab
        ).to("cuda").eval()
        _HIPRUNE_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _HIPRUNE_TOKENIZER = None
        _HIPRUNE_IMAGE_PROCESSOR = None
    else:
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        print(f"Loading {MODEL_ID} (fp16, sdpa attention)... this downloads ~13.5 GB on first run.")
        # device_map="auto" loads shards directly onto the GPU -- Colab's
        # ~12 GB of system RAM cannot hold the full fp16 checkpoint.
        _HIPRUNE_TOKENIZER, _HIPRUNE_MODEL, _HIPRUNE_IMAGE_PROCESSOR, _ = load_pretrained_model(
            MODEL_ID, None, get_model_name_from_path(MODEL_ID),
            device_map="auto", attn_implementation="sdpa",
        )
        _HIPRUNE_MODEL.eval()
        _HIPRUNE_PROCESSOR = None
    _HIPRUNE_MODEL_ID = MODEL_ID

model = _HIPRUNE_MODEL

if IS_QWEN:
    import qwen2_5_vl_HiPrune as hp

    processor = _HIPRUNE_PROCESSOR
    # Re-apply parameters: the module's env vars are only read once, at import.
    hp.OBJECT_LAYER = OBJECT_LAYER
    hp.ALPHA = ALPHA

    # --- cache_position fix for eager attention ----------------------------
    # The repo was only run with flash_attention_2, where the causal mask is
    # skipped entirely. Under eager attention, the pruned prefill passes a
    # cache_position of the ORIGINAL sequence length alongside pruned
    # inputs_embeds, which breaks the 4D causal-mask construction. This
    # wrapper renumbers cache_position to match the pruned length. It changes
    # no HiPrune math -- token selection and pruning are the authors' code.
    if not getattr(hp.Qwen2_5_VLModel, "_hiprune_colab_patched", False):
        _orig_lm_forward = hp.Qwen2_5_VLModel.forward

        def _patched_lm_forward(self, *args, **kwargs):
            cp = kwargs.get("cache_position")
            ie = kwargs.get("inputs_embeds")
            if cp is not None and ie is not None and cp.shape[0] != ie.shape[1]:
                kwargs["cache_position"] = torch.arange(ie.shape[1], device=ie.device)
            return _orig_lm_forward(self, *args, **kwargs)

        hp.Qwen2_5_VLModel.forward = _patched_lm_forward
        hp.Qwen2_5_VLModel._hiprune_colab_patched = True
else:
    tokenizer = _HIPRUNE_TOKENIZER
    image_processor = _HIPRUNE_IMAGE_PROCESSOR

# ------------------------------- Load image --------------------------------
if not os.path.exists(IMAGE_PATH):
    print(f"'{IMAGE_PATH}' not found -- downloading the repo demo image instead.")
    IMAGE_PATH = "/content/hiprune_demo.webp"
    if not os.path.exists(IMAGE_PATH):
        urllib.request.urlretrieve(DEMO_IMAGE_URL, IMAGE_PATH)

pil_image = ImageOps.exif_transpose(Image.open(IMAGE_PATH).convert("RGB"))

# --------------------- Family-specific input preparation -------------------
if IS_QWEN:
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image, "max_pixels": int(MAX_PIXELS)},
            {"type": "text", "text": PROMPT},
        ],
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    grid_thw = inputs["image_grid_thw"]           # (1, 3) = [t, h_patches14, w_patches14]
    merge = model.config.vision_config.spatial_merge_size  # 2
    grid_h = int(grid_thw[0][1]) // merge         # merged-token grid height
    grid_w = int(grid_thw[0][2]) // merge         # merged-token grid width
    n_tokens = grid_h * grid_w                    # merged visual tokens seen by the LLM
    CELL = 14 * merge                             # px per token on the resized image
    resized_size = (int(grid_thw[0][2]) * 14, int(grid_thw[0][1]) * 14)  # (W, H) px
else:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    # 'pad' aspect ratio: square-pad with the CLIP mean color, then 336x336.
    images_tensor = process_images([pil_image], image_processor, model.config).to("cuda", dtype=torch.float16)
    image_sizes = [pil_image.size]

    grid_h = grid_w = 24                          # CLIP-L/336: 24x24 patches
    n_tokens = 576
    CELL = 14                                     # px per token on the 336px input
    merge = 1
    resized_size = (336, 336)

# ------------------------------- Generation --------------------------------
class _StepTimer(StoppingCriteria):
    """Records a timestamp after every generated token (never stops generation).

    The first timestamp marks the end of prefill (vision encoder + full-sequence
    forward + first token); the gaps after that are pure decode steps.
    """

    def __init__(self):
        self.stamps = []

    def __call__(self, input_ids, scores, **kwargs):
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())
        return False


def hiprune_generate(retain_ratio, max_new_tokens=None):
    """Run generation through the original HiPrune forward at a given ratio.

    Returns (response_text, timing dict with prefill/decode/total seconds).
    """
    if IS_QWEN:
        hp.RETAIN = float(retain_ratio)  # module-level constant read inside forward
    else:
        # LLaVA reads the budget as an absolute token COUNT at call time.
        os.environ["HIPRUNE_RETENTION"] = str(max(1, round(n_tokens * retain_ratio)))

    timer = _StepTimer()
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.inference_mode():
        if IS_QWEN:
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
            new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        else:
            # The fork's generate returns (ids, v_token_num, cls_attn), and the
            # ids contain only GENERATED tokens (prompt goes in as embeddings).
            new_ids, _, _ = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    n_new = new_ids.shape[1]
    prefill_s = (timer.stamps[0] if timer.stamps else t_end) - t_start
    decode_s = t_end - timer.stamps[0] if timer.stamps else 0.0
    timing = {
        "total_s": t_end - t_start,
        "prefill_s": prefill_s,           # includes vision encoder + 1st token
        "decode_s": decode_s,
        "n_new_tokens": n_new,
        # first token belongs to prefill, so decode throughput uses n_new - 1
        "decode_tps": (n_new - 1) / decode_s if decode_s > 0 and n_new > 1 else float("nan"),
    }
    torch.cuda.empty_cache()
    if IS_QWEN:
        text = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    else:
        text = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return text, timing


# Warm-up: first CUDA run pays one-time kernel/allocator costs; do a 1-token
# generation first so the baseline vs. pruned timing comparison is fair.
print("\nWarm-up run...")
hiprune_generate(1.0, max_new_tokens=1)

print(f"Running baseline (retention 1.0, all {n_tokens} visual tokens kept)...")
baseline_response, baseline_t = hiprune_generate(1.0)
print(f"Running HiPrune  (retention {RETENTION_RATIO})...")
pruned_response, pruned_t = hiprune_generate(RETENTION_RATIO)

# --------------------- Token categorization (for the plot) ------------------
# One vision-tower pass for the per-layer attention scores, then the exact
# selection arithmetic from the authors' code (Qwen model file lines
# 1856-1884 / LLaVA llava_arch.py lines 140-174), kept separated into
# anchor / buffer / register instead of merged, so we can color them.
# NOTE: selection must run in the attention's native dtype (fp16) with the
# repo's exact ops -- upcasting to float32 changes topk tie-breaking after
# the `- mask` subtraction and yields a slightly different token set than
# the model actually pruned (verified with a randomized differential test).
if IS_QWEN:
    with torch.inference_mode():
        _, attn_list = model.visual(
            inputs["pixel_values"].type(model.visual.dtype), grid_thw=grid_thw
        )
    shallow_attention = attn_list[OBJECT_LAYER - 1]
    deep_attention_src = attn_list[-1]
    DEEP_LABEL = "deep layer (last)"
else:
    with torch.inference_mode():
        _, vt_attentions = model.get_model().get_vision_tower()(images_tensor)
    sel_layer = model.config.mm_vision_select_layer  # -2 for LLaVA-1.5
    # Replicates encode_images: mean over heads, mean over queries, drop CLS.
    shallow_attention = vt_attentions[OBJECT_LAYER - 1].mean(dim=1).mean(dim=1)[0, 1:]
    deep_attention_src = vt_attentions[sel_layer].mean(dim=1).mean(dim=1)[0, 1:]
    DEEP_LABEL = f"deep layer ({sel_layer})"

deep_attention = deep_attention_src.clone()  # the repo mutates this in-place

visual_token_num = round(n_tokens * RETENTION_RATIO)
shallow_token_num = round((visual_token_num * ALPHA) / 5)

anchor_idx = torch.topk(shallow_attention, k=shallow_token_num).indices
shallow_all = torch.cat([anchor_idx,
                         anchor_idx - 1,
                         anchor_idx + 1,
                         anchor_idx - grid_w,
                         anchor_idx + grid_w])
shallow_all = shallow_all.clamp(0, n_tokens - 1)
shallow_all = torch.unique(shallow_all, sorted=False)
buffer_idx = shallow_all[~torch.isin(shallow_all, anchor_idx)]

deep_token_num = visual_token_num - shallow_all.shape[0]
selected_mask = torch.zeros(n_tokens, dtype=torch.bool, device=deep_attention.device)
selected_mask.scatter_(0, shallow_all, 1)
deep_attention -= selected_mask.int()
register_idx = torch.topk(deep_attention, k=deep_token_num).indices

kept_mask = selected_mask.clone()
kept_mask[register_idx] = True
pruned_idx = torch.nonzero(~kept_mask, as_tuple=True)[0]
kept_idx = torch.nonzero(kept_mask, as_tuple=True)[0]

# restore unmutated attention scores for the stats section
deep_attention = deep_attention_src.float()
shallow_attention = shallow_attention.float()

# -------------------------------- Overlay ----------------------------------
COLORS = {"anchor": "#f3a361", "buffer": "#e66d50", "register": "#299d8f"}

if IS_QWEN:
    overlay_base = pil_image.resize(resized_size, Image.BICUBIC)
else:
    # Replicate the model's preprocessing: square-pad with the CLIP mean
    # color, then resize to 336x336 (what the vision encoder actually sees).
    from llava.mm_utils import expand2square

    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    overlay_base = expand2square(pil_image, bg).resize(resized_size, Image.BICUBIC)

overlay = np.array(overlay_base).astype(np.float32)


def _cells(indices):
    for idx in indices.tolist():
        r, c = idx // grid_w, idx % grid_w
        yield slice(r * CELL, (r + 1) * CELL), slice(c * CELL, (c + 1) * CELL)


for rs, cs in _cells(pruned_idx):        # dim pruned patches to near-black
    overlay[rs, cs] *= 0.15
for name, indices in (("anchor", anchor_idx), ("buffer", buffer_idx), ("register", register_idx)):
    tint = np.array([int(COLORS[name][i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    for rs, cs in _cells(indices):       # tint kept patches by category
        overlay[rs, cs] = 0.45 * overlay[rs, cs] + 0.55 * tint

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(pil_image)
axes[0].set_title(f"Original ({pil_image.width}x{pil_image.height})")
axes[1].imshow(overlay.clip(0, 255).astype(np.uint8))
axes[1].set_title(
    f"{MODEL_FAMILY} + HiPrune @ retention {RETENTION_RATIO:.3f} "
    f"({len(kept_idx)}/{n_tokens} tokens kept, grid {grid_w}x{grid_h})"
)
for ax in axes:
    ax.axis("off")
axes[1].legend(
    handles=[mpatches.Patch(color=c, label=n.capitalize()) for n, c in COLORS.items()]
    + [mpatches.Patch(color="black", label="Pruned")],
    loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False,
)
plt.tight_layout()
plt.show()

# --------------------------------- Stats -----------------------------------
n_kept, n_pruned = len(kept_idx), len(pruned_idx)
uniform = 1.0 / n_tokens  # attention mass each token would get if uniform


def attn_row(name, indices):
    if len(indices) == 0:
        return f"  {name:<10}      0     0.0%            -              -"
    obj = shallow_attention[indices].mean().item()
    deep = deep_attention[indices].mean().item()
    return (f"  {name:<10} {len(indices):>6} {len(indices) / n_tokens:>7.1%}"
            f"   {obj:.3e} ({obj / uniform:>5.2f}x)"
            f"   {deep:.3e} ({deep / uniform:>5.2f}x)")


import transformers

txt_cfg = model.config
n_params_total = sum(p.numel() for p in model.parameters())
if IS_QWEN:
    vision_module = model.visual
    vis_cfg = model.config.vision_config
    vis_desc = (f"{vis_cfg.depth} layers, hidden {vis_cfg.hidden_size}, "
                f"{vis_cfg.num_heads} heads, patch {vis_cfg.patch_size}px, "
                f"merge {vis_cfg.spatial_merge_size}x{vis_cfg.spatial_merge_size}, "
                f"full-attn blocks {vis_cfg.fullatt_block_indexes}")
    deep_layer_desc = f"deep layer = {vis_cfg.depth} (last)"
else:
    vision_module = model.get_vision_tower()
    vis_cfg = vision_module.config
    vis_desc = (f"CLIP ViT-L: {vis_cfg.num_hidden_layers} layers, "
                f"hidden {vis_cfg.hidden_size}, {vis_cfg.num_attention_heads} heads, "
                f"patch {vis_cfg.patch_size}px, input {vis_cfg.image_size}px")
    deep_layer_desc = f"deep layer = {sel_layer} (mm_vision_select_layer)"
n_params_vision = sum(p.numel() for p in vision_module.parameters())
n_params_lm = n_params_total - n_params_vision

print("\n" + "=" * 78)
print("MODEL INFO")
print("=" * 78)
print(f"  Model                   : {MODEL_ID}")
print(f"  Total parameters        : {n_params_total / 1e9:.2f}B"
      f"  (LLM {n_params_lm / 1e9:.2f}B + vision {n_params_vision / 1e6:.0f}M)")
print(f"  LLM                     : {txt_cfg.num_hidden_layers} layers, "
      f"hidden {txt_cfg.hidden_size}, {txt_cfg.num_attention_heads} heads "
      f"({getattr(txt_cfg, 'num_key_value_heads', txt_cfg.num_attention_heads)} KV heads), "
      f"vocab {txt_cfg.vocab_size}")
print(f"  Vision encoder          : {vis_desc}")
print(f"  Tied embeddings         : {getattr(txt_cfg, 'tie_word_embeddings', False)}")
print(f"  Dtype / attention       : {model.dtype} / {txt_cfg._attn_implementation}")
print(f"  Device                  : {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
print(f"  Versions                : transformers {transformers.__version__}, torch {torch.__version__}")
print(f"  HiPrune settings        : retention {RETENTION_RATIO}, alpha {ALPHA}, "
      f"object layer {OBJECT_LAYER}, {deep_layer_desc}")

print("\n" + "=" * 78)
print("RESPONSES")
print("=" * 78)
print(f"[Baseline | {n_tokens} visual tokens]\n{baseline_response}\n")
print(f"[HiPrune  | {n_kept} visual tokens, retention {RETENTION_RATIO:.3f}]\n{pruned_response}")

print("\n" + "=" * 78)
print("TOKEN STATISTICS")
print("=" * 78)
if IS_QWEN:
    print(f"  Image resized to        : {resized_size[0]}x{resized_size[1]} px")
    print(f"  Merged token grid       : {grid_w} x {grid_h}  (each token = {CELL}x{CELL} px)")
    print(f"  Total visual tokens     : {n_tokens}  ({n_tokens * merge**2} raw 14px ViT patches)")
else:
    print(f"  Image preprocessed to   : square pad + {resized_size[0]}x{resized_size[1]} px")
    print(f"  Token grid              : {grid_w} x {grid_h}  (each token = {CELL}x{CELL} px)")
    print(f"  Total visual tokens     : {n_tokens}")
print(f"  Kept                    : {n_kept}  ({n_kept / n_tokens:.1%})")
print(f"  Pruned                  : {n_pruned}  ({n_pruned / n_tokens:.1%})")
print(f"    Anchor   tokens       : {len(anchor_idx)}  "
      f"({len(anchor_idx) / n_tokens:.1%} of total, {len(anchor_idx) / max(n_kept, 1):.1%} of kept)")
print(f"    Buffer   tokens       : {len(buffer_idx)}  "
      f"({len(buffer_idx) / n_tokens:.1%} of total, {len(buffer_idx) / max(n_kept, 1):.1%} of kept)")
print(f"    Register tokens       : {len(register_idx)}  "
      f"({len(register_idx) / n_tokens:.1%} of total, {len(register_idx) / max(n_kept, 1):.1%} of kept)")

print("\n" + "=" * 78)
print("MEAN ATTENTION PER CATEGORY   (multiplier vs. uniform = 1/N in parens)")
print("=" * 78)
print(f"  {'category':<10} {'count':>6} {'% total':>8}   {'object layer ' + str(OBJECT_LAYER):<22} {DEEP_LABEL:<20}")
print(attn_row("anchor", anchor_idx))
print(attn_row("buffer", buffer_idx))
print(attn_row("register", register_idx))
print(attn_row("kept", kept_idx))
print(attn_row("pruned", pruned_idx))

print("\n" + "=" * 78)
print("LATENCY   (prefill includes vision encoder + first token)")
print("=" * 78)


def lat_row(name, t):
    return (f"  {name:<10} {t['prefill_s']:>10.2f}s {t['decode_s']:>10.2f}s"
            f" {t['n_new_tokens']:>8} {t['decode_tps']:>12.2f} {t['total_s']:>9.2f}s")


print(f"  {'run':<10} {'prefill':>11} {'decode':>11} {'tokens':>8} {'decode tok/s':>12} {'total':>10}")
print(lat_row("Baseline", baseline_t))
print(lat_row("HiPrune", pruned_t))
if pruned_t["prefill_s"] > 0:
    print(f"\n  Prefill speedup       : {baseline_t['prefill_s'] / pruned_t['prefill_s']:.2f}x")
if baseline_t["decode_tps"] > 0 and pruned_t["decode_tps"] > 0:
    print(f"  Decode throughput gain: {pruned_t['decode_tps'] / baseline_t['decode_tps']:.2f}x")
print("  Note: total time is NOT apples-to-apples when the two runs generate")
print("  different token counts -- compare prefill and decode tok/s instead.")
print("  Absolute numbers are slower than the paper's FlashAttention-2 setup,")
print("  and the identical vision-encoder cost in both runs dilutes the gain.")
print("=" * 78)
