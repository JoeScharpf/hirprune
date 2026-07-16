# ============================================================================
#  HiPrune on Qwen2.5-VL / LLaVA-1.5 / Gemma 4 -- single-cell Colab visualizer
# ============================================================================
#  Paste this entire file into ONE Colab cell and run it (GPU runtime).
#
#  What it does:
#    1. Installs the selected family's dependencies and fetches the HiPrune
#       code. Qwen and LLaVA run the AUTHORS' unmodified code (pinned to a
#       commit). Gemma 4 is OUR PORT of the method -- the HiPrune paper never
#       tested Gemma; see the "HiPrune port for Gemma" section for the spec.
#    2. Runs a baseline generation (retention 1.0, provably keeps all tokens)
#       and a pruned generation at your chosen retention ratio.
#    3. Shows the original image next to an overlay of anchor / buffer /
#       register / pruned patches.
#    4. Prints model metadata, both responses, token / attention statistics,
#       and prefill / decode latency.
#
#  IMPORTANT -- switching families requires a runtime restart: they need
#  incompatible transformers versions (Qwen 4.52.0 / LLaVA 4.37.2 /
#  Gemma 5.14.1). The cell detects this, installs the right version, and
#  tells you to restart. Re-running within the SAME family is fast.
#
#  Gemma 4 weights are license-gated on Hugging Face: accept the license at
#  https://huggingface.co/google/gemma-4-e4b-it and store a token in Colab
#  under the key HF_TOKEN (key icon in the left sidebar > Secrets).
# ============================================================================

# ------------------------------- Parameters --------------------------------
MODEL_FAMILY = "Qwen2.5-VL-3B-Instruct"  # @param ["Qwen2.5-VL-3B-Instruct", "LLaVA-1.5-7B", "Gemma-4-E4B"]
IMAGE_PATH = "/content/demo.jpg"  # @param {type:"string"}
PROMPT = "Describe this image in detail."  # @param {type:"string"}
RETENTION_RATIO = 0.223  # @param {type:"slider", min:0.05, max:1.0, step:0.005}
ALPHA = 0.1  # @param {type:"number"}
# 0 = default for the family: Qwen 16, LLaVA 9 (paper README table),
# Gemma 8 (our choice: middle of its 16-layer encoder -- NOT paper-validated)
OBJECT_LAYER = 0  # @param {type:"integer"}
MAX_NEW_TOKENS = 256  # @param {type:"integer"}
# Qwen only (LLaVA is fixed 336x336 = 576 tokens; Gemma is capped at 280
# soft tokens by its processor). Caps vision tokens; eager attention memory
# grows quadratically with the vision sequence, so keep this modest on a T4.
MAX_PIXELS = 1280 * 28 * 28  # @param {type:"raw"}

# ------------------------- Family configuration ----------------------------
IS_QWEN = MODEL_FAMILY.startswith("Qwen")
IS_LLAVA = MODEL_FAMILY.startswith("LLaVA")
IS_GEMMA = MODEL_FAMILY.startswith("Gemma")

if IS_QWEN:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    REQUIRED_TRANSFORMERS = "4.52.0"
    DEFAULT_OBJECT_LAYER = 16
elif IS_LLAVA:
    MODEL_ID = "liuhaotian/llava-v1.5-7b"
    REQUIRED_TRANSFORMERS = "4.37.2"
    DEFAULT_OBJECT_LAYER = 9
else:
    # gemma-4-e2b-it (~10 GB) is the fallback if this doesn't fit your GPU.
    MODEL_ID = "google/gemma-4-e4b-it"
    REQUIRED_TRANSFORMERS = "5.14.1"
    DEFAULT_OBJECT_LAYER = 8

if OBJECT_LAYER <= 0:
    OBJECT_LAYER = DEFAULT_OBJECT_LAYER

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

# Each transformers version has its own accelerate requirement, and pins from
# a previous family's session survive on the same Colab VM. A bare
# "accelerate" install spec would be "already satisfied" by e.g. LLaVA's
# 0.27.2 pin, which transformers 5.x rejects at device_map time (it needs
# >= 1.1.0), so the constraint must be explicit and version-checked here.
if IS_QWEN:
    _ACCELERATE_REQ = "accelerate>=0.26.0"
elif IS_LLAVA:
    # accelerate 0.27.2 is contemporary with transformers 4.37.2 and has
    # Python 3.12 support.
    _ACCELERATE_REQ = "accelerate==0.27.2"
else:
    _ACCELERATE_REQ = "accelerate>=1.1.0"


def _accelerate_ok():
    try:
        v = importlib.metadata.version("accelerate")
    except importlib.metadata.PackageNotFoundError:
        return False
    if "==" in _ACCELERATE_REQ:
        return v == _ACCELERATE_REQ.split("==")[1]
    floor = _ACCELERATE_REQ.split(">=")[1]

    def _t(s):
        return tuple(int(p) for p in s.split(".")[:3] if p.isdigit())

    return _t(v) >= _t(floor)


if _tf_installed != REQUIRED_TRANSFORMERS or not _accelerate_ok():
    print(f"Installing dependencies for {MODEL_FAMILY} "
          f"(transformers {_tf_installed} -> {REQUIRED_TRANSFORMERS})...")
    if IS_QWEN:
        _pip(f"transformers=={REQUIRED_TRANSFORMERS}", _ACCELERATE_REQ, "qwen_vl_utils")
    elif IS_LLAVA:
        # sentencepiece/protobuf for the slow tokenizer.
        _pip(f"transformers=={REQUIRED_TRANSFORMERS}", _ACCELERATE_REQ,
             "sentencepiece", "protobuf")
    else:
        _pip(f"transformers=={REQUIRED_TRANSFORMERS}", _ACCELERATE_REQ)
    if "transformers" in sys.modules:
        # The old versions are already imported into this Python process and
        # cannot be swapped in place (transformers also caches the accelerate
        # version it saw at import time).
        print("\n" + "!" * 78)
        print("!!  RESTART REQUIRED: the correct library versions are now installed,")
        print("!!  but the old ones are already loaded in this session.")
        print("!!  Go to  Runtime > Restart session,  then run this cell again.")
        print("!" * 78)
        raise SystemExit("Restart the runtime, then re-run this cell.")

# Family code fetch (Qwen/LLaVA are pinned to the same commit; the Gemma
# branch is self-contained in this cell).
if IS_QWEN:
    if not os.path.exists("/content/qwen2_5_vl_HiPrune.py"):
        urllib.request.urlretrieve(QWEN_FILE_URL, "/content/qwen2_5_vl_HiPrune.py")
    if "/content" not in sys.path:
        sys.path.insert(0, "/content")
    try:
        import qwen_vl_utils  # noqa: F401  (missing if runtime was recycled)
    except ImportError:
        _pip("qwen_vl_utils")
elif IS_LLAVA:
    if not os.path.isdir("/content/HiPrune/LLaVA/llava"):
        print("Cloning HiPrune repo (pinned)...")
        subprocess.run(["git", "clone", "--quiet", HIPRUNE_REPO_URL, "/content/HiPrune"], check=True)
        subprocess.run(["git", "-C", "/content/HiPrune", "checkout", "--quiet", HIPRUNE_COMMIT], check=True)
    # The llava package is pure Python -- importable straight from the source
    # tree, no pip install needed.
    if "/content/HiPrune/LLaVA" not in sys.path:
        sys.path.insert(0, "/content/HiPrune/LLaVA")
else:
    # Gemma weights are gated: pick up a token from Colab secrets if present.
    if "HF_TOKEN" not in os.environ:
        try:
            from google.colab import userdata

            _tok = userdata.get("HF_TOKEN")
            if _tok:
                os.environ["HF_TOKEN"] = _tok
        except Exception:
            pass
    if "HF_TOKEN" not in os.environ:
        print("NOTE: no HF_TOKEN found in Colab secrets. Gemma weights are gated --")
        print("      if loading fails with a 401/403, accept the license at")
        print(f"      https://huggingface.co/{MODEL_ID} and add a token under")
        print("      Colab's Secrets panel (key icon) with the name HF_TOKEN.")

# HiPrune hyperparameters for the authors' code. Qwen reads these at IMPORT
# time (module constants, re-applied below); LLaVA reads them at CALL time
# inside encode_images. The Gemma port reads the Python variables directly.
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

if IS_LLAVA:
    # Compat shim: transformers 4.37.2 predates modern torch and may reference
    # the removed private pytree API. Alias it back if this torch dropped it.
    import torch.utils._pytree as _pytree

    if not hasattr(_pytree, "_register_pytree_node") and hasattr(_pytree, "register_pytree_node"):
        _pytree._register_pytree_node = _pytree.register_pytree_node

from transformers import StoppingCriteria, StoppingCriteriaList


# ------------------------ Shared HiPrune selection --------------------------
def hiprune_select(shallow_scores, deep_scores, n_toks, gw, ratio, alpha):
    """The HiPrune anchor/buffer/register selection, one budget's worth.

    This is the exact arithmetic from the authors' code (Qwen model file
    lines 1856-1884 / LLaVA llava_arch.py lines 140-174), kept separated
    into anchor / buffer / register instead of merged, so we can color them.
    Differential-tested against verbatim ports of both.

    NOTE: for Qwen/LLaVA pass scores in their native dtype (fp16) -- the
    repo's ops must run unchanged, since upcasting to float32 changes topk
    tie-breaking after the `- mask` subtraction and yields a slightly
    different token set than the model actually pruned. The Gemma port
    defines float32 scores as its spec.

    Returns (anchor_idx, buffer_idx, register_idx, kept_mask).
    """
    deep = deep_scores.clone()  # the repo mutates this in-place

    budget = round(n_toks * ratio)
    shallow_token_num = round((budget * alpha) / 5)

    anchor_idx = torch.topk(shallow_scores, k=shallow_token_num).indices
    shallow_all = torch.cat([anchor_idx,
                             anchor_idx - 1,
                             anchor_idx + 1,
                             anchor_idx - gw,
                             anchor_idx + gw])
    shallow_all = shallow_all.clamp(0, n_toks - 1)
    shallow_all = torch.unique(shallow_all, sorted=False)
    buffer_idx = shallow_all[~torch.isin(shallow_all, anchor_idx)]

    deep_token_num = budget - shallow_all.shape[0]
    selected_mask = torch.zeros(n_toks, dtype=torch.bool, device=deep.device)
    selected_mask.scatter_(0, shallow_all, 1)
    deep -= selected_mask.int()
    register_idx = torch.topk(deep, k=deep_token_num).indices

    kept_mask = selected_mask.clone()
    kept_mask[register_idx] = True
    return anchor_idx, buffer_idx, register_idx, kept_mask


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
    if IS_LLAVA and gpu_gb < 20:
        print(f"WARNING: LLaVA-1.5-7B needs ~13.5 GB for fp16 weights; this GPU has "
              f"{gpu_gb:.1f} GB. It may fit, but if you hit CUDA out-of-memory, switch "
              f"to an L4 or A100 runtime (Runtime > Change runtime type).")
    if IS_GEMMA and gpu_gb < 20:
        print(f"WARNING: Gemma-4-E4B needs ~16 GB for its weights; this GPU has "
              f"{gpu_gb:.1f} GB. Use an L4 or A100 runtime, or set "
              f"MODEL_ID = 'google/gemma-4-e2b-it' above (~10 GB).")

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
    elif IS_LLAVA:
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
    else:
        from transformers import AutoProcessor

        try:
            from transformers import Gemma4ForConditionalGeneration
        except ImportError:
            from transformers.models.gemma4 import Gemma4ForConditionalGeneration

        # bf16 on Ampere+ (matches the checkpoint); fp16 fallback on T4/Turing.
        _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading {MODEL_ID} ({_dtype})... this downloads ~16 GB on first run.")
        # device_map="cuda" streams checkpoint shards straight to the GPU,
        # so the full model never has to fit in Colab's system RAM.
        _HIPRUNE_MODEL = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=_dtype,
            device_map="cuda",
        ).eval()
        _HIPRUNE_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _HIPRUNE_TOKENIZER = None
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
elif IS_LLAVA:
    tokenizer = _HIPRUNE_TOKENIZER
    image_processor = _HIPRUNE_IMAGE_PROCESSOR
else:
    processor = _HIPRUNE_PROCESSOR

    # --- HiPrune port for Gemma: keep-mask hook -----------------------------
    # Gemma merges vision features into the LLM sequence by masked_scatter
    # over `input_ids == image_token_id` positions, and hard-checks that the
    # number of placeholder slots equals the number of features. Pruning
    # therefore happens on BOTH sides: hiprune_generate drops pruned
    # placeholder ids from input_ids, and this patch makes get_image_features
    # return only the kept soft tokens. Everything downstream (per-layer
    # embeddings, masks, positions, KV cache) runs stock -- Gemma E4B looks
    # up per-layer embeddings from input_ids, so inputs_embeds-only
    # generation would silently degrade the model.
    _GEMMA_KEEP_MASK = None  # bool tensor over soft tokens, or None = keep all

    _gemma_model_cls = type(model.model)
    if not getattr(_gemma_model_cls, "_hiprune_colab_patched", False):
        _orig_gif = _gemma_model_cls.get_image_features

        def _patched_gif(self, pixel_values, image_position_ids=None, **kwargs):
            out = _orig_gif(self, pixel_values, image_position_ids, **kwargs)
            km = globals().get("_GEMMA_KEEP_MASK")
            if km is not None:
                out.pooler_output = out.pooler_output[km.to(out.pooler_output.device)]
            return out

        _gemma_model_cls.get_image_features = _patched_gif
        _gemma_model_cls._hiprune_colab_patched = True

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
elif IS_LLAVA:
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
    resized_size = (336, 336)
else:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # The modeling code calls this kwarg image_position_ids; the vision tower
    # calls it pixel_position_ids. Detect whichever key the processor emits.
    GEMMA_POS_KEY = next(
        (k for k in ("image_position_ids", "pixel_position_ids") if k in inputs), None
    )
    assert GEMMA_POS_KEY is not None, \
        f"No patch-position key found in processor output: {list(inputs.keys())}"

    # Derive the token grid from the patch positions. The processor always
    # pads pixel_values / position ids up to the fixed budget of
    # max_soft_tokens * pooling_kernel_size^2 patches (280*9 = 2520), marking
    # padding patches with position (-1, -1) -- even for a single image. The
    # model masks those out internally; the selection pass below must too.
    _pp = inputs[GEMMA_POS_KEY][0]                # (max_patches, 2) as (x, y)
    _gemma_valid = ~((_pp == -1).all(dim=-1))     # True = real patch
    POOL_K = model.config.vision_config.pooling_kernel_size       # 3
    PATCH = model.config.vision_config.patch_size                 # 16
    patch_w = int(_pp[_gemma_valid, 0].max()) + 1
    patch_h = int(_pp[_gemma_valid, 1].max()) + 1
    grid_w = patch_w // POOL_K                    # pooled soft-token grid
    grid_h = patch_h // POOL_K
    n_tokens = grid_w * grid_h
    CELL = PATCH * POOL_K                         # 48 px per soft token
    resized_size = (patch_w * PATCH, patch_h * PATCH)

    n_placeholders = int((inputs["input_ids"][0] == model.config.image_token_id).sum())
    assert n_placeholders == n_tokens, \
        f"Soft-token grid ({n_tokens}) != image placeholders ({n_placeholders})"
    if "num_soft_tokens_per_image" in inputs:
        _n_soft = int(torch.as_tensor(inputs["num_soft_tokens_per_image"]).flatten()[0])
        assert _n_soft == n_tokens, \
            f"Derived grid ({n_tokens}) != processor num_soft_tokens_per_image ({_n_soft})"

    # Patch index -> soft token index for REAL patches only, exactly as
    # Gemma4VisionPooler._avg_pool_by_positions computes it:
    # (x//k) + (patch_w//k) * (y//k). Padding positions would map to negative
    # indices, so they are excluded here and in the scoring below.
    _gemma_kernel_idx = (
        (_pp[_gemma_valid, 0] // POOL_K) + grid_w * (_pp[_gemma_valid, 1] // POOL_K)
    )

# ------------------ HiPrune port for Gemma (selection pass) -----------------
# Unlike Qwen/LLaVA, where the authors' forward prunes internally, the Gemma
# port computes the keep mask up front: one vision pass with attention
# capture, patch scores aggregated onto soft tokens with the pooler's own
# kernel arithmetic, then the shared HiPrune selection.
if IS_GEMMA:
    vt = model.model.vision_tower

    # The selection needs explicit attention weights; force eager on the
    # vision tower for this one pass (sdpa/flash return None weights).
    _prev_attn_impl = vt.config._attn_implementation
    vt.config._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            _vt_out = vt(
                pixel_values=inputs["pixel_values"],
                pixel_position_ids=inputs[GEMMA_POS_KEY],
                output_attentions=True,
            )
    finally:
        vt.config._attn_implementation = _prev_attn_impl

    _attns = getattr(_vt_out, "attentions", None)
    assert _attns is not None and _attns[0] is not None, \
        "Vision tower returned no attention weights -- eager capture failed"

    def _gemma_soft_scores(layer_attn):
        """Per-soft-token attention score for one encoder layer.

        Patch score = mean over heads, mean over VALID queries (the 'global
        attention' variant the authors use for CLS-free encoders); soft-token
        score = SUM over the 3x3 pooling window, using the same one-hot
        weight construction as Gemma4VisionPooler._avg_pool_by_positions
        (sum instead of the pooler's mean: it keeps the scores a probability
        distribution over soft tokens -- patch scores sum to 1 -- and is
        order-equivalent for topk selection). Padding patches are excluded on
        both axes: padding keys are attention-masked by the encoder anyway,
        but padding query rows are garbage and must not be averaged in.
        Float32 throughout -- this port's spec, since there is no author
        implementation to bit-match.
        """
        attn = layer_attn[0].float().mean(dim=0)                       # (queries, keys)
        patch_scores = attn[_gemma_valid][:, _gemma_valid].mean(dim=0)  # (num_valid,)
        weights = torch.nn.functional.one_hot(
            _gemma_kernel_idx.long(), n_tokens
        ).float()
        return weights.T @ patch_scores                                # (n_tokens,)

    shallow_attention = _gemma_soft_scores(_attns[OBJECT_LAYER - 1])
    deep_attention_src = _gemma_soft_scores(_attns[-1])
    DEEP_LABEL = "deep layer (last)"

    anchor_idx, buffer_idx, register_idx, kept_mask = hiprune_select(
        shallow_attention, deep_attention_src, n_tokens, grid_w, RETENTION_RATIO, ALPHA
    )
    _GEMMA_PRUNED_MASK = kept_mask

    del _vt_out, _attns
    torch.cuda.empty_cache()

    # Per-position keys that must be sliced alongside input_ids when tokens
    # are dropped; everything else (pixel_values, position ids) passes through.
    _GEMMA_SEQ_KEYS = ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")

    def _gemma_pruned_inputs(keep_mask):
        """Drop pruned image-placeholder positions from all per-token tensors."""
        ids = inputs["input_ids"][0]
        img_pos = (ids == model.config.image_token_id).nonzero(as_tuple=True)[0]
        seq_keep = torch.ones_like(ids, dtype=torch.bool)
        seq_keep[img_pos[~keep_mask]] = False
        return {
            k: (v[:, seq_keep] if k in _GEMMA_SEQ_KEYS else v)
            for k, v in inputs.items()
        }

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
    """Run one generation at a given retention ratio.

    Qwen/LLaVA prune inside the authors' forward; Gemma prunes via the
    input_ids + get_image_features hook. Returns (response_text, timing dict).
    """
    global _GEMMA_KEEP_MASK
    if IS_QWEN:
        hp.RETAIN = float(retain_ratio)  # module-level constant read inside forward
    elif IS_LLAVA:
        # LLaVA reads the budget as an absolute token COUNT at call time.
        os.environ["HIPRUNE_RETENTION"] = str(max(1, round(n_tokens * retain_ratio)))
    else:
        if retain_ratio >= 1.0:
            _GEMMA_KEEP_MASK = None  # keep all: hook disabled, stock behavior
            gen_inputs = {k: v for k, v in inputs.items()}
        else:
            assert retain_ratio == RETENTION_RATIO, \
                "Gemma keep mask is precomputed for RETENTION_RATIO only"
            _GEMMA_KEEP_MASK = _GEMMA_PRUNED_MASK
            gen_inputs = _gemma_pruned_inputs(_GEMMA_PRUNED_MASK)

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
        elif IS_LLAVA:
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
        else:
            out_ids = model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
            new_ids = out_ids[:, gen_inputs["input_ids"].shape[1]:]
            _GEMMA_KEEP_MASK = None
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
    if IS_LLAVA:
        text = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    else:
        text = processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    return text, timing


# Warm-up: first CUDA run pays one-time kernel/allocator costs; do a 1-token
# generation first so the baseline vs. pruned timing comparison is fair.
print("\nWarm-up run...")
hiprune_generate(1.0, max_new_tokens=1)

if IS_GEMMA:
    # Equivalence check: run the FULL pruning machinery with a keep-all mask
    # (feature slicing hook active, input rebuild active, zero tokens dropped)
    # and require token-identical output vs. fully native generation. This
    # proves the port is a no-op at retention 1.0.
    with torch.inference_mode():
        _native_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    _all_true = torch.ones(n_tokens, dtype=torch.bool, device=inputs["input_ids"].device)
    _GEMMA_KEEP_MASK = _all_true
    with torch.inference_mode():
        _hooked_ids = model.generate(
            **_gemma_pruned_inputs(_all_true), max_new_tokens=8, do_sample=False
        )
    _GEMMA_KEEP_MASK = None
    _eq = torch.equal(_native_ids, _hooked_ids)
    print(f"Pruning-path equivalence check (native vs. keep-all pruned, 8 tokens): "
          f"{'PASS' if _eq else 'FAIL -- port bug, do not trust results!'}")

print(f"Running baseline (retention 1.0, all {n_tokens} visual tokens kept)...")
baseline_response, baseline_t = hiprune_generate(1.0)
print(f"Running HiPrune  (retention {RETENTION_RATIO})...")
pruned_response, pruned_t = hiprune_generate(RETENTION_RATIO)

# --------------------- Token categorization (for the plot) ------------------
# Gemma computed its selection before generation (it had to -- the keep mask
# drives pruning). Qwen/LLaVA compute it now: one vision-tower pass for the
# per-layer attention scores, then the shared hiprune_select. The pass is
# deterministic (eval, no dropout), so the scores are identical to what the
# model pruned with.
if IS_QWEN:
    with torch.inference_mode():
        _, attn_list = model.visual(
            inputs["pixel_values"].type(model.visual.dtype), grid_thw=grid_thw
        )
    shallow_attention = attn_list[OBJECT_LAYER - 1]
    deep_attention_src = attn_list[-1]
    DEEP_LABEL = "deep layer (last)"
    anchor_idx, buffer_idx, register_idx, kept_mask = hiprune_select(
        shallow_attention, deep_attention_src, n_tokens, grid_w, RETENTION_RATIO, ALPHA
    )
elif IS_LLAVA:
    with torch.inference_mode():
        _, vt_attentions = model.get_model().get_vision_tower()(images_tensor)
    sel_layer = model.config.mm_vision_select_layer  # -2 for LLaVA-1.5
    # Replicates encode_images: mean over heads, mean over queries, drop CLS.
    shallow_attention = vt_attentions[OBJECT_LAYER - 1].mean(dim=1).mean(dim=1)[0, 1:]
    deep_attention_src = vt_attentions[sel_layer].mean(dim=1).mean(dim=1)[0, 1:]
    DEEP_LABEL = f"deep layer ({sel_layer})"
    anchor_idx, buffer_idx, register_idx, kept_mask = hiprune_select(
        shallow_attention, deep_attention_src, n_tokens, grid_w, RETENTION_RATIO, ALPHA
    )
# (Gemma: anchor_idx/buffer_idx/register_idx/kept_mask already computed above.)

pruned_idx = torch.nonzero(~kept_mask, as_tuple=True)[0]
kept_idx = torch.nonzero(kept_mask, as_tuple=True)[0]

# float copies of the unmutated scores for the stats section
deep_attention = deep_attention_src.float()
shallow_attention = shallow_attention.float()

# -------------------------------- Overlay ----------------------------------
COLORS = {"anchor": "#f3a361", "buffer": "#e66d50", "register": "#299d8f"}

if IS_LLAVA:
    # Replicate the model's preprocessing: square-pad with the CLIP mean
    # color, then resize to 336x336 (what the vision encoder actually sees).
    from llava.mm_utils import expand2square

    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    overlay_base = expand2square(pil_image, bg).resize(resized_size, Image.BICUBIC)
else:
    # Qwen and Gemma both resize without padding (Qwen to 28px multiples,
    # Gemma to 48px multiples with mild aspect flooring).
    overlay_base = pil_image.resize(resized_size, Image.BICUBIC)

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

n_params_total = sum(p.numel() for p in model.parameters())
if IS_QWEN:
    txt_cfg = model.config
    vision_module = model.visual
    vis_cfg = model.config.vision_config
    vis_desc = (f"{vis_cfg.depth} layers, hidden {vis_cfg.hidden_size}, "
                f"{vis_cfg.num_heads} heads, patch {vis_cfg.patch_size}px, "
                f"merge {vis_cfg.spatial_merge_size}x{vis_cfg.spatial_merge_size}, "
                f"full-attn blocks {vis_cfg.fullatt_block_indexes}")
    deep_layer_desc = f"deep layer = {vis_cfg.depth} (last)"
elif IS_LLAVA:
    txt_cfg = model.config
    vision_module = model.get_vision_tower()
    vis_cfg = vision_module.config
    vis_desc = (f"CLIP ViT-L: {vis_cfg.num_hidden_layers} layers, "
                f"hidden {vis_cfg.hidden_size}, {vis_cfg.num_attention_heads} heads, "
                f"patch {vis_cfg.patch_size}px, input {vis_cfg.image_size}px")
    deep_layer_desc = f"deep layer = {sel_layer} (mm_vision_select_layer)"
else:
    txt_cfg = model.config.text_config
    vision_module = model.model.vision_tower
    vis_cfg = model.config.vision_config
    vis_desc = (f"{vis_cfg.num_hidden_layers} layers, hidden {vis_cfg.hidden_size}, "
                f"{vis_cfg.num_attention_heads} heads, patch {vis_cfg.patch_size}px, "
                f"pooling {POOL_K}x{POOL_K} patches/soft-token, "
                f"max {vis_cfg.default_output_length} soft tokens")
    deep_layer_desc = f"deep layer = {vis_cfg.num_hidden_layers} (last) [OUR PORT, not paper-validated]"
n_params_vision = sum(p.numel() for p in vision_module.parameters())
n_params_lm = n_params_total - n_params_vision

print("\n" + "=" * 78)
print("MODEL INFO")
print("=" * 78)
print(f"  Model                   : {MODEL_ID}")
if IS_GEMMA:
    print("  HiPrune implementation  : OUR PORT (no author code exists for Gemma)")
else:
    print("  HiPrune implementation  : authors' original code (pinned commit)")
print(f"  Total parameters        : {n_params_total / 1e9:.2f}B"
      f"  (LLM+other {n_params_lm / 1e9:.2f}B + vision {n_params_vision / 1e6:.0f}M)")
print(f"  LLM                     : {txt_cfg.num_hidden_layers} layers, "
      f"hidden {txt_cfg.hidden_size}, {txt_cfg.num_attention_heads} heads "
      f"({getattr(txt_cfg, 'num_key_value_heads', txt_cfg.num_attention_heads)} KV heads), "
      f"vocab {txt_cfg.vocab_size}")
print(f"  Vision encoder          : {vis_desc}")
print(f"  Tied embeddings         : {getattr(txt_cfg, 'tie_word_embeddings', False)}")
print(f"  Dtype / attention       : {model.dtype} / "
      f"{getattr(txt_cfg, '_attn_implementation', 'default')}")
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
_orig_px = pil_image.width * pil_image.height
print(f"  Original image          : {pil_image.width}x{pil_image.height} px  ({_orig_px / 1e6:.1f} MP)")
if IS_QWEN:
    print(f"  Image resized to        : {resized_size[0]}x{resized_size[1]} px")
    if _orig_px < MAX_PIXELS:
        # Qwen never upscales: below the cap, the image itself limits tokens.
        print(f"  NOTE: image is below the MAX_PIXELS cap ({MAX_PIXELS / 1e6:.1f} MP) -- visual token")
        print("        count is limited by the image; a larger image yields more tokens.")
    print(f"  Merged token grid       : {grid_w} x {grid_h}  (each token = {CELL}x{CELL} px)")
    print(f"  Total visual tokens     : {n_tokens}  ({n_tokens * merge**2} raw 14px ViT patches)")
elif IS_LLAVA:
    print(f"  Image preprocessed to   : square pad + {resized_size[0]}x{resized_size[1]} px")
    print(f"  Token grid              : {grid_w} x {grid_h}  (each token = {CELL}x{CELL} px)")
    print(f"  Total visual tokens     : {n_tokens}")
else:
    print(f"  Image resized to        : {resized_size[0]}x{resized_size[1]} px")
    print(f"  Soft-token grid         : {grid_w} x {grid_h}  (each = {CELL}x{CELL} px, "
          f"{POOL_K}x{POOL_K} pooled 16px patches)")
    print(f"  Total visual tokens     : {n_tokens}  ({n_tokens * POOL_K**2} raw 16px ViT patches)")
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
if IS_GEMMA:
    print("  Gemma's small soft-token count (<=280) means pruning gains are")
    print("  smaller than on models with 1000+ visual tokens.")
print("=" * 78)
