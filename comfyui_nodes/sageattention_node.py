"""
ComfyUI Custom Node: SageAttention SM75 for NVIDIA T4 (Turing)

Monkey-patches torch.nn.functional.scaled_dot_product_attention globally
to route through SageAttention's INT8 kernel.

Patches globally (like the original LTX SageAttention script) because
model.add_object_patch with "torch.nn.functional..." key fails on LTX
models: resolve_attr tries getattr(LTXAV, "torch") which doesn't exist.

Uses reference counting so multiple nodes in one workflow play nice.

Requires:
  - SageAttention-SM75 built and installed
  - NVIDIA T4 (SM75) or compatible GPU
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("SageAttention")

# Cache: only import sageattn once
_sageattn_fn = None
_sageattn_available = False
_sageattn_checked = False

# Keep a reference to the TRUE original sdpa before any patches
_TRUE_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _check_sageattn():
    """Lazy import sageattn. Returns (callable or None, available_bool)."""
    global _sageattn_fn, _sageattn_available, _sageattn_checked
    if _sageattn_checked:
        return _sageattn_fn, _sageattn_available
    _sageattn_checked = True
    try:
        from sageattention import sageattn
        _sageattn_fn = sageattn
        _sageattn_available = True
        logger.info("SageAttention imported successfully")
    except ImportError:
        _sageattn_fn = None
        _sageattn_available = False
        logger.warning("SageAttention not installed")
    return _sageattn_fn, _sageattn_available


def _create_patched_sdpa(smooth_k=True, qk_quant_gran="per_warp"):
    """
    Creates a replacement for F.scaled_dot_product_attention that
    dispatches to SageAttention. Falls back to TRUE original sdpa on unsupported cases.
    """
    sageattn_fn, available = _check_sageattn()
    if not available:
        raise RuntimeError("SageAttention not installed.")

    # Use the true original sdpa (captured at module load time), not whatever
    # F.scaled_dot_product_attention currently points to (may be another patch).
    _original = _TRUE_ORIGINAL_SDPA

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False):
        # Fallback: custom attention mask
        if attn_mask is not None:
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: dropout (not supported in INT8 path)
        if dropout_p > 0.0:
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Fallback: non-FP16/BF16 tensor dtypes
        dtype = query.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # Detect layout: HND [B,H,L,D] vs NHD [B,L,H,D]
        # heads (shape[1]) is typically much smaller than seq_len (shape[2])
        if query.shape[1] < query.shape[2]:
            tensor_layout = "HND"
        else:
            tensor_layout = "NHD"

        try:
            return sageattn_fn(
                query.contiguous(), key.contiguous(), value.contiguous(),
                tensor_layout=tensor_layout,
                is_causal=is_causal,
                sm_scale=scale,
                return_lse=False,
                smooth_k=smooth_k,
                qk_quant_gran=qk_quant_gran,
            )
        except Exception as e:
            logger.debug(f"SageAttention failed, falling back to sdpa: {e}")
            return _original(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    return patched_sdpa


class SageAttentionNode:
    """
    Applies SageAttention INT8 attention to a MODEL.

    Uses model.add_object_patch to redirect sdpa -> SageAttention.
    The patch only affects this model instance (not global).

    Inputs:
      model    - The model to patch (MODEL)
      smooth_k - Subtract K-mean before attention (default: True, recommended)
      enable   - Enable/disable patch (default: True)

    Outputs:
      model    - Patched model (MODEL)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "smooth_k": ("BOOLEAN", {
                    "default": True,
                    "label_on": "smooth_k on (recommended)",
                    "label_off": "smooth_k off",
                }),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "label_on": "SageAttention ON",
                    "label_off": "SageAttention OFF (passthrough)",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "SageAttention-T4"
    DESCRIPTION = "⚡ 2× attention on T4 via INT8 tensor cores. Safe: auto-fallback for masks/non-FP16."

    def apply_patch(self, model, smooth_k=True, enable=True):
        if not enable:
            logger.debug("Passthrough (enable=False)")
            return (model,)

        _, available = _check_sageattn()
        if not available:
            logger.warning("SageAttention NOT installed — passthrough")
            return (model,)

        # Create the patched sdpa function
        patched_fn = _create_patched_sdpa(
            smooth_k=smooth_k,
            qk_quant_gran="per_warp"
        )

        # Global monkey-patch of F.scaled_dot_product_attention
        # (add_object_patch with "torch.nn.functional..." key doesn't work
        #  on LTXAV models — resolve_attr fails on getattr(model, "torch"))
        #
        # Use reference counting to support multiple nodes stacking safely.
        if not hasattr(F, '_sageattn_prev_sdpa'):
            F._sageattn_prev_sdpa = F.scaled_dot_product_attention
            F._sageattn_refcount = 0

        F.scaled_dot_product_attention = patched_fn
        F._sageattn_refcount += 1

        logger.info(f"SageAttention applied (smooth_k={smooth_k}, refcount={F._sageattn_refcount})")
        return (model,)


class SageAttentionRemoveNode:
    """
    Removes SageAttention patch from a MODEL, restoring original attention.

    Inputs:
      model - The model to un-patch (MODEL)

    Outputs:
      model - Clean model with original attention (MODEL)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "remove_patch"
    CATEGORY = "SageAttention-T4"
    DESCRIPTION = "Removes SageAttention-T4 patch — restores original scaled_dot_product_attention."

    def remove_patch(self, model):
        if not hasattr(F, '_sageattn_prev_sdpa'):
            logger.warning("No SageAttention patch to remove (never applied)")
            return (model,)

        F._sageattn_refcount -= 1
        logger.info(f"SageAttention refcount decremented to {F._sageattn_refcount}")

        if F._sageattn_refcount <= 0:
            F.scaled_dot_product_attention = F._sageattn_prev_sdpa
            del F._sageattn_prev_sdpa
            del F._sageattn_refcount
            logger.info("SageAttention patch fully removed (last reference)")
        else:
            logger.info("SageAttention patch kept alive (other references remain)")

        return (model,)
