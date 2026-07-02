"""
ComfyUI Custom Node: SageAttention SM75 for NVIDIA T4 (Turing)

Patches attention via ComfyUI's optimized_attention_override mechanism
(model.model_options["transformer_options"]["optimized_attention_override"]),
which intercepts attention at the wrap_attn decorator level — the CORRECT
approach for LTX models (they use transformer_options dispatch).

Built-in ComfyUI reference: comfy/ldm/modules/attention.py :: attention_sage()
Same reshape/mask/dtype logic, but routes to SageAttention-SM75 CUDA kernel.

Requires:
  - SageAttention-SM75 built and installed (pip install -e .)
  - NVIDIA T4 (SM75) or compatible Turing GPU
"""

import torch
import logging

logger = logging.getLogger("SageAttention")

# Cache for SageAttention import
_sageattn_fn = None
_sageattn_available = False
_sageattn_checked = False


def _check_sageattn():
    """Lazy import sageattn (top-level dispatch, auto-detects GPU arch)."""
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
        logger.warning("SageAttention not installed — node will passthrough")
    return _sageattn_fn, _sageattn_available


def _create_sage_attn_func(smooth_k=True, qk_quant_gran="per_warp"):
    """
    Creates an attention function matching ComfyUI's wrap_attn convention.

    The returned function has the same signature as the built-in
    attention_sage() — (q, k, v, heads, mask, attn_precision, ...) — and
    is called via transformer_options["optimized_attention_override"].

    Key design (matches ComfyUI attention_sage exactly):
      - FP32 → FP16 auto-conversion (LTX runs attention in float32)
      - skip_reshape: q=[B,L,H*D] → NHD / q=[B,H,L,D] → HND
      - mask → fallback to attention_pytorch (SageAttention SM75 doesn't support masks)
      - Exception → fallback to attention_pytorch
    """
    sageattn_fn, available = _check_sageattn()
    if not available:
        raise RuntimeError("SageAttention not installed")

    # Counters (reset per node execution — acceptable)
    call_count = 0
    sage_ok_count = 0
    fallback_mask_count = 0
    fallback_err_count = 0
    logged_shapes = set()

    def attention_sage_t4(q, k, v, heads, mask=None, attn_precision=None,
                          skip_reshape=False, skip_output_reshape=False, **kwargs):
        """
        SageAttention SM75 override for ComfyUI optimized_attention.

        Call chain:
          wrap_attn → override(func, q, k, v, heads, ...) → attention_sage_t4(q, k, v, heads, ...)
        """
        nonlocal call_count, sage_ok_count, fallback_mask_count, fallback_err_count
        call_count += 1

        # --- Diagnostic: log first 3 unique shapes ---
        shape_key = f"{tuple(q.shape)}_{q.dtype}"
        if shape_key not in logged_shapes and len(logged_shapes) < 3:
            logged_shapes.add(shape_key)
            logger.info(
                f"[SageAttention] call #{call_count}: "
                f"q.shape={tuple(q.shape)}, k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}, "
                f"dtype={q.dtype}, heads={heads}, mask={mask is not None}, "
                f"skip_reshape={skip_reshape}"
            )

        # --- Fallback: mask (SageAttention SM75 не поддерживает кастомные маски) ---
        if mask is not None:
            fallback_mask_count += 1
            if fallback_mask_count <= 3:
                logger.info(
                    f"[SageAttention] mask fallback #{fallback_mask_count}: "
                    f"mask.shape={tuple(mask.shape)}, dtype={mask.dtype}"
                )
            return _fallback(q, k, v, heads, mask=mask,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape,
                             attn_precision=attn_precision, **kwargs)

        # --- Save original q for fallback (unchanged) ---
        q_orig, k_orig, v_orig = q, k, v
        in_dtype = v.dtype

        # --- FP32 → FP16 (SageAttention SM75 требует FP16) ---
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q_h = q.half()
            k_h = k.half()
            v_h = v.half()
        else:
            q_h, k_h, v_h = q, k, v

        # --- Весь вызов SageAttention под единым try/except ---
        # (reshape + kernel могут упасть на cross-attention с разными Lq/Lk)
        try:
            if skip_reshape:
                # Input: [B, H, L, D] — heads already split (HND layout)
                b, h, l, dim_head = q_h.shape
                tensor_layout = "HND"
                # В HND формате reshape не нужен — передаём как есть
            else:
                # Input: [B, L, H*D] → [B, L, H, D] (NHD)
                b, dim_head = q_h.shape[0], q_h.shape[-1]
                dim_head //= heads
                # Q: [B, Lq, H*D] → [B, Lq, H, D] — l_q берём из q
                q_h = q_h.view(b, -1, heads, dim_head)
                l = q_h.shape[1]  # seq_len Q для output reshape
                # K/V: [B, Lk, H*D] → [B, Lk, H, D] — -1 для разной длины
                k_h = k_h.view(b, -1, heads, dim_head)
                v_h = v_h.view(b, -1, heads, dim_head)
                tensor_layout = "NHD"

            result = sageattn_fn(
                q_h.contiguous(), k_h.contiguous(), v_h.contiguous(),
                tensor_layout=tensor_layout,
                is_causal=False,
                smooth_k=smooth_k,
                qk_quant_gran=qk_quant_gran,
            )
            sage_ok_count += 1

            # Convert back to original dtype (crucial for FP32 models)
            result = result.to(dtype=in_dtype)

            # --- Reshape output back ---
            if tensor_layout == "HND":
                if not skip_output_reshape:
                    result = result.transpose(1, 2).reshape(b, l, heads * dim_head)
            else:
                if skip_output_reshape:
                    result = result.transpose(1, 2)
                else:
                    result = result.reshape(b, l, heads * dim_head)

            # Periodic progress log
            if sage_ok_count % 100 == 0:
                logger.info(
                    f"[SageAttention] progress: {sage_ok_count} OK, "
                    f"{fallback_mask_count} mask, {fallback_err_count} err, "
                    f"total: {call_count}"
                )

            return result

        except Exception as e:
            fallback_err_count += 1
            if fallback_err_count <= 3:
                logger.warning(
                    f"[SageAttention] kernel FAILED #{fallback_err_count}: {e}"
                )
            return _fallback(q_orig, k_orig, v_orig, heads, mask=mask,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape,
                             attn_precision=attn_precision, **kwargs)

    def _fallback(q, k, v, heads, mask=None,
                  skip_reshape=False, skip_output_reshape=False,
                  attn_precision=None, **kwargs):
        """Fallback to ComfyUI's attention_pytorch (standard SDPA)."""
        from comfy.ldm.modules.attention import attention_pytorch
        return attention_pytorch(
            q, k, v, heads, mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs
        )

    return attention_sage_t4


# ---------------------------------------------------------------------------
# ComfyUI Node: SageAttentionT4_Apply
# ---------------------------------------------------------------------------

class SageAttentionNode:
    """
    Applies SageAttention INT8 attention to a MODEL via
    transformer_options["optimized_attention_override"].

    This is the CORRECT mechanism for LTX models: ComfyUI's wrap_attn()
    decorator checks transformer_options for a registered override and
    calls it instead of the default attention function.

    Usage: place BEFORE the KSampler in the workflow.
    Supports: LTX Video, SD1.5, SDXL, Flux (any model using wrap_attn dispatch).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "smooth_k": ("BOOLEAN", {
                    "default": True,
                    "label_on": "smooth_k ON (recommended)",
                    "label_off": "smooth_k OFF",
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
    DESCRIPTION = "2× attention on T4 via INT8 tensor cores. Safe auto-fallback."

    def apply_patch(self, model, smooth_k=True, enable=True):
        if not enable:
            logger.debug("SageAttention: passthrough (enable=False)")
            return (model,)

        _, available = _check_sageattn()
        if not available:
            logger.warning("SageAttention not installed — passthrough")
            return (model,)

        # Create the attention function
        attn_func = _create_sage_attn_func(
            smooth_k=smooth_k,
            qk_quant_gran="per_warp"
        )

        # Apply via ComfyUI's optimized_attention_override
        # wrap_attn() checks: transformer_options["optimized_attention_override"]
        # and calls: override(func, q, k, v, heads, ...)
        m = model.clone()
        if "transformer_options" not in m.model_options:
            m.model_options["transformer_options"] = {}

        # The override receives (func, *args, **kwargs) from wrap_attn.
        # func is the original wrapped function (unused here).
        def attention_override(func, *args, **kwargs):
            return attn_func(*args, **kwargs)

        m.model_options["transformer_options"][
            "optimized_attention_override"
        ] = attention_override

        logger.info(
            f"SageAttention applied via optimized_attention_override "
            f"(smooth_k={smooth_k})"
        )
        return (m,)


class SageAttentionRemoveNode:
    """
    Removes SageAttention override from a MODEL, restoring original attention.

    Use when you need to chain SageAttention for part of a workflow
    and fall back to standard attention for the rest.
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
    DESCRIPTION = "Removes SageAttention override — restores original attention."

    def remove_patch(self, model):
        m = model.clone()
        topts = m.model_options.get("transformer_options")
        if topts and "optimized_attention_override" in topts:
            del topts["optimized_attention_override"]
            logger.info("SageAttention override removed")
        else:
            logger.warning("No SageAttention override found on model")
        return (m,)
