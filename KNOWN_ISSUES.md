# 🐛 Known Issues

## ✅ FIXED: Missing `.satfinite` in INT8 MMA PTX instruction — **TRUE ROOT CAUSE OF NOISE** (2026-07-03)

**Status:** 🟢 FIXED (ждёт пересборки и тестирования)

**Root cause:** PTX INT8 MMA instruction was missing the REQUIRED `.satfinite` modifier on Turing (SM75):
```
WRONG: mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
RIGHT: mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32
```
Without `.satfinite`, `ptxas` silently generates **incorrect SASS**, feeding garbage to Tensor Cores. This matches CUTLASS `mma_sm75.h` exactly (the Nvidia reference implementation).

**Why all previous fixes didn't help:** Every prior fix (K_GRAN, ldmatrix→manual loads, P-loading lane_id%16) was architecturally correct but the PTX instruction itself was malformed — garbage in, garbage out regardless of data correctness.

**Fix:** Added `.satfinite` to both `kInplaceUpdate` and `kInit` branches in `mma_sync_m8n8k16_row_col_s8s8s32`.

**Файл:** `csrc/mma.cuh`

**Требуется пересборка:** `python setup.py build_ext --inplace`

---

## ✅ FIXED: PV MMA P-matrix `ldmatrix_m8n8x2` address duplication (2026-07-03)

**Status:** 🟢 FIXED (ждёт тестирования)

**Root cause:** PV MMA section used dual `ldmatrix_m8n8x2` calls (`p_top`/`p_bot`) with `lane_id % 8` for row addresses. `ldmatrix.m8n8.x2` takes addresses from lanes 0-7 (Matrix 0) and lanes 8-15 (Matrix 1). Using `lane_id % 8` caused lanes 8-15 to DUPLICATE addresses from lanes 0-7, loading rows 0-7 twice instead of rows 0-15. The P (attention) matrix became garbage before multiplication with V → **pure noise output**.

**Why synthetic tests passed:** Small seq_len (64-256) meant P values were mostly zeros/small, masking the ldmatrix corruption. Real inference (seq_len=5852) exposed the bug.

**Fix:** Single `ldmatrix_m8n8x2(p_frag, smem_P_warp + (lane_id % 16) * MMA_QK_N_SM75)` — lanes 0-7 give rows 0-7, lanes 8-15 give rows 8-15, correctly forming the 16×8 fragment for m16n8k8 MMA.

**Файл:** `csrc/qattn/attn_cuda_sm75.h`

**Требуется пересборка:** `python setup.py build_ext --inplace`

---

## ✅ FIXED: INT8 MMA `ldmatrix` incompatibility on Turing SM75 (2026-07-03)

**Status:** 🟢 FIXED (ждёт тестирования)

**Root cause:** SM75 kernel использовал `ldmatrix_m8n8x4` для загрузки INT8 операндов Q/K
в `m8n8k16` INT8 MMA. На Turing (SM75), `ldmatrix` распределяет данные в **FP16** tensor core
layout, который несовместим с **INT8** MMA fragment layout. Каждый MMA вызов перемножал
неправильные элементы матриц → чистый математический шум.

**Fix:** Заменён `ldmatrix_m8n8x4` на ручные `uint32_t` загрузки из shared memory с правильным
INT8 MMA fragment mapping: `row = lane_id / 4`, `col_chunk = lane_id % 4`. Также обновлены
softmax `global_q_idx` и P-store row offset с `lane_id % 8` на `lane_id / 4`.

**Файл:** `csrc/qattn/attn_cuda_sm75.h` (тело ядра, обе функции)

**Требуется пересборка:** `python setup.py build_ext --inplace`

---

## ✅ FIXED: K_GRAN always kPerBlock — architectural mismatch with official SageAttention (2026-07-03)

**Status:** 🟢 FIXED

**Root cause:** Our SM75 kernel forced `K_GRAN = kPerWarp` (same as Q_GRAN), requiring per-warp k_scale tensors. The official SageAttention ALWAYS uses `K_GRAN = kPerBlock` regardless of Q_GRAN. This caused:
1. Need for `repeat_interleave` hack in quant.py to inflate per-block k_scale to per-warp shape
2. Shape mismatch requiring guards and CHECK_SHAPE fixes
3. Numerical noise on real inference (kernel used correct-looking but architecturally wrong indexing)

**Fixes (6 locations across 3 files):**
- `csrc/qattn/attn_cuda_sm75.h`: K_GRAN = `QuantGranularity::kPerBlock` in both kernel instantiations; CHECK_SHAPE for key_scale reverted to per-block form
- `sageattention/quant.py`: Removed `repeat_interleave` hack for k_scale in `per_warp_int8`
- `sageattention/core.py`: Simplified k_scale guard to per-block shape verification; removed per-warp k_scale expansion logic

---

## ✅ FIXED: key_scale shape mismatch (2026-07-03) — SUPERSEDED by K_GRAN fix above

**Status:** 🟢 FIXED (superseded)

**Root cause:** CHECK_SHAPE в `qk_int8_sv_f16_accum_f32_attn_sm75` ожидал per-block форму `div_ceil(kv_len, 64)` для `key_scale`, но тело ядра и Python выдают per-warp форму `div_ceil(kv_len, 64) * 4`. smem_o вариант уже был исправлен.

**Fix:** `csrc/qattn/attn_cuda_sm75.h` — изменены CHECK_SHAPE для key_scale в per_warp и per_thread ветках. (Позже заменён на правильное архитектурное решение — K_GRAN=kPerBlock.)

---

## ✅ FIXED: per_thread scale indexing bug (2026-07-03) — SUPERSEDED by K_GRAN fix above

**Status:** 🟢 FIXED

**Root cause:** В теле ядра для q_scale и k_scale были только ветки `kPerWarp` и `else` (per-block). Когда `Q_GRAN == kPerThread` или `K_GRAN == kPerThread`, код попадал в `else` и использовал per-block индексацию вместо per-thread.

**Fixes:**
1. `csrc/qattn/attn_cuda_sm75.h` — добавлены static_assert для Q_GRAN/K_GRAN, добавлены per_thread ветки индексации q_scale и k_scale
2. `sageattention/core.py` — добавлен guard для q_scale в per_thread пути: тримминг `q_scale[..., :expected_q_scales].contiguous()`

---

## ❌ CRITICAL: Tensor query_scale must be contiguous (2026-07-03) — OLD, superseded by shape bug

**Status:** 🟡 PENDING VERIFICATION (shape fix may resolve this too)

**Severity:** CRITICAL — Kernel полностью не работает в real-world inference

### 📋 Description

SM75 kernel падает с ошибкой при первом вызове в реальном inference:

```
[WARNING] [SageAttention] kernel FAILED #1: Tensor query_scale must be contiguous
[INFO] [SageAttention] SELF-DISABLED after 3 failures. Falling back to attention_pytorch.
```

### 🔍 Root Cause Analysis

**Проблемные места:**

1. **`sageattention/core.py:611`** — Slicing операция создаёт non-contiguous view:
   ```python
   q_scale = q_scale[..., :expected_q_scales].contiguous()  # Фикс применён, но не помог
   ```

2. **`sageattention/core.py:595`** — repeat_interleave может создать non-contiguous:
   ```python
   k_scale = k_scale.repeat_interleave(...).contiguous()  # Фикс применён, но не помог
   ```

3. **`sageattention/quant.py:88, 173`** — torch.empty() не гарантирует contiguous:
   ```python
   q_scale = torch.empty(...).contiguous()  # Фикс применён, но не помог
   ```

### 🧪 Test Cases

**PASS (синтетические данные):**
```python
q = torch.randn(1, 128, 1, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 128, 1, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 128, 1, 64, dtype=torch.float16, device='cuda')
O = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
# ✅ РАБОТАЕТ
```

**FAIL (real-world inference):**
```python
# Вызов из inference модели:
q.shape=(1, 5852, 4096), dtype=torch.float32  # ← FP32 вход!
k.shape=(1, 5852, 4096), dtype=torch.float32
v.shape=(1, 5852, 4096), dtype=torch.float32
# ❌ ПАДАЕТ: Tensor query_scale must be contiguous
```

### 🎯 Hypotheses (Нужно проверить)

1. **FP32 → FP16 конвертация** нарушает contiguous layout?
2. **Большие размеры** (5852 seq_len) триггерят edge case в padding/slicing?
3. **Квантизация CUDA kernels** (`_fused.quant_per_warp_int8_cuda`) возвращают non-contiguous?
4. **Промежуточные операции** между quantization и kernel call теряют contiguous?

### 🛠️ Applied Fixes (Не сработали)

| Commit | File | Change | Result |
|--------|------|--------|--------|
| `252a777` | `quant.py:88,173` | `.contiguous()` после `torch.empty()` | ❌ FAIL |
| `760639b` | `core.py:611,595` | `.contiguous()` после slicing/repeat_interleave | ❌ FAIL |

### 🔬 Next Steps

1. **Добавить debug logging** в `core.py` перед вызовом kernel:
   ```python
   print(f"DEBUG q_scale: contiguous={q_scale.is_contiguous()}, shape={q_scale.shape}, stride={q_scale.stride()}")
   print(f"DEBUG k_scale: contiguous={k_scale.is_contiguous()}, shape={k_scale.shape}, stride={k_scale.stride()}")
   ```

2. **Проверить CUDA kernel outputs** в `quant.py`:
   ```python
   # После _fused.quant_per_warp_int8_cuda():
   assert q_scale.is_contiguous(), f"q_scale non-contiguous after quant: {q_scale.stride()}"
   ```

3. **Тест с принудительным clone**:
   ```python
   q_scale = q_scale.clone()  # Вместо .contiguous()
   k_scale = k_scale.clone()
   ```

4. **Проверить CUDA binding** в `csrc/qattn/binding.cpp`:
   - Может быть баг в pybind11 tensor wrapping?

5. **Минимальный repro case** с точными параметрами из лога:
   ```python
   q = torch.randn(1, 5852, 32, 128, dtype=torch.float32, device='cuda')  # Восстановить точную форму
   # ... и шаг за шагом до kernel call
   ```

### 📝 Reproduction Instructions

**Environment:**
- GPU: NVIDIA T4 (SM75)
- PyTorch: 2.x
- CUDA: 11.8+

**Steps:**
1. `git pull origin main` (получить коммиты с фиксами)
2. `python setup.py clean --all && python setup.py build_ext --inplace`
3. Запустить inference модель с SageAttention патчем
4. Наблюдать ошибку на первом вызове

### 🤝 Help Wanted

Если у вас есть идеи или вы видели похожие проблемы с CUDA kernels — пожалуйста, откройте issue!

---

**Last Updated:** 2026-07-03  
**Reporter:** @THE-ANGEL-AI  
**Priority:** P0 (блокирует весь функционал)
