# 🐛 Known Issues

## ✅ FIXED: Per-row normalization — PV output / LSE row mapping mismatch (2026-07-03)

**Status:** 🟢 FIXED (ждёт пересборки и тестирования)

**Root cause:** С INT8 MMA groupOf=4 лейаутом, каждый поток (lane) имеет `m_i`/`l_i` для своей Q-строки (`lane_id/4`). Но PV MMA output маппит каждый поток на **2** выходные строки: `2*(lane_id/4)` и `2*(lane_id/4)+1`. Старая нормализация использовала ОДИН `l_rcp` (от Q-строки `lane_id/4`) для обеих строк → строка `2*(lane_id/4)+1` нормализовалась чужим знаменателем.

**Fix (3 changes in `attn_cuda_sm75.h`):**

| Секция | Было | Стало |
|---|---|---|
| **Ренормализация** (в K-loop) | Один `o_scale` для всех 4 RO-аккумуляторов | `__shfl_sync` достаёт `m_old`/`m_i` для каждой из 2 выходных строк → `o_scale_0`/`o_scale_1` |
| **Финальная норма** | Один `l_rcp = (lane_id < 16) ? 1/l_i[0] : 1/l_i[1]` | `__shfl_sync` достаёт `l_i` для каждой строки → `l_rcp_0`/`l_rcp_1` |
| **LSE** | `lane_id < 8` + `lane_id` для row offset | `lane_id % 4 == 0` + `lane_id / 4` (один поток на Q-строку) |

**Ключевая формула:** `src_lane = (row % 8) * 4` — маппит выходную строку на первый поток её groupOf=4 группы.

**Файл:** `csrc/qattn/attn_cuda_sm75.h`

**Требуется пересборка:** `python setup.py build_ext --inplace`

---

## ✅ FIXED: `shfl_xor` reduction distances `{8,16}` → `{1,2}` — warp-level softmax (2026-07-03)

**Status:** 🟢 FIXED (ждёт пересборки и тестирования)

**Root cause:** Онлайн-softmax использовал `__shfl_xor_sync` с дистанциями `{8, 16}` для warp-level редукции `m_local` (max) и `l_local` (sum). При groupOf=4 лейауте INT8 MMA, 4 потока обрабатывают одну Q-строку. Дистанции `{8, 16}` смешивали данные **между разными Q-строками** без редукции внутри строки → каждый поток в группе получал разное `m_local`/`l_local` → некорректный softmax → шум.

**Fix:** Заменено на `{1, 2}` — butterfly reduction внутри группы из 4 потоков (одна Q-строка). Все 4 потока получают идентичный max/sum для своей строки. Разные Q-строки получают разные значения — корректно для row-wise softmax.

**Почему `{1,2}` а не `{1,2,4,8,16}`:** Полная warp редукция дала бы всем 32 потокам одинаковый max/sum → per-sub-tile softmax вместо per-row → математически некорректно для attention (каждая Q-строка нормализуется своим знаменателем).

**Файл:** `csrc/qattn/attn_cuda_sm75.h`

**Требуется пересборка:** `python setup.py build_ext --inplace`

---

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

**Last Updated:** 2026-07-03  
**Reporter:** @THE-ANGEL-AI  
**Priority:** P0 (блокирует весь функционал)
