# Plan: Fix SM75 CUDA Kernel `key_scale shape error`

## 1. Контекст окружения

### 1.1. Модель
- **LTX 2.3 22B** (distilled v1.1) — DiT-based audio-video foundation model
- **Архитектура:** 48 uniform `LTX2VideoTransformerBlock`
  - **Video stream:** 32 heads × head_dim=128 → inner_dim=4096
  - **Audio stream:** 32 heads × head_dim=64 → inner_dim=2048
  - **Cross-attention:** text (3840), A2V (2048→4096), V2A (4096→2048)
  - Все Q,K,V — отдельные `nn.Linear` (no fused layers)
- **Формат:** GGUF UD (Unsloth Dynamic 2.0) — per-layer mixed-precision quant
  - Q5_K_M = ~18 GB, Q4_K_M = ~14 GB, Q3_K_XL = ~12.5 GB
  - **GGUF влияет только на веса, НЕ на тензорные shapes attention**
- **Загрузка:** `ComfyUI-GGUF` (city96) → `GGMLOps()` → on-the-fly dequant

### 1.2. Железо
- **2 × NVIDIA T4 15 GB** (Kaggle), dual GPU, без P2P
- Коммуникация карт через PyTorch **forward hook** (cross-card hidden state transfer)
- SageAttention запускается на **каждой карте независимо** для своей половины DiT блоков

### 1.3. Интеграция SageAttention
- Нода `LTX2_MultiGPU_SageAttention` → `core/sage_attention.py`
- API: `patcher.model_options["attention_patch"]["default"] = _sage_attn_wrapper`
- Внутри: `sageattn.sageattn(q, k, v)` (тензоры уже на целевой GPU)
- SM75 dispatch: → `sageattn_qk_int8_pv_fp16_cuda_sm75()` → `per_warp_int8` quant → SM75 kernel

### 1.4. Тензорные shapes (attention level)
```
Video self-attention:  Q (B, 32, S, 128), K (B, 32, S, 128), V (B, 32, S, 128)
Video cross-attn text: Q (B, 32, S, 128), K (B, 32, T, 128), V (B, 32, T, 128)
Video cross-attn A2V:  Q (B, 32, S, 128), K (B, 32, S2, 64), V (B, 32, S2, 64)  ← mixed head_dim!
Audio self-attention:  Q (B, 32, S, 64),  K (B, 32, S, 64),  V (B, 32, S, 64)
Audio cross-attn V2A:  Q (B, 32, S, 64),  K (B, 32, S2, 128), V (B, 32, S2, 128) ← mixed head_dim!
Text cross-attn:       Q (B, 32, S, 128/64), K (B, 32, 3840, 128/64), V (B, 32, 3840, 128/64)
```
- S = qo_len (зависит от разрешения: 768×512 → ~4000-6000 токенов)
- T = text_len (~77-256 для промпта)
- **num_qo_heads=32, num_kv_heads=32** (MHA, не GQA) — в стандартной конфигурации
- **SageAttention НЕ поддерживает head_dim=64** — только head_dim=128? Нужно проверить kernel.

### 1.5. num_kv_heads=8 — откуда?

Оригинальная ошибка показывает `num_kv_heads=8`. Возможные причины:
1. **GQA в одном из cross-attention слоёв** — если QKV проекции используют grouped-query (возможно в ComfyUI patch)
2. **Audio attention слои с head_dim=64** — ошибка была зафиксирована именно при обработке аудио? Нужно проверить.
3. **ComfyUI-LTX2-MultiGPU модифицирует attention** — multi-GPU split может менять attention config

> **Вывод:** фикс должен быть универсальным для любых `num_qo_heads` и `num_kv_heads`.

---

## 2. Root Cause

### 2.1. Проблема
SM75 CUDA kernel (`_qattn_sm75`) падает с `key_scale shape error` на каждом вызове
при использовании SageAttention с LTX 2.3 на T4.

### 2.2. Трассировка

**Python `per_warp_int8` (quant.py:105-180)** генерирует `k_scale` с **per-block** размерностью:
```python
k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), dtype=torch.float32)
# Для kv_len=5852, BLKK=64: shape = (b, h_kv, 92)
```

**SM75 kernel assertion (attn_cuda_sm75.h:709-710)** ожидает **per-warp** размерность:
```cpp
CHECK_SHAPE(key_scale, batch_size, num_kv_heads,
    static_cast<long>(div_ceil(kv_len, CTA_K_SM75) * (CTA_K_SM75 / WARP_K_SM75)));
// CTA_K_SM75=64, WARP_K_SM75=16 → ceil(kv_len/64) * 4
// Для kv_len=5852: shape = (b, h_kv, 368)
```

**Мисматч:** `(b, h_kv, 92)` vs `(b, h_kv, 368)` — **4x разница** (CTA_K/WARP_K = 4).

### 2.3. Почему Q scale работает, а K — нет?

`per_warp_int8` генерирует **Q scale** с per-warp размерностью:
```python
q_scale = torch.empty((b, h_qo, ceil(qo_len/BLKQ) * (BLKQ/WARPQ)), ...)
# BLKQ=64, WARPQ=16 → ceil(qo_len/64) * 4 — совпадает с ожиданием kernel
```

Но для **K scale** та же функция использует **per-block** quant (`quant_per_block_int8_cuda`),
который производит только 1 scale на блок CTA, а kernel ожидает 4 scale (по одному на warp).

### 2.4. Почему kernel обращается за пределы массива?

В ядре (attn_cuda_sm75.h:214-217):
```cpp
uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * warps_per_cta_k;
// = ceil(5852/64) * 4 = 92 * 4 = 368
uint32_t k_scale_idx = batch_id * num_kv_heads * num_warp_block_k
    + kv_head_id * num_warp_block_k
    + k_tile_idx * warps_per_cta_k + warp_idx_k;
// k_scale_idx ∈ [0, b * h_kv * 368)
// Но key_scale имеет b * h_kv * 92 элементов → OUT OF BOUNDS
```

---

## 3. Fix Options

### 3.1. Option 1: Minimal Python fix — expand per-block → per-warp

**Суть:** Добавить `repeat_interleave` в Python после генерации `k_scale`.

**Где менять:** `sageattention/quant.py` — `per_warp_int8()`

**Изменения:**
```python
def per_warp_int8(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=16, tensor_layout="HND"):
    # ... существующий код ...
    
    # Per-block K scale (как сейчас)
    k_scale_block = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), ...)
    _fused.quant_per_block_int8_cuda(k, k_int8, k_scale_block, BLKK, _tensor_layout)
    
    # Expand to per-warp: ceil(kv_len/64) → ceil(kv_len/64) * 4
    k_scale = k_scale_block.repeat_interleave(BLKK // WARPK, dim=-1)
    
    return q_int8, q_scale, k_int8, k_scale
```

**Плюсы:**
- Минимум изменений (2-3 строки Python)
- Не требует перекомпиляции CUDA
- Математически корректен: все warp'ы в одном K-блоке используют
  один и тот же per-block scale (разумное приближение)
- Работает для любых `num_kv_heads` и `head_dim`

**Минусы:**
- Потеря точности K quant (per-block вместо per-warp)
- Не фиксирует inconsistency в API

### 3.2. Option 2: Per-warp K quant CUDA kernel

**Суть:** Создать новый fused CUDA kernel `quant_per_warp_k_int8_cuda`,
который генерирует per-warp scale для K (4 scale на блок).

**Где менять:**
- `csrc/fused/fused.cu` — новый kernel + C++ функция
- `csrc/fused/pybind.cpp` — экспорт
- `sageattention/quant.py` — новый wrapper
- `sageattention/core.py` — вызов нового quant

**Плюсы:**
- Полная точность per-warp quant для K
- Правильное semantic соответствие

**Минусы:**
- ~100+ строк CUDA кода
- Требует перекомпиляции
- Сложнее отладка

### 3.3. Option 3: Kernel change — accept per-block K scale

**Суть:** Модифицировать assertion и индексное выражение в ядре для per-block K scale.

**Варианты:**
- **3a:** Изменить assertion и индекс (один `QK_QUANT_GRAN` для Q и K)
- **3b:** Разделить `QK_QUANT_GRAN` на `Q_QUANT_GRAN` и `K_QUANT_GRAN`

**Плюсы:**
- Позволяет разную гранулярность для Q и K
- Без overhead на repeat_interleave

**Минусы:**
- Модификация CUDA ядра
- Требует перекомпиляции

---

## 4. Дополнительные проблемы

### 4.1. Cross-attention reshape (УЖЕ ИСПРАВЛЕНО в 5466664)
K может иметь **другой seq_len** чем Q (например, text: 77 токенов vs video: 5852).
Фикс: `view(b, -1, ...)` вместо `view(b, h_kv, ...)` в reshape.

### 4.2. FP32→FP16 конверсия (SM75 limitation)
SM75 kernel требует **`torch.float16`**. Входные тензоры могут быть `bfloat16`.
Конверсия уже есть в `core.py:494-498`, но добавлет overhead.

### 4.3. SageAttention self-disable (УЖЕ РЕАЛИЗОВАНО в 99476bc)
После 3 kernel failures → fallback на `attention_pytorch` без попыток.
Это предотвращает ~20s/it overhead, но и не даёт ускорения.

### 4.4. Audio attention (head_dim=64)
LTX 2.3 audio stream использует head_dim=64. SM75 kernel должен быть протестирован
с head_dim=64. Текущие параметры `CTA_Q=64, CTA_K=64, WARP_Q=16, WARP_K=16` работают
с head_dim=64 (divisible by 16 for INT8 MMA).

### 4.5. Multi-GPU: SageAttention на обеих картах
При split стратегии `blocks_50_50`, SageAttention запускается на cuda:0 (блоки 0-21)
и cuda:1 (блоки 22-43). **Каждая карта вызывает `sageattn.sageattn()` независимо** —
фикс должен работать на обеих картах.

---

## 5. Рекомендация

### Шаг 1: Option 1 (минимальный Python fix) — высший приоритет
Сделано: `per_warp_int8` теперь расширяет `k_scale` до per-warp layout через `repeat_interleave`, а `core.py` передаёт `WARPK=16`.

### Шаг 2: Option 2 (per-warp K CUDA kernel) — после верификации Шага 1
Когда базовый функционал заработает — можно улучшить точность.

### Шаг 3: Тестирование
После фикса — запустить LTX 2.3 inference:
1. Нет `key_scale shape error`
2. Качество генерации не ухудшилось (визуально)
3. Скорость: target ~60s/it (без overhead на failed sageattn)
4. Проверить обе карты (cuda:0 и cuda:1) в multi-GPU режиме
5. Проверить все типы attention: video self-attn, cross-attn text, cross-attn A2V, audio

---

## 6. Связанные константы (kernel → quant)

| Kernel constant | Value | Quant parameter | Must match? |
|----------------|-------|----------------|-------------|
| CTA_Q_SM75 | 64 | BLKQ | ✅ Да (core.py: `blkq_quant=64`) |
| WARP_Q_SM75 | 16 | WARPQ | ✅ Да (core.py: `warpq_quant=16`) |
| CTA_K_SM75 | 64 | BLKK | ✅ Да (core.py: `blkk_quant=64`) |
| WARP_K_SM75 | 16 | WARPK | ❌ **Нет параметра** — добавить |
| HEAD_DIM | 128/64 | — | ✅ Kernel должен поддерживать оба |

---

## 7. Структура патча

```
SageAttention-SM75-path/
├── sageattention/
│   ├── quant.py          # DONE: +WARPK parameter, +repeat_interleave для k_scale
│   └── core.py           # DONE: передать WARPK=16 в per_warp_int8
└── PLAN.md               # этот файл (обновлён)
```

### Детали изменений в `quant.py`:

```python
def per_warp_int8(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=16, tensor_layout="HND"):
    # ... unchanged up to k_scale allocation ...
    
    # Было (per-block):
    # k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), ...)
    
    # Стало (per-block + expand to per-warp):
    k_scale_block = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK),
                                 device=q.device, dtype=torch.float32)
    _fused.quant_per_block_int8_cuda(k, k_int8, k_scale_block, BLKK, _tensor_layout)
    k_scale = k_scale_block.repeat_interleave(BLKK // WARPK, dim=-1)
    # k_scale shape: (b, h_kv, ceil(kv_len/64) * 4)
    
    return q_int8, q_scale, k_int8, k_scale
```

### Детали изменений в `core.py`:

```python
# В sageattn_qk_int8_pv_fp16_cuda_sm75() [DONE]:
q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
    q, k_to_quant, km=km_padded,
    tensor_layout=tensor_layout,
    BLKQ=blkq_quant,     # 64
    WARPQ=warpq_quant,    # 16
    BLKK=blkk_quant,      # 64
    WARPK=16,             # NEW: WARP_K_SM75
)
```

---

## 8. Верификация

### Проверка assertion после фикса:
```python
# kv_len=5852, BLKK=64, WARPK=16, num_kv_heads=8, batch=1
k_scale_block.shape  # → (1, 8, 92)
k_scale.shape        # → (1, 8, 368)  ← Теперь совпадает с kernel expectation
```

### Проверка `num_kv_heads` вариаций:
- MHA: num_kv_heads=32 → k_scale shape `(1, 32, 368)` ✅
- GQA 4:1: num_kv_heads=8 → k_scale shape `(1, 8, 368)` ✅
- GQA 2:1: num_kv_heads=16 → k_scale shape `(1, 16, 368)` ✅
- head_dim=64: kv_len может быть другим, но формула та же ✅

---

## 9. Анализ WhatDreamsCost-ComfyUI (`patches.py`)

### 9.1. Источник
Репозиторий: https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI
Файл: `patches.py` — основной механизм пачинга attention для PromptRelay/LTX Director.

### 9.2. Ключевая функция: `_masked_attention`

```python
def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    # Bypass wrap_attn (sage/etc may ignore masks) by calling attention_pytorch directly.
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )
```

**Это намеренный обход SageAttention** для masked attention. WhatDreamsCost знает, что SageAttention (и другие fused kernels) скипают/игнорируют маски, и корректно обходит их.

### 9.3. Цепочка вызовов для LTX

1. **`apply_patches(model_clone, arch="ltx", mask_fn)`** (строка 210+):
   - Сохраняет `mask_fn` в `transformer_options["promptrelay_mask_fn"]`
   - Пачит `attn2` (text→video cross-attention) и `audio_attn2` (audio cross-attention) каждого блока

2. **`_make_ltx_mask_wrapper(underlying, mask_fn, attr)`** (строка 120+):
   - Вычисляет PromptRelay маску через `mask_fn`
   - Если маска есть → устанавливает `optimized_attention_override = _make_masked_override(prev)`
   - Вызывает underlying forward

3. **`_make_masked_override(prev_override)`** (строка 90+):
   - Если `mask is not None` → вызывает `attention_pytorch` напрямую
   - Если `mask is None` → цепочка к `prev_override` (или оригинальному `func`)
   - **Маска = None → проваливается к SageAttention**

### 9.4. Какие attention вызывают SageAttention

| Attention тип | Пачится WhatDreamsCost? | Маска? | SageAttention вызывается? |
|---|---|---|---|
| **Self-attention (attn1)** — video, audio | ❌ Нет | ❌ Нет | **✅ ДА** |
| **Cross-attention (attn2)** — только глобальный промпт | ✅ Да | ❌ Нет (mask_fn→None) | **✅ ДА** |
| **Cross-attention (attn2)** — несколько сегментов | ✅ Да | ✅ Да (PromptRelay маска) | **❌ НЕТ** (обходится) |
| **Cross-attention audio (audio_attn2)** — без сегментов | ✅ Да | ❌ Нет | **✅ ДА** |
| **Cross-attention audio (audio_attn2)** — с сегментами | ✅ Да | ✅ Да | **❌ НЕТ** |

### 9.5. Что это значит для нашего фикса

1. **Self-attention (attn1) НЕ пачится WhatDreamsCost** — SageAttention будет вызываться всегда для self-attention. Это самый частый тип attention в DiT блоках. Фикс критически важен.

2. **Cross-attention без PromptRelay сегментов** — тоже идёт через SageAttention.

3. **masked cross-attention обходится намеренно** — это не баг, а фича. SageAttention не умеет маски, и это ок.

4. **num_kv_heads=8** в оригинальной ошибке — скорее всего от GQA в одном из cross-attention слоёв, где Q имеет 32 heads, а KV — 8. ComfyUI может делать repeat KV heads перед вызовом attention.

### 9.6. Проверка self-attention dispatch

Дополнительно нужно убедиться, что **attn1** (self-attention) LTX блока тоже использует `optimized_attention` (а не другой механизм). Это надо проверить в исходниках ComfyUI-LTXVideo. Если attn1 вызывает `optimized_attention` → SageAttention перехватит.

---

## 10. Дополнительные шаги (после базового фикса)

### 10.1. Проверить attn1 dispatch в ComfyUI-LTXVideo
- Найти файл attention блока LTXVideo в ComfyUI кастом-нодах
- Убедиться, что self-attention вызывает `optimized_attention`
- Если нет — добавить пач для attn1

### 10.2. Graceful fallback при наличии mask
- В `sageattn_qk_int8_pv_fp16_cuda_sm75()`: если `is_causal=False` и есть подозрение на маску — не крашиться, а fallback на `attention_pytorch`
- Использовать `IS_CAUSAL` флаг как индикатор

### 10.3. Профилирование: сколько вызовов реально идёт через Sage
- Добавить счётчик успешных/обойдённых вызовов SageAttention
- Убедиться, что хотя бы 50% attention вызовов проходят через Sage

### 10.4. Тестирование с WhatDreamsCost
- LTX Director с одним глобальным промптом (без сегментов) → Sage должен работать
- LTX Director с несколькими сегментами → masked attention обходится (ожидаемо)
- Retake Mode → проверить что Sage не мешает
