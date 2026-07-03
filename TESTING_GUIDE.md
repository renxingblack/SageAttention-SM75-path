# SM75 Kernel Testing Guide для Kaggle T4

**Date:** 2026-07-03  
**Status:** CRITICAL FIX APPLIED - Ready for Testing

---

## 🔧 Что было исправлено

**Commit:** `df5c2c6` - fix: correct renormalization timing in SM75 kernel (CRITICAL)

**Проблема:**
- Renormalization происходила ДО PV MMA, нарушая математику FlashAttention
- Формула: `O = exp(m_old - m_new)*O_old + P@V` была некорректна

**Решение:**
- Добавлено сохранение `m_old` перед mq loop
- Renormalization перенесена ПОСЛЕ mq loop, но ДО PV MMA
- Теперь математика корректна

---

## 📦 Установка на Kaggle T4

### 1. Клонировать репозиторий
```bash
!git clone https://github.com/THE-ANGEL-AI/SageAttention-SM75-path.git
%cd SageAttention-SM75-path
!git log --oneline -5  # Проверь что df5c2c6 есть в истории
```

### 2. Установить зависимости
```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install packaging ninja
```

### 3. Собрать kernel
```bash
!python setup.py clean --all  # Очистить старые артефакты
!python setup.py build_ext --inplace
```

**Ожидаемый вывод:**
```
Detected CUDA version: 11.8
Building for compute capabilities: {'7.5'}
BF16 support disabled (Turing SM75).
...
copying build/lib.linux-x86_64-cpython-3.10/sageattention/_qattn_sm75... -> sageattention
```

### 4. Проверить импорт
```python
import torch
import sageattention
from sageattention import sageattn

print("✓ SageAttention imported successfully")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
```

**Ожидаемый вывод:**
```
✓ SageAttention imported successfully
Device: Tesla T4
Compute capability: (7, 5)
```

---

## 🧪 Unit Tests

### Test 1: Простой attention (малый размер)
```python
import torch
import torch.nn.functional as F
from sageattention import sageattn

# Setup
B, H, T, D = 1, 1, 128, 64
device = 'cuda'
dtype = torch.float16

# Generate random inputs
torch.manual_seed(42)
Q = torch.randn(B, T, H, D, dtype=dtype, device=device)
K = torch.randn(B, T, H, D, dtype=dtype, device=device)
V = torch.randn(B, T, H, D, dtype=dtype, device=device)

# Reference (FP16 PyTorch)
with torch.no_grad():
    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

# SageAttention SM75 (INT8 QK, FP16 PV)
with torch.no_grad():
    O_sm75 = sageattn(Q, K, V, tensor_layout="NHD", is_causal=False)

# Compute error
abs_error = (O_sm75 - O_ref).abs()
max_error = abs_error.max().item()
mean_error = abs_error.mean().item()

print(f"Max absolute error:  {max_error:.6f}")
print(f"Mean absolute error: {mean_error:.6f}")
print(f"Max relative error:  {(abs_error / (O_ref.abs() + 1e-6)).max().item():.6f}")

# Pass/Fail
if max_error < 0.1:  # 10% tolerance for INT8
    print("✅ TEST PASSED")
else:
    print("❌ TEST FAILED")
    print(f"   Expected max error < 0.1, got {max_error}")
```

**Ожидаемый результат:**
```
Max absolute error:  0.03xxxx
Mean absolute error: 0.00xxxx
Max relative error:  0.05xxxx
✅ TEST PASSED
```

---

### Test 2: Causal mask
```python
import torch
from sageattention import sageattn
import torch.nn.functional as F

B, H, T, D = 1, 1, 256, 64
Q = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')

# Reference
O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# SageAttention
O_sm75 = sageattn(Q, K, V, tensor_layout="NHD", is_causal=True)

error = (O_sm75 - O_ref).abs().max().item()
print(f"Causal mask error: {error:.6f}")
print("✅ PASS" if error < 0.1 else "❌ FAIL")
```

---

### Test 3: Проверка на NaN/Inf
```python
import torch
from sageattention import sageattn

B, H, T, D = 2, 4, 512, 64
Q = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')

O = sageattn(Q, K, V, tensor_layout="NHD")

has_nan = torch.isnan(O).any().item()
has_inf = torch.isinf(O).any().item()

print(f"Contains NaN: {has_nan}")
print(f"Contains Inf: {has_inf}")
print(f"Output range: [{O.min().item():.4f}, {O.max().item():.4f}]")

if not has_nan and not has_inf:
    print("✅ NO NaN/Inf detected")
else:
    print("❌ FAIL: NaN or Inf found!")
```

---

## 🐛 Debugging Tips

### Если kernel не компилируется:
```bash
# Проверить CUDA версию
!nvcc --version

# Проверить PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Пересобрать с verbose
!python setup.py build_ext --inplace --verbose
```

### Если kernel вызывает runtime error:
```python
# Включить debug mode
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Проверить размеры тензоров
print(f"Q shape: {Q.shape}, dtype: {Q.dtype}, device: {Q.device}")
print(f"K shape: {K.shape}, dtype: {K.dtype}, device: {K.device}")
print(f"V shape: {V.shape}, dtype: {V.dtype}, device: {V.device}")
```

### Если результаты неверные (error > 10%):
```python
# Проверить квантизацию
from sageattention.quant import per_warp_int8

Q_int8, q_scale = per_warp_int8(Q)
print(f"Q scale range: [{q_scale.min():.6f}, {q_scale.max():.6f}]")

# Проверить каждую голову отдельно
for h in range(H):
    O_ref_h = F.scaled_dot_product_attention(Q[:,:,h:h+1,:], K[:,:,h:h+1,:], V[:,:,h:h+1,:])
    O_sm75_h = sageattn(Q[:,:,h:h+1,:], K[:,:,h:h+1,:], V[:,:,h:h+1,:], tensor_layout="NHD")
    error_h = (O_sm75_h - O_ref_h).abs().max().item()
    print(f"Head {h}: error = {error_h:.6f}")
```

---

## 📊 Ожидаемые результаты

| Test | Expected Max Error | Status |
|------|-------------------|--------|
| Simple attention (T=128) | < 5% | Should PASS |
| Causal mask (T=256) | < 5% | Should PASS |
| Large batch (B=2, H=4, T=512) | < 10% | Should PASS |
| No NaN/Inf | 0 | Should PASS |

---

## 📝 Reporting Results

После тестирования, пожалуйста, сообщи:

1. **Компиляция:** ✅ Success / ❌ Failed
2. **Test 1 (simple):** max_error = ???
3. **Test 2 (causal):** max_error = ???
4. **Test 3 (NaN/Inf):** ✅ Clean / ❌ Found
5. **Любые error messages или warnings**

---

## 🔄 Если тесты не прошли

1. Проверь commit hash: `git log --oneline -1` должен показать `df5c2c6`
2. Убедись что GPU = T4 (SM75): `nvidia-smi`
3. Пришли полный лог компиляции и runtime errors
4. Я проведу дополнительную диагностику

---

**Generated by:** Пятница AI Agent 🐰  
**Ready for:** Kaggle T4 GPU Testing  
**Critical Fix:** Renormalization timing corrected
