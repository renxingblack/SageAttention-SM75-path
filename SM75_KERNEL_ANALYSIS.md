# SM75 Kernel Root Cause Analysis
**Date:** 2026-07-03  
**Target:** NVIDIA T4 (SM75 Turing)  
**Repository:** https://github.com/THE-ANGEL-AI/SageAttention-SM75-path

---

## Executive Summary

Анализ выявил **критическую ошибку в порядке выполнения renormalization** в SM75 kernel, которая нарушает математическую корректность FlashAttention online softmax алгоритма.

---

## 🔴 CRITICAL BUG: Renormalization Timing Error

**Location:** `attn_cuda_sm75.h:232-362`

### Current (BROKEN) Flow:

```cpp
for (nk = 0; nk < NUM_N_TILES; ++nk) {          // K column tiles
    for (mq = 0; mq < NUM_M_TILES_QK; ++mq) {   // Q row tiles (0, 1)
        // 1. Compute S = Q @ K^T via m8n8k16 MMA
        mma_sync_m8n8k16(RS_accum, q_frag, k_frag);
        
        // 2. Online softmax: update m_i, l_i
        m_new = max(m_prev, max(s0, s1));
        o_scale = exp(m_prev - m_new);
        l_i[mq] *= o_scale;
        
        // 3. ❌ WRONG: Renormalize RO BEFORE PV MMA!
        if ((mq == 0 && lane_id < 16) || (mq == 1 && lane_id >= 16)) {
            RO_accum[fk*4+0] *= o_scale;  // ← Модифицирует RO
            RO_accum[fk*4+1] *= o_scale;
            RO_accum[fk*4+2] *= o_scale;
            RO_accum[fk*4+3] *= o_scale;
        }
        
        // 4. Compute P = exp(S - m_new) and write to smem_P
        p0 = exp(s0 - m_new);
        p1 = exp(s1 - m_new);
        smem_P[row][col] = p0, p1;
    }
    
    __syncwarp();
    
    // 5. ❌ WRONG: PV MMA uses ALREADY-SCALED RO!
    for (fk = 0; fk < NUM_N_V_TILES; ++fk) {
        ldmatrix(p_frag, smem_P);
        ldmatrix(v_frag, smem_V);
        mma_sync_m16n8k8(RO_accum + fk*4, p_frag, v_frag); // ← RO += P@V
    }
}
```

### Why This Is WRONG:

**FlashAttention-2 Correct Formula:**
```
O_new = exp(m_old - m_new) * O_old + P @ V
```

**What SM75 Actually Computes:**
```
RO *= exp(m_old - m_new)        // ← Step 3: модифицирует RO
RO += P @ V                      // ← Step 5: добавляет к УЖЕ масштабированному RO
```

**Result:**  
RO содержит `exp(m_old - m_new) * O_old + P @ V` ТОЛЬКО после **первой итерации nk**.  
На второй итерации:
```
RO *= exp(m_old2 - m_new2)      // ← Масштабирует (scaled_O_old + P1@V1)
RO += P2 @ V2                    // ← Добавляет новый вклад
```

Это **НЕ эквивалентно**:
```
O = exp(m_old2 - m_new2) * O_old + P2 @ V2
```

---

## Architecture Comparison

| Component | SM80 Official | SM75 Fork | Status |
|-----------|--------------|-----------|---------|
| INT8 QK MMA | m16n8k32 → 8 int32 | m8n8k16 → 2 int32 | ✓ Adapted |
| FP16 PV MMA | m16n8k16 → 4 float32 | m16n8k8 → 4 float32 | ✓ Adapted |
| Loop order | mq outer, nk inner | **nk outer, mq inner** | ❌ WRONG |
| Renormalization | Before each PV | **Before entire PV block** | ❌ WRONG |
| P buffer | No intermediate | 16×8 smem buffer | ⚠️ Different |

---

## Recommended Fix

### Option A: Restore Standard Loop Order (PREFERRED)

```cpp
// Move back to mq outer, nk inner (like official SM80)
for (int mq = 0; mq < NUM_M_TILES_QK; ++mq) {
    for (int nk = 0; nk < NUM_N_TILES; ++nk) {
        // 1. QK MMA
        mma_sync_m8n8k16(RS_accum, q_frag, k_frag);
        
        // 2. Online softmax
        m_new = max(m_prev, max(s0, s1));
        o_scale = exp(m_prev - m_new);
        
        // 3. Renormalize RO (per thread-group)
        if ((mq == 0 && lane_id < 16) || (mq == 1 && lane_id >= 16)) {
            RO_accum[fk*4+0] *= o_scale;
            // ...
        }
        
        // 4. Compute P
        p0 = exp(s0 - m_new);
        p1 = exp(s1 - m_new);
        
        // 5. ✓ CORRECT: Immediate PV MMA after renorm
        for (fk = 0; fk < NUM_N_V_TILES; ++fk) {
            // Load V for this (mq, nk, fk) tile
            half* v_ptr = smem_V + (k_start_warp + nk*8 + lane_id%8) * HEAD_DIM_PADDED + fk*8;
            ldmatrix_trans(v_frag, v_ptr);
            
            // Accumulate: RO += P @ V
            // P is in registers (p0, p1), V is in v_frag
            mma_sync_m16n8k8(RO_accum + fk*4, p_frag_from_p0_p1, v_frag);
        }
        
        l_i[mq] += l_local;  // Update denominator
    }
}
```

**Pros:**
- ✓ Математически корректно
- ✓ Минимальные изменения в остальном коде
- ✓ Не требует smem_P buffer

**Cons:**
- ⚠️ Нужно формировать p_frag из (p0, p1) для m16n8k8
- ⚠️ m16n8k8 требует A=2 regs, но каждый mq даёт только 1/2 данных

---

### Option B: Fix Renormalization in Current Structure

```cpp
for (int nk = 0; nk < NUM_N_TILES; ++nk) {
    // BEFORE mq loop: save old m, d
    float m_old[NUM_M_TILES_QK];
    #pragma unroll
    for (int i = 0; i < NUM_M_TILES_QK; ++i) {
        m_old[i] = m_i[i];
    }
    
    // Compute QK and update m, d
    for (int mq = 0; mq < NUM_M_TILES_QK; ++mq) {
        // QK MMA + softmax update
        mma_sync_m8n8k16(...);
        m_new = max(m_old[mq], ...);
        m_i[mq] = m_new;
        // Write P to smem
        smem_P[...] = exp(s - m_new);
    }
    
    __syncwarp();
    
    // AFTER mq loop: renormalize using m_old vs m_new
    #pragma unroll
    for (int mq = 0; mq < NUM_M_TILES_QK; ++mq) {
        float o_scale = exp(m_old[mq] - m_i[mq]);
        if ((mq == 0 && lane_id < 16) || (mq == 1 && lane_id >= 16)) {
            #pragma unroll
            for (int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                RO_accum[fk*4+0] *= o_scale;
                RO_accum[fk*4+1] *= o_scale;
                RO_accum[fk*4+2] *= o_scale;
                RO_accum[fk*4+3] *= o_scale;
            }
        }
    }
    
    // NOW do PV MMA
    for (int fk = 0; fk < NUM_N_V_TILES; ++fk) {
        ldmatrix(p_frag, smem_P);
        ldmatrix(v_frag, smem_V);
        mma_sync_m16n8k8(RO_accum + fk*4, p_frag, v_frag);
    }
}
```

**Pros:**
- ✓ Минимальные изменения в структуре
- ✓ Сохраняет smem_P подход

**Cons:**
- ⚠️ Требует сохранения m_old перед mq loop
- ⚠️ Renormalization всё ещё per-thread-group (сложнее читать)

---

## Testing Strategy

### 1. Unit Test: m8n8k16 INT8 MMA
```python
import torch
from sageattention import _qattn_sm75

Q = torch.randint(-127, 127, (8, 16), dtype=torch.int8, device='cuda')
K = torch.randint(-127, 127, (8, 16), dtype=torch.int8, device='cuda')
S_ref = Q.float() @ K.T.float()

# Call kernel with single QK tile
S_kernel = test_qk_mma_sm75(Q, K)
assert torch.allclose(S_kernel, S_ref, atol=1)
```

### 2. Unit Test: m16n8k8 FP16 MMA
```python
P = torch.randn(16, 8, dtype=torch.float16, device='cuda')
V = torch.randn(8, 64, dtype=torch.float16, device='cuda')
O_ref = P @ V

O_kernel = test_pv_mma_sm75(P, V)
assert torch.allclose(O_kernel, O_ref, rtol=1e-3)
```

### 3. Integration Test: Full Attention
```python
B, H, T, D = 1, 1, 128, 64
Q = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, T, H, D, dtype=torch.float16, device='cuda')

# Reference
O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

# SageAttention SM75
from sageattention import sageattn
O_sm75 = sageattn(Q, K, V, tensor_layout="NHD", is_causal=False)

error = (O_sm75 - O_ref).abs().max().item()
print(f"Max absolute error: {error}")
# Expected: < 0.05 (5% due to INT8 quantization)
```

---

## Implementation Priority

1. **CRITICAL (Do First):** Implement Option B fix for renormalization timing
2. **HIGH:** Add unit tests for MMA primitives
3. **MEDIUM:** Verify P-tile write pattern has no conflicts
4. **LOW:** Optimize ldmatrix loads (already functional, just inefficient)

---

## Expected Outcome

After fix:
- ✓ Kernel produces numerically correct results (error < 5% vs FP16 reference)
- ✓ No NaN/Inf in outputs
- ✓ Passes all unit tests
- ✓ Ready for Kaggle T4 deployment

---

**Generated by:** Пятница AI Agent 🐰  
**Status:** Ready for implementation review
