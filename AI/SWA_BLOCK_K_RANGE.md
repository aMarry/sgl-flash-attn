# Prefill 阶段 SWA 块级 K 范围推导

> 对应源文件：`csrc/flash_attn/src/block_info.h`（`BlockMN::get_n_block_min_max`）

---

## 一、基本设定

| 参数 | 典型值 |
|---|---|
| `kBlockM` | 128（Q 块大小，smem 行数） |
| `kBlockN` | 128（K 块大小，smem 列数） |
| `window_size_left` | 128 |
| `window_size_right` | 0（causal） |
| PackGQA qhead_per_khead | 8 |

PackGQA 下，smem 中的 128 行并不是 128 个序列 token，而是 `kBlockM / qhead_per_khead = 16` 个序列 token 乘以 `qhead_per_khead` 头数。因此：

```
m_idx = smem_row_idx / qhead_per_khead   →  序列维度 token 索引
```

---

## 二、单个 Q token 的有效 K 范围

对序列位置为 `q` 的 Q token（causal + SWA，`window_size_right=0`）：

```
有效 K 范围 = [q - window_size_left, q + 1)
            = [q - 128, q + 1)
            共 129 个 token（自身 + 左边 128 个）
```

> **注意**：窗口大小是 **129**，不是 128。`window_size_left=128` 表示向左延伸 128 个位置，加上自身 = 129。

---

## 三、块级 K 范围（`get_n_block_min_max`）

当 Q block = 128 时，block 内包含多个序列 token，各 token 的窗口边界不同。  
代码取 **block 内所有 token 需求的并集**：

### 3.1 n_block_max（右边界）

用 Q block 内**最大** token 位置（`m_idx_max`）决定：

```cpp
int m_idx_max = (m_block + 1) * kBlockM;       // smem 行号上界
// PackGQA:
m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1;
// = (m_block * 16) + 16  （第几个序列 token）

int n_idx_right = m_idx_max + seqlen_k - seqlen_q;   // causal: seqlen_k == seqlen_q 则 = m_idx_max
n_block_max = min(total_k_blocks, ceil_div(n_idx_right, kBlockN));
```

### 3.2 n_block_min（左边界）

用 Q block 内**最小** token 位置（`m_idx_min`）决定：

```cpp
int m_idx_min = m_block * kBlockM;
// PackGQA:
m_idx_min = qhead_per_khead_divmod.divide(m_idx_min);
// = m_block * 16

int n_idx_left = m_idx_min + seqlen_k - seqlen_q - window_size_left;
// = m_block*16 - 128

n_block_min = max(0, n_idx_left / kBlockN);
```

### 3.3 示例（seqlen_q = seqlen_k = 10240，PackGQA=8）

每个 m_block 对应序列中的 **16 个 token**（smem 的 128 行 / 8 头）。

| m_block | Q token 范围 | m_idx_min | m_idx_max | n_block_min | n_block_max | K blocks 数 |
|:-------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|:-----------:|
| 0       | [0, 16)      | 0         | 16        | 0           | 1           | 1           |
| 8       | [128, 144)   | 128       | 144       | 0           | 2           | 2           |
| 16      | [256, 272)   | 256       | 272       | 1           | 3           | 2           |
| 100     | [1600, 1616) | 1600      | 1616      | 11          | 13          | 2           |

稳态下每个 m_block 只需处理 **2 个 K block**（因为 `window_size_left = kBlockN = 128`，窗口宽度恰好横跨两个 K 块边界）。

---

## 四、三阶段循环：精细 mask 处理

块级范围给出了需要处理的 K block 集合。对于边界 K block，需要进一步区分哪些 (Q token, K token) 对在窗口内。代码分三阶段循环：

```
n_block_max
│
│  ← Phase 1 (右边界 mask，causal/local)
│    K block 满足：某些 Q token 已超出其 causal 边界，但 m_idx_min 还未到
│    边界由 get_n_block_min_causal_local_mask() 计算（用 m_idx_MIN）
│    n_mask_causal = max(n_block_min, m_idx_min / kBlockN)
│
│  ← Phase 2 (无 mask，全覆盖)
│    所有 Q token 都完整覆盖该 K block
│    上界：n_block_min_before_local_mask（用 m_idx_MAX 减 window_size_left）
│
│  ← Phase 3 (左边界 mask，local)
│    K block 满足：Q block 中部分 token 的左窗口边界在其中
│    边界由 get_n_block_min_before_local_mask() 计算（用 m_idx_MAX）
│    n_mask_local = max(n_block_min, ceil_div(m_idx_max - window_size_left, kBlockN))
│
n_block_min
```

### 4.1 Phase 1 右边界计算

```cpp
// get_n_block_min_causal_local_mask
int m_idx_min = m_block * kBlockM;  // PackGQA: / qhead_per_khead
int n_idx_right = m_idx_min + window_size_right;  // causal: +0
return max(n_block_min, n_idx_right / kBlockN);
```

### 4.2 Phase 3 左边界计算

```cpp
// get_n_block_min_before_local_mask
int m_idx_max = (m_block + 1) * kBlockM;  // PackGQA: 上界序列 token
int n_idx_left = m_idx_max - window_size_left;
return max(n_block_min, ceil_div(n_idx_left, kBlockN));
```

### 4.3 m_block=8 示例

```
Q token 范围 [128, 144)，n_block_min=0, n_block_max=2

n_mask_causal = max(0, 128/128) = 1  → Phase 1: n_block ∈ [1, 2)
n_mask_local  = max(0, ceil_div(144-128, 128)) = 1  → Phase 2: 空
                                                     → Phase 3: n_block ∈ [0, 1)

Phase 1（n_block=1，K token [128, 256)）：
  q=128: 右边界=129，左边界=0  → K[128] 有效，K[129..] 截断
  q=143: 右边界=144，左边界=15 → K[128..143] 有效

Phase 3（n_block=0，K token [0, 128)）：
  q=128: 左边界=0   → 全部有效
  q=143: 左边界=15  → K[0..14] 截断，K[15..127] 有效
```

---

## 五、元素级 mask（mask.h）

在每个需要 mask 的 K block 内，对每个 (Q smem行, K smem列) 独立判断：

```cpp
// Local_mask 分支（PackGQA 下 mma_m_idx 是序列维度 token 索引）
int col_limit_right = mma_m_idx + 1 + seqlen_k - seqlen_q;  // causal 右边界（不含）
int col_limit_left  = col_limit_right - 1 - window_size_left;  // SWA 左边界（含）

// 对每个矩阵元素(m_idx, n_col)：
if (n_col >= col_limit_right || n_col < col_limit_left)
    score = -INFINITY;
```

每行（每个 Q token）独立计算窗口，精确实现逐 token 的 SWA。

---

## 六、设计要点总结

| 层级 | 粒度 | 计算方式 | 目的 |
|---|---|---|---|
| `get_n_block_min_max` | K block | Q block 内边缘 token 的并集 | 跳过完全在窗口外的 K block |
| 三阶段循环 | K block | 按 mask 类型分段 | 区分需要 mask 的 K block |
| `mask.h` 元素级 | 单个 (Q,K) | 每行独立计算 col_limit | 精确 SWA 窗口边界 |

**关键不变式**：

1. `n_block_max` 用 Q block **末尾** token 决定（包含最远能看到的 K token）
2. `n_block_min` 用 Q block **起始** token 决定（左窗口最远延伸）
3. 块内的精细裁剪完全交由元素级 mask 处理，两层分工明确
