# Megatron Dumper 分步开发计划

> 基于 [megatron_dumper_design.md](./megatron_dumper_design.md) 设计文档
> 共分 4 个阶段，按依赖顺序递进开发

---

## 阶段一：基础设施与并行适配层

### 目标

搭建项目骨架，实现 Megatron 并行环境感知能力

### 开发内容

#### 1.1 项目结构初始化

```
megatron/core/debug_utils/
├── __init__.py              # 模块导出
├── parallel_adapter.py      # 并行状态适配器
├── metadata.py              # 元数据定义
└── utils.py                 # 工具函数
```

#### 1.2 ParallelAdapter 实现 (`parallel_adapter.py`)

| 方法 | 功能 |
|------|------|
| `get_parallel_info()` | 获取完整并行信息 (TP/PP/DP/CP/EP/VPP) |
| `should_dump_for_dp()` | DP rank 过滤判断 |
| `gather_across_tp()` | 跨 TP rank 聚合张量 |
| `get_tp_group()` | 获取 TP 通信组 |
| `_get_layer_offset()` | 计算 PP stage 层号偏移 |

#### 1.3 元数据与工具 (`metadata.py`, `utils.py`)

- `TensorShardingType` 枚举 (REPLICATED, TP_COLUMN, TP_ROW, CP_SPLIT, EP_SPLIT)
- `TENSOR_SHARDING_MAP` 常见张量位置的分片类型映射
- `parse_dump_filename()` 文件名解析函数
- 日志配置和初始化

### 交付物

- 可独立测试的并行状态获取模块
- 单元测试覆盖各并行维度信息获取

### 验收标准

- [ ] `ParallelAdapter.get_parallel_info()` 在分布式环境下返回正确信息
- [ ] 非分布式环境下优雅降级
- [ ] 单元测试通过

---

## 阶段二：核心 Dump 引擎

### 目标

实现张量 dump 的核心逻辑，支持多维度过滤和本地存储

### 开发内容

#### 2.1 FilterEngine 实现 (`filter_engine.py`)

| 方法 | 功能 |
|------|------|
| `__init__()` | 解析层号/名称/迭代过滤配置 |
| `should_dump()` | 综合判断是否应该 dump |
| `_parse_layer_filter()` | 解析 "0,1,5-10,last" 格式 |
| `_parse_iteration_filter()` | 解析 "every:100,first:10" 格式 |
| `_check_iteration()` | 迭代号匹配检查 |

**过滤语法支持**：
- 层号: `"0,1,2"`, `"0-5"`, `"first,last"`
- 迭代: `"0,100"`, `"100-200"`, `"every:100"`, `"first:10"`
- 名称: 正则表达式 `"attention|mlp"`

#### 2.2 StorageBackend 实现 (`storage_backend.py`)

| 方法 | 功能 |
|------|------|
| `save()` | 保存数据到文件 |
| `load()` | 加载数据 |
| `_write_sync()` | 同步写入实现 |

**文件格式**：
```python
{
    "data": torch.Tensor,
    "metadata": {...}
}
```

#### 2.3 MegatronDumper 核心 (`dumper.py`)

| 方法/属性 | 功能 |
|-----------|------|
| `__new__()` | 单例模式实现 |
| `__init__()` | 环境变量配置读取 |
| `dump()` | 核心 dump 方法 |
| `dump_dict()` | 批量 dump 字典/对象 |
| `set_ctx()` | 设置上下文变量 |
| `context()` | 上下文管理器 |
| `_build_filepath()` | 构建文件路径 |
| `_log_tensor_info()` | 日志输出张量信息 |

**环境变量配置**：
```bash
MEGATRON_DUMPER_ENABLE=1
MEGATRON_DUMPER_DIR=/tmp/megatron_dumps
MEGATRON_DUMPER_WRITE_FILE=1
MEGATRON_DUMPER_DP_RANK_0_ONLY=1
MEGATRON_DUMPER_LAYERS="0,1,last"
MEGATRON_DUMPER_NAMES="attention|mlp"
MEGATRON_DUMPER_ITERATIONS="0,every:100"
```

### 交付物

- 可手动调用的 `dumper.dump()` 功能
- 支持过滤的 dump 机制
- 规范化的文件存储

### 验收标准

- [ ] `dumper.dump("name", tensor)` 正确保存文件
- [ ] 过滤配置生效（层号/名称/迭代）
- [ ] 文件命名符合规范
- [ ] 日志输出包含张量统计信息

---

## 阶段三：高级功能与自动化

### 目标

实现自动 Hook 注册、异步写入、梯度 dump 等高级特性

### 开发内容

#### 3.1 Hook 注册系统

| 方法 | 功能 |
|------|------|
| `register_transformer_hooks()` | 为模型自动注册 dump hooks |
| `_register_layer_hooks()` | 单层 hook 注册 |
| `_extract_layer_id()` | 从模块名提取层号 |
| `remove_all_hooks()` | 移除所有已注册 hooks |

**支持的 dump_points**：
- `pre_attention` - attention 前
- `post_attention` - attention 后
- `post_mlp` - MLP 后
- `post_layernorm` - LayerNorm 后

#### 3.2 生命周期管理

| 方法 | 调用时机 |
|------|----------|
| `on_training_start()` | 训练开始，初始化 session |
| `on_iteration_start(iter)` | 每个迭代开始 |
| `on_micro_batch_start(mb_id)` | 每个 micro-batch 开始 |

**Session 管理**：
- `_generate_session_name()` - 生成并跨 rank 同步 session 名
- `_ensure_dump_dir()` - 确保目录存在
- `_save_session_metadata()` - 保存 session 元数据 JSON

#### 3.3 异步写入

```python
# 实现要点
- self._write_queue = queue.Queue()
- self._writer_thread = threading.Thread(target=self._async_writer_loop, daemon=True)
- CPU 张量复制: value.detach().cpu().clone()
- flush() 等待队列清空
```

| 配置 | 说明 |
|------|------|
| `MEGATRON_DUMPER_ASYNC=1` | 启用异步写入 |

#### 3.4 TP 聚合与梯度 Dump

| 方法 | 功能 |
|------|------|
| `_maybe_aggregate_tp()` | 根据 sharding_type 决定是否聚合 |
| `dump_gradients()` | dump 模块参数的梯度 |

| 配置 | 说明 |
|------|------|
| `MEGATRON_DUMPER_AGGREGATE_TP=1` | 启用 TP 聚合 |
| `MEGATRON_DUMPER_GRADIENTS=1` | 启用梯度 dump |

### 交付物

- 自动 Hook 注册功能
- 异步写入支持
- 完整的生命周期管理
- TP 聚合和梯度 dump 能力

### 验收标准

- [ ] `register_transformer_hooks()` 自动捕获指定位置的张量
- [ ] 异步写入不阻塞训练主循环
- [ ] Session 元数据正确保存
- [ ] TP 聚合后张量形状正确
- [ ] 梯度 dump 包含所有参数梯度

---

## 阶段四：配套工具与集成测试

### 目标

提供数据加载、比对、聚合工具，完成与 Megatron 训练脚本的集成

### 开发内容

#### 4.1 DumpLoader 实现 (`dump_loader.py`)

| 方法 | 功能 |
|------|------|
| `__init__(dump_dir)` | 初始化，指定 dump 目录 |
| `list_sessions()` | 列出所有 session |
| `load_session(name)` | 加载指定 session |
| `list_iterations()` | 列出所有迭代 |
| `list_tensors(iteration)` | 列出某迭代的所有张量（返回 DataFrame） |
| `load(name, iteration, ...)` | 按条件加载单个张量 |
| `load_all_tp_shards(name, iteration)` | 加载所有 TP 分片 |

**使用示例**：
```python
loader = DumpLoader("/data/dumps/session_xxx")
df = loader.list_tensors(iteration=0)
tensor = loader.load("layer_0.attention_output", iteration=0, tp_rank=0)
```

#### 4.2 DumpComparator 实现 (`dump_comparator.py`)

| 方法 | 功能 |
|------|------|
| `__init__(baseline_dir, target_dir, tolerance)` | 初始化比对器 |
| `compare_tensor(name, iteration)` | 比对单个张量 |
| `compare_all(iteration, names)` | 批量比对 |
| `generate_report(output_path)` | 生成 HTML 报告 |

**ComparisonResult 数据类**：
```python
@dataclass
class ComparisonResult:
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    baseline_shape: tuple
    target_shape: tuple
```

**CLI 入口**：
```bash
python -m megatron.core.debug_utils.dump_comparator \
    --baseline /data/dumps/baseline \
    --target /data/dumps/target \
    --tolerance 1e-5 \
    --output report.html
```

#### 4.3 DumpAggregator 实现 (`dump_aggregator.py`)

| 方法 | 功能 |
|------|------|
| `aggregate_tp(name, iteration, tp_size, dim)` | 聚合 TP 分片 |
| `aggregate_pp(name_pattern, iteration, pp_size)` | 聚合 PP stages |

#### 4.4 集成与测试

**训练脚本集成示例**：
```python
from megatron.core.debug_utils import dumper

# 训练开始
dumper.on_training_start()
dumper.register_transformer_hooks(model)

# 训练循环
for iteration in range(start, end):
    dumper.on_iteration_start(iteration)
    for mb_id, mb in enumerate(micro_batches):
        dumper.on_micro_batch_start(mb_id)
        output = model(mb)
        loss.backward()
```

**测试计划**：
| 测试类型 | 内容 |
|----------|------|
| 单元测试 | 各组件独立测试，mock parallel_state |
| 集成测试 | 小模型端到端 dump 和加载 |
| 比对测试 | FP32 vs BF16 精度比对验证 |
| 性能测试 | 同步/异步写入开销测量 |

### 交付物

- 完整的 DumpLoader / DumpComparator / DumpAggregator 工具
- CLI 比对工具
- 集成示例代码
- 测试用例

### 验收标准

- [ ] DumpLoader 能正确加载和解析所有 dump 文件
- [ ] DumpComparator 能检测出数值差异并生成报告
- [ ] CLI 工具可用
- [ ] 集成测试端到端通过
- [ ] 性能测试：异步写入开销 < 5%

---

## 开发依赖关系

```
阶段一 ──────────────────────────────────────────────────────────────────┐
  │                                                                      │
  ▼                                                                      │
阶段二 ──────────────────────────────────────────────────────────────────┤
  │                                                                      │
  ▼                                                                      │
阶段三 ──────────────────────────────────────────────────────────────────┤
  │                                                                      │
  ▼                                                                      │
阶段四 ◄─────────────────────────────────────────────────────────────────┘
```

## 里程碑总结

| 阶段 | 核心交付 | 里程碑标志 |
|------|----------|------------|
| **阶段一** | ParallelAdapter | 并行适配器单测通过 |
| **阶段二** | Dumper Core + Filter + Storage | 手动 `dump()` 可工作 |
| **阶段三** | Hooks + Async + Gradients | 自动 Hook dump 可工作 |
| **阶段四** | Loader + Comparator + 集成 | 端到端比对测试通过 |

---

## 文件清单

完成后的目录结构：

```
megatron/core/debug_utils/
├── __init__.py              # 模块导出
├── dumper.py                # 核心 Dumper 单例
├── parallel_adapter.py      # Megatron 并行状态适配器
├── filter_engine.py         # 多维度过滤引擎
├── storage_backend.py       # 存储后端
├── metadata.py              # 元数据定义
├── tensor_processor.py      # 张量预处理（可选）
├── dump_loader.py           # Dump 数据加载器
├── dump_comparator.py       # Dump 数据比对器
├── dump_aggregator.py       # 跨 rank 张量聚合器
├── utils.py                 # 工具函数
└── __main__.py              # CLI 入口
```

---

*文档版本: 1.0*
