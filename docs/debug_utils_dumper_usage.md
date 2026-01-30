# Debug Utils Dumper 使用指南

本指南介绍 `megatron.core.debug_utils.dumper` 的常用配置与使用方式，帮助你在训练/推理过程中采集中间张量并进行分析。

## 1. 快速开始

### 1.1 最小可用示例

```python
import torch

from megatron.core.debug_utils import dumper

# Enable dumper explicitly if env var is not set.
dumper.enable = True
dumper.dump_dir = "/tmp/megatron_dumps"
dumper.dump_dp_rank_0_only = False

dumper.on_training_start()

for iteration in range(2):
    dumper.on_iteration_start(iteration)
    x = torch.randn(2, 4)
    dumper.dump("hidden_states", x, layer_id=0)

dumper.flush()
```

### 1.2 只记录日志、不写文件

```python
from megatron.core.debug_utils import dumper

dumper.enable = True
dumper.write_file = False  # Log only, no files written.
dumper.on_iteration_start(0)
dumper.dump("attention_output", tensor, save=True)
```

> `save=False` 也可临时禁止写入；若 `write_file=False` 则全部不落盘。

## 2. 环境变量配置

常用环境变量如下（均为字符串，`"1"` 表示开启）：

```text
MEGATRON_DUMPER_ENABLE=1
MEGATRON_DUMPER_DIR=/tmp/megatron_dumps
MEGATRON_DUMPER_WRITE_FILE=1
MEGATRON_DUMPER_DP_RANK_0_ONLY=1
MEGATRON_DUMPER_ASYNC=0
MEGATRON_DUMPER_LAYERS="0,1,last"
MEGATRON_DUMPER_NAMES="attention|mlp"
MEGATRON_DUMPER_ITERATIONS="0,every:100"
MEGATRON_DUMPER_AGGREGATE_TP=0
MEGATRON_DUMPER_GRADIENTS=0
MEGATRON_DUMPER_LOG_STATS=1
```

## 3. 核心 API

### 3.1 生命周期方法

```python
from megatron.core.debug_utils import dumper

dumper.on_training_start()        # Create a session and write session metadata.
dumper.on_iteration_start(10)     # Reset iteration counters.
dumper.on_micro_batch_start(0)    # Reset micro-batch counters.
dumper.flush()                    # Wait for async writes.
```

### 3.2 dump / dump_dict

```python
from megatron.core.debug_utils import dumper

# Single tensor or object.
dumper.dump("layer_0.attention.query", tensor, layer_id=0)

# Batch dump from a dict.
dumper.dump_dict(
    "attention",
    {"query": q, "key": k, "value": v},
    layer_id=0,
)
```

### 3.3 上下文变量

```python
from megatron.core.debug_utils import dumper

dumper.set_ctx(phase="forward", layer_id=3)
dumper.dump("mlp_output", tensor)
dumper.set_ctx(layer_id=None)  # Clear a single key.

with dumper.context(phase="prefill", layer_id=5):
    dumper.dump("attention_output", tensor)
```

## 4. 过滤机制

支持按层号、名称、迭代过滤，既可以通过环境变量配置，也可以运行时动态更新：

```python
from megatron.core.debug_utils import dumper

dumper.filter_engine.update_layer_filter("0,1,last")
dumper.filter_engine.update_name_filter("attention|mlp")
dumper.filter_engine.update_iteration_filter("0,every:100")
```

过滤语法示例：

- 层号：`"0,1,5-10,last"`
- 名称正则：`"attention|mlp"`
- 迭代：`"0,100"`, `"100-200"`, `"every:100"`, `"first:10"`

## 5. Hook 自动注册

用于自动抓取 TransformerLayer 中间张量：

```python
from megatron.core.debug_utils import dumper

dumper.enable = True
dumper.register_transformer_hooks(
    model,
    dump_points=["post_attention", "post_mlp"],
)

# Remove all hooks when done.
dumper.remove_all_hooks()
```

支持的 `dump_points`：

- `pre_attention`
- `post_attention`
- `post_mlp`
- `post_layernorm`

## 6. TP 聚合

启用后，会在 TP rank 0 汇聚分片张量并写入完整张量：

```python
from megatron.core.debug_utils import dumper, TensorShardingType

dumper.aggregate_tp = True
dumper.dump("self_attention.query", tensor, sharding_type=TensorShardingType.TP_COLUMN)
```

> 当 `MEGATRON_DUMPER_AGGREGATE_TP=1` 时，非 TP rank 0 会跳过写入。

## 7. 梯度 Dump

在 `loss.backward()` 之后调用：

```python
from megatron.core.debug_utils import dumper

dumper.dump_gradients_enabled = True
dumper.dump_gradients(model.layer, "layer_0", layer_id=0)
```

## 8. 异步写入

启用异步写入后，建议在程序结束前调用 `flush()`：

```python
from megatron.core.debug_utils import dumper

dumper.enable = True
dumper.storage.async_write = True  # Or set MEGATRON_DUMPER_ASYNC=1 before init.

# ... dump calls ...
dumper.flush()
```

## 9. 文件组织与读取

文件结构：

```text
{dump_dir}/
  {session_name}/
    iter_000000/
      name=...___iter=0___mb=0___grank=0___tprank=0___pprank=0___idx=1[___layer=0].pt
    session_metadata.json
```

读取示例：

```python
import torch

payload = torch.load(filepath, weights_only=False)
data = payload["data"]
metadata = payload.get("metadata", {})
```

> 注意：`name` 中请避免包含 `___` 或 `=`，否则 `parse_dump_filename` 可能无法正确解析。

## 10. 常见问题

- **没有生成文件**：确认 `dumper.enable=True`、`write_file=True`、`dump_dir` 可写。
- **分布式只写 rank 0**：默认 `MEGATRON_DUMPER_DP_RANK_0_ONLY=1`，如需全部写出请设为 `0`。
- **异步写入缺文件**：程序退出前调用 `dumper.flush()`。

