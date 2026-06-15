# 时序-层级序列增强实现规划

## 文档目的

本文档评估前述时序-层级（Temporal-Hierarchical，TH）序列增强方案基于当前代码框架的实现可行性，并记录：

- 各方案的实现难度和预期成本；
- 可以复用的现有代码及必要重构；
- 具体模块、类、配置和测试落地路径；
- 数据泄漏与评测协议约束；
- 推荐实现顺序。

本文档聚焦 `train_SMB_decoder` 的数据侧增强设计，暂不直接实现代码。

## 当前代码基础（已具备，建议先做局部重构）

当前实现已经提供大部分候选方案所需的信息：

| 所需信息 | 当前来源 |
| --- | --- |
| 行为类型和层级 | `history_behaviors`、`behavior_level` |
| 目标行为 | `target_behavior` |
| 交互顺序 | 每个用户历史列表的原始顺序 |
| Session 边界 | `session` |
| 时间戳 | `time`，当前归一化为半小时单位 |
| 数据切分边界 | `valid_pos`、`test_pos` |
| 用户级确定性随机性 | Fixed-ratio dataset 中的 `_user_seed(uid)` |

当前主要限制不是信息缺失，而是增强代码的组织方式：

- 增强逻辑嵌入完整 Dataset 子类。
- Decoder augmentation 和 fixed-ratio 实现重复了筛选及样本组装逻辑。
- 增强次数、行为比例等参数编码在 task 名称中。
- 各 Dataset 手工拼接 cache 文件名。
- 当前一次性预处理并缓存 pickle，不适合依赖 epoch 的动态增强。
- Fixed-ratio task 同时修改 train、valid 和 test 历史，适合鲁棒性实验，但不应作为训练增强对比的默认评测协议。

因此，多数静态增强方案在增加轻量 policy 层后都比较容易实现。

## 可行性总览

| 方案 | 实现可行性 | 预计成本 | 是否修改模型 | 推荐状态 |
| --- | --- | --- | --- | --- |
| Time-Decayed Behavior Dropout | 容易 | 低 | 否 | 第一批实现 |
| Session-Aware Dropout | 容易 | 低至中 | 否 | 第一批实现 |
| User-Adaptive Ratio | 中等 | 中 | 否 | Policy 抽象后实现 |
| Dataset-Level Fixed Proportion | 容易至中等 | 中 | 否 | 作为控制实验实现 |
| Target-Conditioned Augmentation | 中等 | 中 | 当前 target-aware 样本下不需要 | 明确协议后实现 |
| Multi-View Sequence Augmentation | 中等 | 中 | 第一版不需要 | 基础 policy 后实现 |
| Curriculum Augmentation | 当前缓存结构下较难 | 高 | 需要 Trainer/数据管线配合 | 暂缓 |

## 推荐的共享实现架构（优先重构）

### Policy 接口

新增策略模块：

```text
SeqRec/datasets/session_behavior/augmentation_policies.py
```

建议使用结构化序列输入，不再传递四个平行 list：

```python
@dataclass(frozen=True)
class BehaviorSequence:
    items: list[str]
    behaviors: list[str]
    session_ids: list[int]
    times: list[float]


@dataclass(frozen=True)
class AugmentedView:
    name: str
    keep_indices: list[int]
    metadata: dict[str, Any]


class SequenceAugmentationPolicy(Protocol):
    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        ...
```

`AugmentationContext` 只应包含因果可用信息：

```text
uid
mode
target_index
target_behavior
target_level
behavior_level
max_behavior_level
```

Policy 应返回保留索引，而不是复制后的 item 数组。统一 helper 再将同一个 mask 应用到所有对齐字段，可避免 `items`、`behaviors`、`session_ids` 和 `times` 错位。

### 统一 Decoder Dataset

新增：

```text
SeqRec/datasets/session_behavior/augmented_decoder.py
```

推荐类名：

```text
SMBPolicyAugmentedDatasetForDecoder
```

建议继承 `SMBExplicitDatasetForDecoder`，这样现有 decoder-only collator 可以继续通过 `isinstance` 自动识别，无需改变训练 label 协议。

该类负责：

1. 构造因果历史和预测 target。
2. 根据用户、split 和可选 view id 构造确定性 RNG。
3. 调用所选 augmentation policy。
4. 校验 policy 返回的索引。
5. 组装现有样本 schema：
   `item`、`inters`、`session_ids`、`extended_session_ids`、`actions`、`time` 和 `behavior`。
6. 根据配置决定是否额外输出完整历史视图。

Policy 不负责 tokenizer 字符串和 token-level metadata 的构造。

### Policy Registry

可在同一模块或以下新模块增加轻量 registry：

```text
SeqRec/datasets/session_behavior/augmentation_registry.py
```

建议支持的名称：

```text
none
uniform_level
fixed_ratio
time_decay
session
user_adaptive_ratio
dataset_proportion
target_conditioned
multi_view
```

不建议为每个策略新增一套完整 Dataset 类。

### 显式参数

建议在 `DatasetArgs` 增加增强参数，不再把所有值编码进 `--tasks`：

```text
--sequence_augmentation none
--augmentation_views 1
--augmentation_seed 42
--augmentation_keep_original
--augmentation_eval_mode original
--behavior_keep_ratios 1.0,0.8,0.6
--time_decay_type exponential
--time_decay_tau 48
--recent_session_count 1
```

如果平铺参数过多，也可以先支持紧凑的 JSON/config 文件：

```text
--sequence_augmentation_config config/augmentation/time_decay.json
```

推荐迁移方式：

- 保留现有 task 名称，保证旧命令兼容。
- 新增统一 task，例如 `smb_policy_decoder`。
- 使用显式 dataset 参数选择 policy 和具体配置。
- 后续不再新增编码大量参数的 task 名称。

该改动需要将新 dataset 参数通过 `TrainSMBDecoder.load_train_data` 传入 `load_SMB_datasets`。

### Cache 标识

每个静态 policy 应提供稳定、可序列化的配置：

```python
policy.cache_key()
```

Dataset cache 文件名至少包含：

- policy 名称；
- 规范化 policy 配置的短 hash；
- augmentation seed；
- view 数量；
- 是否包含原始视图；
- split 和 index suffix。

这样可以避免修改 policy 参数后错误复用旧缓存。

### 训练与评测协议

默认协议建议为：

```text
train：增强后的历史
valid：原始历史
test：原始历史
```

这样才能将性能变化明确归因于训练增强。

另设鲁棒性协议：

```text
train：增强或原始历史
valid/test：显式损坏后的历史
```

当前 fixed-ratio 在 train、valid、test 上保持一致裁剪的行为可以保留为具名鲁棒性实验，但不应成为训练增强方案的默认对比方式。

## Time-Decayed Behavior Dropout（容易，最高优先级）

### 可复用部分

可直接复用：

- `times`；
- `behavior_level`；
- target behavior 保护逻辑；
- 确定性 RNG；
- 原始视图加增强视图的样本组织方式。

无需修改模型、collator、tokenizer 或样本 schema。

### Policy 设计

新增：

```text
TimeDecayDropoutPolicy
```

对每个历史交互计算：

```text
p_drop(i) = severity * level_weight(level_i) * age_weight(delta_t_i)
```

当前 `time` 是从用户第一次交互开始递增的时间值，因此使用：

```text
delta_t_i = target_time - time_i
```

第一版推荐：

```text
age_weight = 1 - exp(-delta_t / tau)
level_weight(l) = 1 / (l + 1)
```

这样近期交互的 age weight 接近零，较早的浅层行为最容易删除，target-level 历史则完全保护或赋予很小的删除权重。

### 必要保护条件

- 永远保留预测 target。
- 至少保留 `min_history_items` 个历史交互。
- 至少保留最近 `min_recent_items` 个交互。
- 可选：保留全部历史 target-level 行为。
- 将概率限制在 `[0, max_drop_probability]`。
- 正确处理时间戳相同和时间跨度为零的情况。

### 建议参数

```text
time_decay_type = exponential | linear_rank | bucket
time_decay_tau = 48.0
time_decay_severity = 0.5
time_decay_max_drop = 0.9
time_decay_min_recent_items = 1
time_decay_preserve_target_level = true
```

### 测试重点

- 旧交互的经验删除率高于近期交互。
- 浅层行为的删除率高于高层行为。
- 所有对齐字段保持一致。
- 相同 seed 产生相同视图。
- 不使用 target 之后的时间戳。
- target 和要求保留的近期交互不被删除。
- 时间跨度为零时不报错。

## Session-Aware Dropout（容易，最高优先级）

### 可复用部分

当前数据已经提供归一化 session id 和 session 边界，不需要修改预处理格式。

### Policy 设计

新增：

```text
SessionAwareDropoutPolicy
```

先按 session 对历史索引分组，并只基于该 session 计算：

```text
recency
maximum behavior level
contains target-level behavior
interaction count
```

第一版推荐规则：

1. 始终保留最近 `recent_session_count` 个 session。
2. 如果存在高层行为 session，至少保留最近一个。
3. 根据 recency 和最大行为层级对更早 session 做采样。
4. 被选中的 session 内交互完整保留。

第二阶段可以在已保留 session 内继续做 behavior dropout，但第一版应保持 session 原子性。

### 建议参数

```text
recent_session_count = 1
session_keep_probability = 0.5
session_time_decay_tau = 7
session_high_level_bonus = 0.3
session_preserve_target_level = true
```

### Cache 与 split 细节

随机决策应使用 user seed 加稳定 session id。不要仅根据截断 prefix 中的 session 序号生成随机性，否则同一个历史 session 在 train 和 valid prefix 中可能得到不同决策。

### 测试重点

- Session 被整体保留或整体删除。
- 要求保留的最近 session 始终存在。
- 兼容 prefix 之间对历史 session 的决策稳定。
- Session 顺序不改变。
- 单 session 用户保持有效。
- 稀疏用户仍保留足够历史。

## User-Adaptive Ratio（中等，中高优先级）

### 可行性

基于当前历史可以实现，但统计估计器需要比筛选逻辑本身更谨慎。

### Policy 设计

新增：

```text
UserAdaptiveRatioPolicy
```

基于当前因果历史 prefix 估计用户行为比例：

```text
r_user,l = (count_user,l + alpha * r_global,l) / (count_user,target + alpha)
```

再向训练集全局先验收缩：

```text
r_final,l = confidence_u * r_user,l
          + (1 - confidence_u) * r_global,l
```

并设置上下界：

```text
r_final,l = clip(r_final,l, min_ratio_l, max_ratio_l)
```

只有某层行为数量超过 cap 时才进行下采样。

### 全局统计

新增训练集统计 helper：

```text
SeqRec/datasets/session_behavior/statistics.py
```

统计只能使用每个用户的训练 prefix：

```text
history[:valid_pos]
```

结果存入带版本标识的轻量 JSON cache。Valid 和 test 必须复用训练集得到的 prior，不能根据未来交互重新计算。

### Target 数为零的用户

不能像当前 fixed-ratio 一样在 target count 为零时直接返回原序列。应使用平滑分母或序列长度 cap：

```text
cap_l = min(
    ratio_cap_from_global_prior,
    level_share_cap_l * history_length,
)
```

### 风险

- 增强模式可能过度编码用户 conversion propensity。
- 强依赖 target count 的规则可能抹平用户间有价值的差异。
- 极稀疏用户必须使用较强的全局收缩。

因此，建议先完成 Dataset-Level Fixed Proportion 对照，再判断该方案是否适合作为主增强。

### 测试重点

- 全局统计只使用训练 prefix。
- 稀疏用户向全局先验回退。
- 正确处理 target count 为零的用户。
- 比例始终处于配置上下界内。
- Seed 和 prefix 规则具有确定性。

## Dataset-Level Fixed Behavior Proportion（容易至中等，控制实验）

### Policy 设计

新增：

```text
DatasetProportionPolicy
```

复用同一个训练集统计 helper，并支持两类 preset：

```text
natural：接近原始训练分布
balanced：限制占比过高的浅层行为
```

使用 soft cap，而不是强制精确比例：

```text
max_count_l = ceil(history_length * target_share_l * tolerance)
```

只对超过 cap 的层级进行下采样。

### 方案价值

- 能处理没有 target behavior 的用户。
- 相比按每个用户 target count 归一，是更干净的全局控制实验。
- 可以区分“分布平衡收益”和“用户自适应收益”。

### 测试重点

- 全局统计只来自训练集。
- 未超过 cap 的层级不被修改。
- 输出分布向目标分布靠近，但不被强制完全一致。
- 短历史受到保护。

## Target-Conditioned Augmentation（中等，需要先确认协议）

### 可行性

Decoder 训练在构造每个样本时已经知道 target behavior，因此无需修改模型，Policy 可以直接使用：

```text
context.target_level
```

但当前 `SMBExplicitDatasetForDecoder` 每个用户只在 split 边界构造一个训练 target，相比“每个位置都构造 target”的 dataset，其 target-level 多样性有限。

### Policy 设计

新增：

```text
TargetConditionedPolicy
```

建议作为其他基础 policy 的 wrapper：

```text
TargetConditionedPolicy(base_policy=time_decay)
```

它根据 target 调整以下关系的保留权重：

```text
same-level evidence
one-level-below evidence
upward-path evidence
general temporal evidence
```

对于高层 target，更多保留近期低层前置信号；对于浅层 target，更多保留 same-level 和近期 temporal evidence。

### 泄漏约束

只有当推理和评测时已知 behavior prompt 或目标行为类型时，使用 target behavior 才是合法的。如果实际任务还需要预测行为类型，该增强会泄漏 target 信息。

第一轮实验应限定在 behavior-specific next-item prediction，即 target behavior 本身属于任务定义。

### 分布捷径风险

保留交互的数量或类型不能唯一暴露 target level。建议：

- 不同 target level 的 keep probability 区间相互重叠。
- 控制不同 target 下的期望序列长度接近。
- 保留一个不带条件的 full view。
- 使用简单 count feature 或 augmentation metadata 训练 probe，检查是否可轻易预测 target level。

### 测试重点

- 只使用当前样本 target。
- 不查看未来 behavior。
- 不同 target level 的期望历史长度接近。
- Target condition 确实改变预期 relation category 的权重。

## Multi-View Sequence Augmentation（中等，中高优先级）

### 可行性

当前 decoder dataset 已经能够为每个用户输出多条增强序列，因此只需将基于 ratio 的无语义视图替换为具名语义 policy。

### Policy 设计

新增：

```text
MultiViewAugmentationPolicy
```

组合已有 policy 或确定性 selector：

```text
full
recent
same_level
upward_evidence
session_subsampled
```

第一版推荐四类视图：

1. `full`：不修改的因果历史。
2. `recent`：time-decayed 或固定近期窗口。
3. `hierarchy`：重点保留 target-level、same-level 和前一层 evidence。
4. `session`：session-aware 子采样。

第一轮不要同时启用所有可能视图。

### 样本权重

简单地为每个用户复制多个 view，会同时改变数据集大小和用户权重。建议支持：

```text
view_sampling = all | one_per_epoch | weighted_static
view_weights
```

第一版为了兼容静态 cache，可以实现 `all` 或 `weighted_static`。`one_per_epoch` 需要在线采样，放到后续阶段。

如果当前 Trainer loss 不消费 sample weight，则应在 dataset 构造时做 weighted-static sampling，而不是添加不会生效的 `sample_weight` 字段。

### 与模型侧 MultiView 的关系

数据侧 MultiView 应能独立用于 TH Base 和 relation-bias 模型。主实验矩阵需要明确区分：

```text
仅数据 MultiView
仅模型 MultiView
二者同时使用
```

避免将性能变化错误归因于其中一侧。

### 测试重点

- 每个 view 符合声明的语义。
- Full view 完全不变。
- 两个 policy 产生相同 keep indices 时不重复输出。
- 每用户最大 view 数得到限制。
- 数据集增长规模可预测。
- View 顺序和 seed 具有确定性。

## Curriculum Augmentation（较难，暂缓）

### 当前静态 Cache 不足的原因

当前 Dataset 在 `__init__` 中一次性生成全部样本并缓存 `inter_data`。依赖 epoch 或 global step 的 curriculum 无法改变这些已缓存样本。

### 所需架构

需要选择以下一种路径：

1. 在 `__getitem__` 中在线增强，并共享 epoch 状态。
2. 预计算多个 view，再由 sampler 根据 epoch 选择。
3. Trainer callback 更新 dataset policy 或 sampler。

未来更推荐“预计算语义视图 + epoch-aware sampler”，这样无需反复构造 tokenizer 字符串，也能控制增强成本。

可能新增：

```text
SeqRec/datasets/session_behavior/view_dataset.py
SeqRec/trainers/callbacks/augmentation_schedule.py
SeqRec/datasets/samplers/curriculum_view_sampler.py
```

### 额外要求

- DDP 下传播 `set_epoch(epoch)`。
- 不同 rank 上保持确定性且避免重复。
- Resume checkpoint 时恢复 curriculum 状态。
- `num_workers > 0` 时状态对 worker 安全。
- Cache identity 不依赖当前 epoch。
- 每个 epoch 记录 view 分布。

只有静态语义视图确认有效后，才建议实现该方案。

## 推荐的文件级实施路径

### 第一阶段：共享静态 Policy 层（优先实现）

新增：

```text
SeqRec/datasets/session_behavior/augmentation_policies.py
SeqRec/datasets/session_behavior/augmented_decoder.py
```

修改：

```text
SeqRec/datasets/session_behavior/__init__.py
SeqRec/datasets/loaders/session_behavior.py
SeqRec/tasks/training/train_SMB_decoder.py
SeqRec/utils/args.py
```

同时为当前 uniform-level 和 fixed-ratio 行为提供兼容 adapter。

### 第二阶段：第一批 Policy

按顺序实现并验证：

1. `TimeDecayDropoutPolicy`。
2. `SessionAwareDropoutPolicy`。
3. `DatasetProportionPolicy`。

这些方案语义清晰，泄漏风险相对较低。

### 第三阶段：自适应和组合 Policy

实现：

1. 训练集行为统计模块。
2. `UserAdaptiveRatioPolicy`。
3. `TargetConditionedPolicy`。
4. `MultiViewAugmentationPolicy`。

### 第四阶段：在线采样与 Curriculum

仅在静态 MultiView 有效，并且缓存数据膨胀成为实际问题时继续。

## 验证规划

### 单元测试

新增：

```text
tests/datasets/session_behavior/test_augmentation_policies.py
tests/datasets/session_behavior/test_augmented_decoder.py
tests/datasets/loaders/test_session_behavior_augmentation.py
```

使用人工构造用户覆盖：

- 单 session 和多 session；
- 相同时间戳和不规则时间间隔；
- 无 target behavior；
- 全部交互位于同一层级；
- 极短历史；
- 多个高层行为；
- Train/valid prefix 共享历史 session。

### 通用不变量

每个 policy 必须满足：

- 原始时间顺序不变。
- 所有对齐字段长度一致。
- 所有保留索引都来自因果历史。
- 预测 target 不会被插回历史。
- 满足最小历史约束。
- 固定 seed 可复现。
- Valid/test 交互不参与训练统计。
- 影响行为的参数变化后 cache key 必须变化。

### 集成验证

每个新 task/policy 至少执行：

1. 在不加载完整真实数据的情况下验证 loader 参数解析。
2. 构造极小 synthetic train/valid/test dataset。
3. 验证 collator 收到的样本 schema 不变。
4. 运行 `python -m compileall main.py SeqRec`。
5. 按项目配置对修改的 Python 文件运行 `flake8`。
6. 在 CPU 上完成一次短 DataLoader iteration。
7. 环境允许时运行 one-step GPU smoke test，不启动完整训练。

### 实验日志

Dataset 构造时记录：

```text
policy 名称和规范化配置
输入/输出序列长度分布
各行为层级 keep rate
各时间 bucket keep rate
每用户保留 session 数
view 数量及出现频率
未发生变化的样本比例
```

这些统计是解释推荐结果和排查训练数据量意外变化所必需的。

## 最终建议

当前框架无需修改 TH 模型，就可以支持大部分候选序列增强。推荐实现主线为：

```text
共享 Policy 抽象
+ 统一 Decoder Dataset
+ 显式增强参数
+ 默认仅增强训练集
```

建议实验顺序：

```text
Time-Decayed Dropout
→ Session-Aware Dropout
→ Dataset-Level Proportion 控制实验
→ User-Adaptive Ratio
→ Target-Conditioned Augmentation
→ 语义 Multi-View Augmentation
→ 仅在静态视图有效后考虑 Curriculum
```

Time-decayed 和 session-aware 最容易实现，也最符合当前 TH 建模叙事。User-adaptive 和 target-conditioned 同样可行，但必须增加更严格的泄漏与分布捷径检查。Curriculum 会越过当前静态 cache 边界，需要 Dataset、sampler、Trainer、DDP 和 resume 行为协同修改，因此现阶段应继续暂缓。
