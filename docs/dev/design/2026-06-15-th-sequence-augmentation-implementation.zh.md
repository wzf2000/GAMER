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

## Decoder 预训练语义（设计修正，待代码对齐）

`smb_explicit_decoder` 的预期训练单元是完整 causal sequence，而不是一个固定单 target。现有 `SMBExplicitDatasetForDecoder` 会把每个训练样本存成：

```text
inters = sequence[:-1]
item = sequence[-1]
```

但 `DecoderOnlyCollator` 会拼接 `input_ids + labels`，并且对 decoder-response dataset 训练时不会 mask `input_ids` 部分。因此实际 loss 是在完整序列上做 causal LM shift loss：

```text
sequence[:-1] + sequence[-1]
```

原始 `smb_explicit_decoder_4` augmentation 应解释为 full-sequence augmentation：

```text
原始完整序列
+ 4 个按 ratio 生成的完整序列 dropout 视图
```

每个视图再用旧 schema 表示成 `inters=view[:-1]`、`item=view[-1]`，只是为了复用现有 collator 和 evaluation 代码。

当前 policy dataset 实现偏离了这个协议。它先固定 `sequence[-1]` 作为预测 target，只对 `sequence[:-1]` 做 policy dropout，然后再把固定 target 拼回去。实际得到的是：

```text
policy(history) + original_tail
```

而不是：

```text
policy(full_sequence)
```

因此，当前 policy augmentation 实验应视为 history-only semantic dropout 消融，而不是与原始 decoder augmentation 完全对齐的最终 policy 版本。后续代码应改为让每个 policy 直接在完整训练序列上生成视图。每个生成视图再拆回 `inters=view[:-1]` 和 `item=view[-1]` 以保持兼容。

对齐后的协议要求：

- 除非显式设置 `augmentation_drop_original`，否则始终包含原始完整序列。
- 增强 policy 作用于 `items[:valid_pos]`、`behaviors[:valid_pos]`、`session_ids[:valid_pos]` 和 `times[:valid_pos]`。
- 过滤后保持原始时间顺序和所有对齐字段长度一致。
- 每个输出视图至少包含两个交互，这样才能拆成 prefix 和 tail。
- 按各 policy 的定义保护指定行为层级，但不要全局强制原始最后一个交互仍然作为尾部，除非这是某个具名 policy 的设计。
- Valid/test Dataset 保持不变：评测仍然是 `history -> candidate target`。
- 从 history-only policy view 切换到 full-sequence policy view 时必须改变 cache 标识。

## 实现状态总览（持续更新）

| 方案或组件 | 实现状态 | 验证状态 | 后续定位 |
| --- | --- | --- | --- |
| 共享 Policy 接口与结构化序列 | 已实现 | 已通过单元测试、compileall 和 flake8 | 作为后续策略扩展基础 |
| 统一 Decoder Dataset | 已实现，但需要 full-sequence 对齐 | 已通过 history-only 实现的 synthetic loader 和样本 schema 测试 | 协议修正后继续作为统一入口 |
| 显式参数、cache key 和 `smb_policy_decoder` | 已实现 | 已验证 CLI、cache 隔离和旧 task 解析兼容 | 继续沿用 |
| Time-Decayed Behavior Dropout | 已实现，但需要 full-sequence dataset 对齐 | 已验证确定性、近期保护和层级保护 | 对齐后重新实验 |
| Session-Aware Dropout | 已实现，但需要 full-sequence dataset 对齐 | 已验证 session 原子性和最小历史保护 | 对齐后重新实验 |
| Dataset-Level Fixed Proportion | 已实现，但需要 full-sequence dataset 对齐 | 已验证 soft cap 和最高层保护 | 对齐后的控制实验 |
| 训练前缀行为统计 | 已实现 | 已验证仅使用 `history[:valid_pos]` | 为全局先验提供基础 |
| User-Adaptive Ratio | 已实现，但需要 full-sequence dataset 对齐 | 已验证零 target 回退和训练前缀先验 | 对齐后重新实验 |
| Target-Conditioned Augmentation | 已实现，但 full-sequence view 下需要重新审视协议 | 已验证 history-only target anchoring 下的 same-level/precursor 恢复 | 重新定义为 tail-conditioned augmentation |
| Multi-View Sequence Augmentation | 已实现，但需要 full-sequence dataset 对齐 | 已验证语义视图生成和 Dataset 去重 | 对齐后重新实验 |
| Curriculum Augmentation | 未实现 | 未验证 | 暂缓 |

## 共享实现架构（已实现并验证）

### Policy 接口（已实现并验证）

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
    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        ...
```

`AugmentationContext` 只应包含因果可用信息：

```text
uid
target_behavior
target_level
target_time
behavior_level
max_behavior_level
```

Policy 应返回保留索引，而不是复制后的 item 数组。统一 helper 再将同一个 mask 应用到所有对齐字段，可避免 `items`、`behaviors`、`session_ids` 和 `times` 错位。

### 统一 Decoder Dataset（已实现并验证）

新增：

```text
SeqRec/datasets/session_behavior/augmented_decoder.py
```

已实现类：

```text
SMBPolicyAugmentedDatasetForDecoder
```

该类继承 `SMBExplicitDatasetForDecoder`，现有 decoder-only collator 可以继续通过 `isinstance` 自动识别，无需改变训练 label 协议。

该类负责：

1. 为每个用户 prefix 构造完整 causal 训练序列。
2. 根据用户、split 和可选 view id 构造确定性 RNG。
3. 调用所选 augmentation policy。
4. 校验 policy 返回的索引。
5. 将每个输出的完整序列视图拆成现有 decoder schema：
   `item`、`inters`、`session_ids`、`extended_session_ids`、`actions`、`time` 和 `behavior`。
6. 根据配置决定是否额外输出原始完整序列视图。

Policy 不负责 tokenizer 字符串和 token-level metadata 的构造。

### Policy Registry（部分实现）

当前由 `SMBPolicyAugmentedDatasetForDecoder._build_policy()` 统一解析已支持策略，尚未拆出独立 registry 模块。后续策略数量增加时可迁移到：

```text
SeqRec/datasets/session_behavior/augmentation_registry.py
```

当前已支持：

```text
time_decay
session
dataset_proportion
user_adaptive_ratio
target_conditioned
multi_view
```

尚未支持：

```text
none
uniform_level
fixed_ratio
```

不建议为每个策略新增一套完整 Dataset 类。

### 显式参数（已实现并验证）

建议在 `DatasetArgs` 增加增强参数，不再把所有值编码进 `--tasks`：

```text
--sequence_augmentation none
--augmentation_views 1
--augmentation_seed 42
--augmentation_drop_original
--time_decay_type exponential
--time_decay_tau 48
--recent_session_count 1
--dataset_proportion_preset natural
```

当前已通过 `DatasetArgs` 和 `TrainSMBDecoder.load_train_data` 接入显式参数，并新增统一 task：

```text
--tasks smb_policy_decoder
```

当前状态：

- 保留现有 task 名称，保证旧命令兼容。
- 已新增统一 task `smb_policy_decoder`。
- 使用显式 dataset 参数选择 policy 和具体配置。
- 后续不再新增编码大量参数的 task 名称。

紧凑 JSON/config 文件尚未实现，可在参数继续增长时再考虑。

### Cache 标识（已实现并验证）

每个静态 policy 应提供稳定、可序列化的配置：

```python
policy.cache_config()
```

Dataset cache 文件名至少包含：

- policy 名称；
- 规范化 policy 配置的短 hash；
- augmentation seed；
- view 数量；
- 是否包含原始视图；
- split 和 index suffix。

这样可以避免修改 policy 参数后错误复用旧缓存。

### 训练与评测协议（默认协议已实现）

默认协议建议为：

```text
train：增强后的完整训练序列
valid：原始历史
test：原始历史
```

这样才能将性能变化明确归因于训练增强。

`smb_policy_decoder` 的目标协议应是上述 full-sequence 训练协议。当前实现仍是 history-only policy view，应先修正后再将 policy 结果解释为最终结论。以下独立鲁棒性协议尚未统一接入新 Policy Dataset：

```text
train：增强或原始历史
valid/test：显式损坏后的历史
```

当前 fixed-ratio 在 train、valid、test 上保持一致裁剪的行为可以保留为具名鲁棒性实验，但不应成为训练增强方案的默认对比方式。

## Time-Decayed Behavior Dropout（已实现并验证，第一批实验）

### 可复用部分

可直接复用：

- `times`；
- `behavior_level`；
- protected level 保护逻辑；
- 确定性 RNG；
- 原始视图加增强视图的样本组织方式。

无需修改模型、collator、tokenizer 或样本 schema。

### Policy 设计

新增：

```text
TimeDecayDropoutPolicy
```

对完整训练序列中的每个交互计算：

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

这样近期交互的 age weight 接近零，较早的浅层行为最容易删除，受保护的高层行为则完全保留或赋予很小的删除权重。

### 必要保护条件

- 至少保留足够构造 decoder 样本的交互数量。
- 至少保留最近 `min_recent_items` 个交互。
- 可选：保留全部 target-level 行为。
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
- 要求保留的近期交互和 protected-level 交互不被删除。
- 时间跨度为零时不报错。

## Session-Aware Dropout（已实现并验证，第一批实验）

### 可复用部分

当前数据已经提供归一化 session id 和 session 边界，不需要修改预处理格式。

### Policy 设计

新增：

```text
SessionAwareDropoutPolicy
```

先按 session 对完整序列索引分组，并只基于该 session 计算：

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

## User-Adaptive Ratio（已实现并验证，第二批实验）

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

## Dataset-Level Fixed Behavior Proportion（已实现并验证，控制实验）

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

## Tail-Conditioned Augmentation（当前以 Target-Conditioned 实现，需重新审视协议）

### 可行性

在 full-sequence decoder 协议下，augmentation 不再默认存在一个单独固定的训练 target。序列尾部行为仍可作为某些 policy 的 anchor，但这应是显式 policy 选择，而不是 Dataset 的默认语义。

如果采用 tail-conditioned policy，可以使用：

```text
context.target_level
```

这里的 `target_level` 更准确地说是原始序列尾部行为层级。

### Policy 设计

新增：

```text
TargetConditionedPolicy
```

建议作为其他基础 policy 的 wrapper：

```text
TargetConditionedPolicy(base_policy=time_decay)
```

当前版本仅支持 `time_decay` 作为 base policy。其他 base policy 组合尚未实现。

它根据 anchor behavior 调整以下关系的保留权重：

```text
same-level evidence
one-level-below evidence
upward-path evidence
general temporal evidence
```

对于高层 anchor，更多保留近期低层前置信号；对于浅层 anchor，更多保留 same-level 和近期 temporal evidence。

### 泄漏约束

使用序列尾部行为作为增强条件在训练数据生成中是可行的，但它会改变增强分布。如果该 policy 希望模拟 behavior-specific inference，则 anchor behavior 必须对应已知 behavior prompt。如果实际任务还需要预测行为类型，强 target-conditioned 的保留模式仍可能产生分布捷径。

该策略应作为更强的增强先验单独报告，而不是默认数据侧协议。作为主增强前，需要先完成分布诊断。

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

## Multi-View Sequence Augmentation（已实现并验证，第二批实验）

### 可行性

原始 decoder dataset 已经能够为每个用户输出多条完整增强序列，因此 policy dataset 应保留这个协议，将基于 ratio 的 full-sequence 视图替换为具名语义 full-sequence policy。

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

1. `full`：不修改的完整训练序列。
2. `recent`：time-decayed 或固定近期窗口。
3. `hierarchy`：重点保留 target-level、same-level 和前一层 evidence。
4. `session`：session-aware 子采样。

第一轮不要同时启用所有可能视图。

Dataset 单独保留可选原始完整序列视图。当前实现生成的是 history-only 视图，应改为生成 full-sequence 视图：

```text
multi_view_recent
multi_view_hierarchy
multi_view_session
```

同一用户产生相同 keep indices 的视图会在 Dataset 层去重。`augmentation_views` 表示整组语义视图的重复采样次数。

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

- 每个 full-sequence view 符合声明的语义。
- Full view 完全不变。
- 两个 policy 产生相同 keep indices 时不重复输出。
- 每用户最大 view 数得到限制。
- 数据集增长规模可预测。
- View 顺序和 seed 具有确定性。

## Curriculum Augmentation（未实现，暂缓）

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

## 文件级实施进度

### 第一阶段：共享静态 Policy 层（已实现并验证）

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

已保持原有 uniform-level 和 fixed-ratio task 的解析兼容，但尚未将它们迁移为新 Policy adapter。

### 第二阶段：第一批 Policy（已实现并验证）

按顺序实现并验证：

1. `TimeDecayDropoutPolicy`。
2. `SessionAwareDropoutPolicy`。
3. `DatasetProportionPolicy`。

这些方案语义清晰，泄漏风险相对较低。

### 第三阶段：自适应和组合 Policy（已实现并验证）

实现：

1. 训练集行为统计模块。
2. `UserAdaptiveRatioPolicy`。
3. `TargetConditionedPolicy`。
4. `MultiViewAugmentationPolicy`。

### 第四阶段：在线采样与 Curriculum（未实现，暂缓）

仅在静态 MultiView 有效，并且缓存数据膨胀成为实际问题时继续。

## 验证状态

### 已实现的测试

新增：

```text
tests/datasets/session_behavior/test_augmentation_policies.py
tests/datasets/session_behavior/test_policy_augmented_dataset.py
```

当前共有 13 个 synthetic tests，已覆盖：

- 单 session 和多 session；
- 时间衰减的近期与最高层保护；
- 固定 seed 的可复现性；
- session 原子保留与最小历史保护；
- dataset proportion soft cap；
- 对齐字段长度校验；
- 三种策略的 loader 构造；
- 训练增强、验证原始的协议；
- 旧 task 解析兼容。
- 训练统计仅使用训练 prefix。
- User-Adaptive 在 target count 为零时的全局先验回退。
- Target-Conditioned 对 same-level 和 precursor evidence 的恢复。
- Multi-View 具名语义视图及 hierarchy view 约束。
- 六种静态 policy 的 loader 构造。

### 通用不变量

每个 policy 必须满足：

- 原始时间顺序不变。
- 所有对齐字段长度一致。
- 所有保留索引都来自因果训练 prefix。
- 每个输出训练视图至少包含两个交互。
- 满足最小序列长度约束。
- 固定 seed 可复现。
- Valid/test 交互不参与训练统计。
- 影响行为的参数变化后 cache key 必须变化。

### 已完成的集成验证

每个新 task/policy 至少执行：

1. Loader 参数解析和旧 task 兼容检查。
2. 极小 synthetic train/valid dataset 构造。
3. Dataset 输出样本 schema 检查。
4. `python -m compileall main.py SeqRec tests`。
5. 按项目配置对修改的 Python 文件运行 `flake8`。
6. CPU 上的数据集构造和取样。
7. `train_SMB_decoder --help` 的 CLI 参数暴露检查。

已在 ShortVideoAD 上执行：

- `Qwen3TemporalHierarchicalMultiViewSoft` 搭配 `session` augmentation，具体为 `smb_policy_decoder` 加 `--sequence_augmentation session`；结果路径为 `results/ShortVideoAD/smb_policy_decoder/Qwen3TemporalHierarchicalMultiViewSoft_aug_session/results-smb_explicit-original.json`。

尚未执行：

- Collator 后的完整 batch 前向。
- One-step GPU smoke test。
- 其他静态 policy 的完整训练和推荐指标实验。

### 实验日志（部分实现）

Dataset 构造时记录：

```text
policy 名称和规范化配置
输入/输出序列长度分布
各行为层级 keep rate
各时间 bucket keep rate
view 数量及出现频率
未发生变化的样本比例
```

当前已记录 policy 配置、输入/输出平均长度、各层级 keep rate、时间 bucket keep rate、view 数和未变化比例。每用户保留 session 数仅存在于 policy metadata，尚未汇总到 Dataset 日志。

## 当前结论与下一步

以下基础主线已经实现并通过 synthetic verification：

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

当前已完成的 ShortVideoAD policy runs 来自现有 history-only policy 实现。它们可以作为诊断结果，但尚未与原始 `smb_explicit_decoder_4` 的 full-sequence augmentation 协议对齐。第一个完成的 run 是在 `Qwen3TemporalHierarchicalMultiViewSoft` 上使用 `session` augmentation，具体为 `smb_policy_decoder` 加 `--sequence_augmentation session`，并使用 `aug_session` run suffix。这里对比的 baseline 是同一 backbone 在原始 `smb_explicit_decoder_4` 任务下的结果，也就是原始 explicit decoder 的 4 倍 full-sequence 序列增强方案，并不是无增强。其 merged behavior 结果为：

| Model / Policy | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MultiViewSoft + 原始 4x augmentation（`smb_explicit_decoder_4`） | 0.0496 | 0.1508 | 0.2218 | 0.0239 | 0.0780 | 0.1212 | 0.0657 | 0.0796 |
| MultiViewSoft + policy session augmentation（`smb_policy_decoder`, `sequence_augmentation=session`） | 0.0432 | 0.1386 | 0.2043 | 0.0205 | 0.0701 | 0.1096 | 0.0589 | 0.0717 |

CVR 目标行为结果同样下降：

| Model / Policy | HR@1 | HR@5 | HR@10 | R@1 | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MultiViewSoft + 原始 4x augmentation（`smb_explicit_decoder_4`） | 0.0417 | 0.1354 | 0.2038 | 0.0328 | 0.1036 | 0.1577 | 0.0739 | 0.0918 |
| MultiViewSoft + policy session augmentation（`smb_policy_decoder`, `sequence_augmentation=session`） | 0.0381 | 0.1256 | 0.1915 | 0.0294 | 0.0958 | 0.1481 | 0.0684 | 0.0858 |

第一个 policy 结果相对原始 4x explicit-decoder augmentation baseline 是负向的，但现在更应解释为协议不一致的警告。当前 policy dataset 对 `history` 做语义 dropout 后再拼回原始尾部，而原始 baseline 是对完整训练序列做 dropout。下一步不应继续把当前 policy 结果作为最终结论，而应先将 `smb_policy_decoder` 修正为 full-sequence policy views，验证其数据统计与 `smb_explicit_decoder_4` 对齐后，再重新运行完整 policy family。Curriculum 会越过当前静态 cache 边界，需要 Dataset、sampler、Trainer、DDP 和 resume 行为协同修改，因此仍未实现并继续暂缓。
