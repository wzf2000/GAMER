# GAMER Candidate-level CVR 样本与打分机制

本文重点说明 GAMER Ranking Decoder 中三个直接影响实验解释的问题：Candidate-level CVR 样本如何构造，candidate item 如何进入 GAMER，以及 prediction head 如何从 backbone 表示得到最终分数。当前实现本质上是“生成式 backbone 加判别式 ranking head”的候选级二分类模型，而不是 behavior token 生成模型。

**1. Candidate-level CVR 样本如何构造？** 数据首先按用户的 session 顺序确定预测目标。对于至少包含三个 session 的用户，倒数第三个 session 用于训练，倒数第二个 session 用于验证，最后一个 session 用于测试；目标 session 之前的交互构成用户历史。训练脚本默认设置 `max_his_len=100`，因此最多保留最近 100 条历史交互。

训练和验证阶段以“用户、目标 session 中的一条交互记录”为一条 candidate-level 样本。同一目标 session 内的所有交互共享相同的用户历史，但每条交互的 item 分别作为 candidate。标签由该条交互记录的 behavior 决定：ShortVideoAD 中最高层级行为 `cvr` 对应正例 `ranking_label=1`，`p3s` 和 `click` 对应负例 `ranking_label=0`。

这里需要严格区分“非转化交互”和“未转化 item”。当前代码按交互记录赋标签，并不先按 item 聚合。如果同一个 item 在目标 session 中先出现 `click`、之后又出现 `cvr`，代码会构造一条负样本和一条正样本，而不是把该 item 统一标记为已转化。因此，将负例描述为“曝光但未转化的 item”只是一种近似的业务解释；更准确的定义是“目标 session 中 behavior 不是最高层级的真实交互记录”。

负例不是从全 item 库随机抽取的，也不存在固定的负采样比例。目标 session 中所有真实的 `p3s`、`click` 和 `cvr` 记录都会进入数据集，所以负正比例由数据的自然分布决定。训练开始时，程序会统计完整训练集中的正例数和负例数，并设置 `pos_weight = negative_count / positive_count`，通过加权 `BCEWithLogitsLoss` 提高 CVR 正例在损失中的贡献。这里的 `pos_weight` 是损失权重，不是负采样比例，也不会改变样本数量。

测试数据在缓存中以 session 为单位保存，评估时再将最后一个 session 内的交互逐条展开为候选并打分。这种存储方式避免了在缓存中为每个候选重复保存相同历史，但最终仍采用与训练一致的候选级 0/1 标签定义。样本划分和标签构造的具体实现位于 [`ranking.py`](../SeqRec/datasets/session_behavior/ranking.py)。

**2. Candidate item 如何输入 GAMER？** Candidate 与历史共同构成一条输入序列。代码中的核心操作是 `input_ids = history_tokens + base_behavior_token + candidate_item_tokens`。历史中的每条交互由真实 behavior token 和 item semantic-ID tokens 组成；candidate 不携带真实 behavior，而是统一加入最低层级行为作为已知条件。ShortVideoAD 的 CTR 和 CVR 都使用 `<behavior_p3s>`。

因此，当前实现不是先用一个独立编码器得到 candidate embedding，再与预先计算的 user embedding 组合，也不是双塔结构。历史和 candidate 只执行一次共享的 backbone forward。由于 candidate 位于序列末尾并使用因果注意力，candidate tokens 可以关注此前的完整用户历史，其 hidden states 已经是融合用户上下文后的候选表示；历史 tokens 则无法看到后续 candidate。

Candidate query token 和全部 semantic-ID tokens 的 `relation_actions` 都设置为 `p3s` 对应的 action index 1。CTR/CVR 的目标行为只用于构造标签阈值，不改变 candidate 输入，因此相同 candidate 在两个任务中的 backbone 输入保持一致。

虽然历史和 candidate 在同一条序列中完成编码，prediction head 仍会显式提取并组合历史侧与候选侧表示。这种“先联合编码，再在 head 中组合”的方式不同于“历史与候选分别独立编码后再匹配”：前者的 candidate hidden states 在进入 head 之前已经与用户历史发生了多层 attention 交互。

**3. Prediction head 使用什么表示，结构是什么，训练哪些参数？** 当前任务使用 `llm_pair` 打分模式。`history_state` 取 candidate 起始位置前一个 token 的最后一层 hidden state，用于表示 candidate 出现之前的用户历史；`candidate_state` 取 `<behavior_p3s>` 和 candidate 全部 semantic-ID tokens 的最后一层 hidden states 均值。这里使用的是上下文化 hidden states 的 mean pooling，不是原始 token embeddings 的平均。

模型随后计算 `history_state * candidate_state` 作为逐元素交叉特征，并将 `history_state`、`candidate_state` 和交叉特征拼接为一个 `3H` 维向量。当前配置关闭了额外的 user embedding，即 `ranking_use_user_embedding=False`，所以 prediction head 的输入只包含上述三部分，不再拼接独立的 user-ID 表示。

Prediction head 不是单个线性层，而是一个两层 MLP。其结构为 `Dropout -> Linear(3H, H) -> PReLU -> Linear(H, 1)`，最终输出一个标量 `logit`。执行候选排序时可以直接比较 logit；如果需要解释为概率，则使用 `sigmoid(logit)`。由于 sigmoid 单调递增，两种分数形式产生相同的候选排序。

训练时并非只更新 prediction head，而是联合微调整个 `Qwen3TemporalHierarchical` backbone 和 ranking head。代码没有冻结 backbone，CVR 的加权二分类损失会同时更新上下文编码参数、行为层级注意力参数和 MLP 参数。最终保存的是包含 backbone 与 ranking head 的完整 checkpoint，不存在另一个消费导出特征的独立下游排序模型。

**4. 不同 CVR baseline 如何处理相同样本？** MeanPooling、DIN、DIENCVR、BSTCVR、HSTUCVR 和 SASRecCVR 由 `train_SMB_rec.py` 中的判别式训练路径管理。它们使用相同的 `SMBDINDataset`，因此 session 划分、历史截断、candidate-level 标签和“不做随机负采样”的规则与前文一致。不同之处在于，这些模型接收的是整数 item ID 序列和单个 `candidate_item` ID，而不是 GAMER 使用的 semantic-ID token 序列。batch 中同时提供历史行为编号，但是否实际使用该字段取决于具体模型。

这些 baseline 共享同一套数据协议。原始 item ID 在进入模型前统一加 1，使 0 可以作为 padding ID；历史 behavior 编号同样加 1，使 0 表示 padding。`DINCollator` 将不同长度的历史右侧补零，并生成 `inputs`、`behaviors`、`seq_len`、`candidate_item`、`label`、`behavior` 和 `uid`。其中 `inputs` 与 `behaviors` 的形状为 `[batch_size, sequence_length]`，`seq_len`、`candidate_item`、`label`、`behavior` 和 `uid` 的形状为 `[batch_size]`。`label` 是浮点型 CVR 二分类标签，`uid` 不参与模型打分，主要用于计算 GAUC 等用户分组指标。

这些 baseline 也共享统一的模型构造协议。`train_SMB_rec.py` 根据 `backbone` 名称加载对应的 `Config` 和模型类，并统一传入 `config`、`n_items`、`n_users`、`max_his_len`、`target_behavior_id` 和 `n_behaviors`。具体模型可以只显式声明自己需要的参数，并通过 `**kwargs` 接收其余字段。因而新增一个 binary-CVR baseline 时，通常需要提供模型类、配置类和配置目录，并将模型注册到 `train_SMB_rec.py` 的导入列表及 `is_binary_cvr` 集合中。

从训练器视角看，稳定的模型接口不是各自的 `forward()` 签名，而是 `calculate_loss(interaction)` 和 `predict(interaction)`。`calculate_loss` 接收完整 batch 字典并返回一个标量 loss，供共享 `SMBRec.Trainer` 执行反向传播；`predict` 接收相同字典并返回形状为 `[batch_size]` 的未归一化 logits，供验证和测试累计指标。模型内部可以自由决定是平均池化、GRU、Transformer、HSTU，还是 candidate-aware attention，也可以选择是否使用 `behaviors` 和 `seq_len`，只要对外满足这两个接口即可。SASRecCVR 的内部 `forward()` 与其他模型不同，但它通过 `calculate_loss` 和 `predict` 适配了同一外部协议。

这些 baseline 共用 `SMBRec.Trainer` 完成优化、验证、early stopping 和 checkpoint 保存。训练器会将 batch 中的全部 tensor 移到同一设备，调用 `calculate_loss` 更新模型全部可训练参数，并在验证阶段调用 `predict`。当前脚本统一请求 AUC、PRAUC、LogLoss、Accuracy、Precision、Recall、F1 和 GAUC，相关统计由同一个 `BinaryMetricAccumulator` 完成；指标列表中的最后一项作为 checkpoint 选择和 early stopping 的主指标，因此当前脚本将 GAUC 放在最后。最优模型保存为 `best_model.pth`，按 epoch 保留的模型保存为 `epoch_N_model.pth`，测试阶段再按 `ckpt_num` 选择并恢复对应参数。

因此，所谓 base 的共通协议可以概括为：使用同一 candidate-level 数据集和 ID 编码规则，接收同一 batch 字典，通过 `calculate_loss` 输出训练损失，通过 `predict` 输出每个候选的 scalar logit，并复用同一训练器、checkpoint 规则和二分类评估器。各 base 的差异被限制在“如何把历史与 candidate 转换为 logit”这一模型内部过程。

Ranking Decoder 与上述判别式 baseline 只共享任务语义层协议，即相同的 session 划分原则、candidate-level CVR 标签、无随机负采样设置以及 binary-CVR 指标定义；它不共享相同的 tensor 和训练器接口。Ranking Decoder 使用 `SMBRankingDatasetForDecoder` 与 `DecoderOnlyRankingCollator`，batch 主要包含 tokenized `input_ids`、`attention_mask`、`ranking_labels`、`relation_actions` 和 session 位置字段，模型返回 Transformers 风格的 `loss` 与 `logits`，训练和 checkpoint 管理由生成式训练路径负责。因此，判别式 base 可以在 `train_SMB_rec.py` 的统一协议内互换，而 Ranking Decoder 是同一任务定义下的另一套实现协议，不能直接作为该接口中的一个普通 `backbone` 替换项。

**MeanPooling** 将历史 item ID 映射为 embedding，对所有非 padding 的历史 item embeddings 做简单平均，得到 `user_interest`；candidate 则通过同一张 item embedding 表独立编码。最终分数是 `user_interest` 与 `candidate_embedding` 的点积。该模型不使用历史行为编号，没有 attention、序列演化模块或额外 MLP，是当前 baseline 中最直接的静态兴趣汇总方法。

**DIN** 同样分别查表得到历史 item embeddings 和 candidate embedding，但不会对历史直接平均。模型将每个历史位置的 `history_embedding`、`candidate_embedding`、二者乘积以及二者差值拼接，通过一个小型 attention network 计算 candidate-aware 权重，再对历史进行加权求和。得到的 `user_interest` 与 candidate embedding 及二者乘积共同输入 MLP，输出最终 logit。当前 DIN 实现接收 `behavior_seq` 参数但并未将其用于前向计算。

**DIENCVR** 首先使用 GRU 对历史 item embedding 序列建模，得到随时间演化的 `evolution_states`，然后将 candidate embedding 投影到 GRU hidden size。模型以 candidate 为条件，对各时刻 evolution state 计算 attention，加权得到最终 `user_interest`，再将 `user_interest`、candidate state 和二者乘积输入 MLP。与 DIN 相比，它先通过 GRU 表示兴趣演化；与完整论文版 DIEN 相比，当前代码是较简化的“GRU 加 target-aware attention”实现，并未使用历史行为编号。

**BSTCVR** 会将 candidate item ID 直接追加到历史 item ID 序列末尾，再加入 position embedding，随后将整条“历史加 candidate”序列送入双向 Transformer。模型取最后一个位置，也就是 candidate 位置的 Transformer 输出，通过单个线性层得到 logit。它在输入组织上与 Ranking Decoder 都属于 candidate 拼接式结构，但 BSTCVR 使用普通 item-ID embedding、不加入显式 behavior token，并采用双向 attention；Ranking Decoder 使用 semantic-ID tokens、显式历史 behavior 和 Temporal-Hierarchical causal attention。

**HSTUCVR** 只将历史序列送入 HSTU layers。历史输入由 item embedding、position embedding 和 behavior embedding 相加构成，当前配置启用 `use_behavior_embedding=true`，因此它是这些判别式 baseline 中实际利用历史 behavior 编号的模型。经过多层 gated attention 后，模型取最后一个有效历史位置作为 `user_state`，candidate 仍通过 item embedding 表单独编码，最终以 `user_state` 和 `candidate_embedding` 的点积作为 logit。Candidate 不会追加进 HSTU 历史序列，也不存在额外的 MLP prediction head。

**SASRecCVR** 使用带因果 mask 的 SASRec Transformer 编码历史 item ID 序列，取最后一个有效历史位置作为序列表示，再与独立编码的 candidate embedding 做点积。它与 HSTUCVR 都采用“先编码历史、再与 candidate 点积”的结构，但 SASRecCVR 不使用 behavior embedding，历史编码器也是标准 Transformer。该模型已接入 `train_SMB_rec.py` 的 binary-CVR 分发逻辑，不过当前仓库没有与其他五个模型对应的独立 `train_SMB_SASRecCVR.sh` 启动脚本。

上述判别式 baseline 均直接对 scalar label 使用各模型内部的 `BCEWithLogitsLoss`。与 Ranking Decoder 不同，它们当前没有根据训练集正负数量设置 `pos_weight`。因此，虽然这些模型共用相同样本和测试指标，但 Ranking Decoder 与判别式 baseline 在 item 表示、candidate 交互方式、历史 behavior 使用方式以及类别不平衡处理上都存在差异，实验比较时不能只用“更换 backbone”来概括。

当前打开的 [`HSTUCVR_item_id/result-smb_din.json`](../results/ShortVideoAD/smb_din/HSTUCVR_item_id/result-smb_din.json) 对应上述 HSTUCVR 判别式路径。该结果中的分数来自“带 behavior 的 HSTU 历史表示与 candidate item-ID embedding 点积”，而不是 Ranking Decoder 的“candidate semantic-ID tokens 拼接到历史后，再由 `llm_pair` MLP 打分”。两类结果使用相同的 candidate-level CVR 标签定义，但模型计算路径不同。

综上，当前 Ranking Decoder 使用目标 session 内的真实交互构造 candidate-level CVR 样本，不做随机负采样；`<behavior_p3s>` 和 candidate semantic-ID tokens 被拼接在带 behavior 的用户历史之后，并通过同一个 GAMER backbone 联合编码；prediction head 使用最后一层的 `history_state`、candidate hidden-state mean pooling 以及二者的逐元素乘积，通过两层 MLP 输出候选 CVR 分数，同时对完整 backbone 和 head 进行端到端微调。Prediction head 及损失的实现位于 [`wrappers.py`](../SeqRec/models/generative/common/wrappers.py)，训练配置和正例权重统计位于 [`train_SMB_ranking_decoder.py`](../SeqRec/tasks/training/train_SMB_ranking_decoder.py)；判别式 baseline 的共享样本位于 [`session_behavior.py`](../SeqRec/datasets/discriminative/session_behavior.py)，各模型实现位于 [`discriminative`](../SeqRec/models/discriminative) 目录。
