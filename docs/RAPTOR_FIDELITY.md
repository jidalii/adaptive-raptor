# RAPTOR 实现一致性核对

日期：2026-11-11。范围：本仓库构建的 RAPTOR 树和检索，逐项对照两个来源：

- **论文**：Sarthi et al., ICLR 2024（arXiv 2401.18059 v1，HTML 版逐字核对）；
- **上游代码**：`parthsarthi03/raptor` 提交 `7da1d48`，即 vendored 的 `src/raptor/`。

结论先行：

- **算法代码与上游逐字节一致**。`scripts/check_environment.py` 报告 `src/raptor == upstream identical`。
- 论文和上游代码本身就有几处不一致，凡是论文没写明的，我们都以上游代码为准。
- 有意的偏差共 6 处，全部已记录在 `UPSTREAM.md`，原因见下表。
- 相关测试 18/18 通过（`tests/test_index_and_retrieval.py`），包括：
  - `test_fixed_chunks_match_upstream_split_text`：切块与上游 `split_text` 逐字节一致；
  - `test_collapsed_matches_upstream`：collapsed 检索结果与上游 `retrieve` 一致；
  - 同一 seed 跨进程建树可复现。

## 1. 逐项对照

| 环节 | 论文 | 上游代码 | 本仓库 | 判定 |
|---|---|---|---|---|
| 切块 | 100 token；"句子超出 100 token 的上限时，整句移到下一个 chunk" | `split_text(text, cl100k, 100)` | 调用同一个 `split_text`，chunk 大小 100，无重叠 | ✓ 一致（有测试） |
| 嵌入 | SBERT `multi-qa-mpnet-base-cos-v1` | `SBertEmbeddingModel` 可选；**`RetrievalAugmentation` 默认用 OpenAI ada** | SBERT `multi-qa-mpnet-base-cos-v1`，聚类和检索共用同一个模型 | ✓ 与论文一致 |
| 降维和聚类 | UMAP 加 GMM，先全局后局部，用 BIC 选簇数，软聚类，簇超过阈值就递归聚类 | `reduction_dimension=10`，`threshold=0.1`，`max_length_in_cluster=3500`，BIC 最多 50 簇 | 相同默认值，通过 `ClusterTreeConfig` 传入 | ✓ 一致 |
| UMAP 随机性 | 未说明 | 不传 `random_state`，并行运行，不同进程建出的树会不同 | 建树时传入 `random_state=seed`（224），UMAP 单线程运行 | **偏差 1**：为了可复现 |
| 停止建层 | "直到无法继续聚类" | 某层节点数 ≤ `reduction_dimension + 1` 就停止，最多 `num_layers=5` 层 | 相同 | ✓ 一致 |
| 摘要模型 | gpt-3.5-turbo | `GPT3TurboSummarizationModel` | DeepSeek `deepseek-flash`，`temperature=0`，显式关闭 thinking | **偏差 2**：模型不同，无法避免；关闭 thinking 是因为隐藏推理会把摘要输出清空 |
| 摘要 prompt | system 为 "You are a Summarizing Text Portal"；user 为 "Write a summary of the following, including as many key details as possible: {context}:" | system 为 "You are a helpful assistant."；user 相同 | 逐字使用上游代码的 prompt | 论文和代码不一致，我们跟随代码 |
| 摘要长度 | 未给上限。实测平均 131 token，QASPER 平均 145.9，NarrativeQA 平均 129.7 | `summarization_length=100` | 200 | **偏差 3**：上限设为 100 会截断每一条摘要，而论文报告的平均长度本身就超过 100，所以 200 更接近论文 |
| 叶子顺序 | 未说明 | 多线程建叶子时，顺序取决于哪个线程先完成 | 按 index 重新排序 | **偏差 4**：聚类输入顺序确定，内容不变 |
| 检索：collapsed | 主实验用 "collapsed tree，最多 2000 token，约相当于 top-20 节点" | `retrieve()` 默认 `top_k=10`，`max_tokens=3500` | `collapsed@2000`：`top_k` 设为全部节点，只受 2000 token 预算限制 | ✓ 与论文主设置一致。`top_k=10` 会把上下文限制在约 1,000–1,300 token，不符合"约 20 个节点" |
| 检索：`raptor_original` | — | 上游 `retrieve()` 原样调用 | `top_k=10`，`max_tokens=2000`，`collapse_tree=true` | 这是"上游代码默认行为"的基线，**不是论文的主设置**；论文主设置是 `collapsed@2000` |
| 装包边界 | 节点要满足 `total + size < max` 才加入 | 遇到 `total + size > max` 就停止 | 沿用上游 | 论文是 `<`，代码是 `>`，只在恰好等于上限时有差别，可忽略 |
| tree traversal | 每层保留 top-k，主实验没有用 | `retrieve_information`：使用配置里的 `top_k`，不受 token 预算限制 | 不在 catalog 中；P0b 用独立实现做了离线检查 | 不适用 |
| 拼接上下文 | 未说明 | `get_text`：每个节点内部的换行替换为空格，节点之间用 `\n\n` 分隔 | 节点之间用 `\n\n` 分隔，节点内部保留原有换行 | **偏差 5**（轻微）：只影响节点内部的换行 |
| 问答模型和 prompt | UnifiedQA-3B（400 token 上下文）、GPT-3、GPT-4；论文没有给出 QASPER 的 prompt | `GPT3TurboQAModel` 的 prompt 偏向多选题 | DeepSeek 加上数据集专用的简短回答 prompt | **偏差 6**：模型不同，无法避免 |
| 评测指标 | QASPER 用 F1；NarrativeQA 用 BLEU-1/4、ROUGE-L、METEOR（附录 H 说明修改过 AllenNLP 的评测脚本） | — | QASPER：与官方 evaluator 一致的 F1。NarrativeQA：ROUGE-L F1 和 token-F1，各取两个参考答案中的较高分；没有实现 BLEU 和 METEOR | NarrativeQA 的数值和论文**不能直接比较**，只能比较不同配置之间的相对差异 |

## 2. 和论文规模的比较

论文附录 I 列出了检索到的节点来自哪些层：
- NarrativeQA 的树有 0–3 层（SBERT、BM25）或 0–4 层（DPR）；
- QASPER 和 QuALITY 只有 0–2 层。

我们的 QASPER dev 树最高层分布为 L0/L1/L2 = 1/185/65 篇，与论文的 QASPER 一致。所以"QASPER 的树浅"不是实现的问题，而是数据集本身的特点。

论文表 2（UnifiedQA-3B 作为问答模型）里，QASPER 上 SBERT 有 RAPTOR 是 36.70 F1，没有 RAPTOR 是 36.23，只差 0.47；NarrativeQA 上 ROUGE-L 是 30.87 对 29.26，差 1.61。可见论文自己报告的 RAPTOR 在 QASPER 上的提升也很小，NarrativeQA 上的提升更明显一些。

## 3. 结论

- 树的构建和 collapsed 检索与上游代码一致，并且有测试保证。`collapsed@2000` 对应论文的主设置。
- 6 处偏差中，2 处是为了复现性（UMAP 的 seed、叶子顺序），1 处是为了忠实于论文（摘要长度上限 200），2 处是换了模型而无法避免（摘要模型、问答模型），1 处是轻微的格式差异（节点内部换行）。没有任何一处改变 RAPTOR 的算法本身。
- 因为问答模型、prompt 和评测指标都不同，这里的结果**不能和论文的绝对数值直接比较**。我们能做的比较是：在同一个模型和同一个指标下，RAPTOR 的层次检索相对 leaf 检索能带来多少提升。论文表 1 和表 2 报告的也是这种相对提升。
