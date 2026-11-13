# Tree-aware LLM optimizer 修改方案

日期：2026-11-11。状态：待实施设计；本文只定义修改，不代表已实现或已批准付费实验。

## 修订后的执行顺序

**P0 结构审计 → P0b 独立离线证据检查 → 通过后才启动 P1/P2 集成 → 获准后在长文档上做 P3。** 第 3–8 节是条件性工程设计，不是立即执行的重构清单。P0b 不通过则停止本轮投入，文档保留为未来工作；不据此宣称所有数据集上的 tree-aware retrieval 都无效。

当前最小交付仅需一个独立诊断脚本及报告，不改 pipeline、配置 schema 或 vendored RAPTOR。P0b 已获准离线实施；只允许独立脚本、相关测试与结果记录，不启动 P1/P2 集成。

## 1. 当前实现的准确结论

**当前 optimizer catalog 只有 collapsed 检索家族，三种策略的区别是候选节点范围。整个仓库并非没有 traversal。**

| 当前动作 | 候选节点 | 实际检索过程 |
|---|---|---|
| `leaf_only@B` | L0 | 对候选节点统一排序，按预算取前缀 |
| `summary_only@B` | 所有 L≥1 | 同上 |
| `collapsed@B` | 所有层 | 同上 |

`retrieval.py::retrieve()` 对以上三种动作均调用 `retrieve_information_collapse_tree()`。`_layer_view()` 已支持按层集合构建候选视图，但 `strategy_layers()` 只提供上述三个集合。

上游 `src/raptor/tree_retriever.py` 存在逐层检索；`retrieve_original()` 也允许通过 `raptor_original.collapse_tree=false` 进入该分支。但此入口是独立 baseline，不是 optimizer 可选择的 catalog 动作。

重要限制：上游 traversal 分支调用 `retrieve_information()`，不执行 `max_tokens` 限制；逐层选择使用 `TreeRetrieverConfig.top_k`，而不是 `retrieve(top_k=...)` 传入的值。不能只设置 `collapse_tree=false`，就宣称已经得到同预算 traversal。

当前 LLM 仅生成 `answer_form / granularity / n_facts / cross_section`。A1 用画像统计表估计可接受概率，A2 使用规则，`PlanSelector` 选择达到阈值的最低成本配置。LLM 没有看到实际树元数据，也没有直接生成计划。

## 2. 修改目标与研究边界

目标：检验“问题与实际树信息是否能预测不同检索计划的收益”。新增能力包括：

1. 任意明确层集合上的 collapsed 检索。
2. 有确定预算语义的 top-down traversal。
3. 将实际树元数据传给 planner，并输出可验证的有限计划 ID。
4. 保留旧策略、checkpoint 和历史结果的解释方式。

第一阶段不训练 RL，不迁移 PostgreSQL，不扩大到六个数据集，不重跑旧 QASPER router 搜索。PostgreSQL 是后续存储选项；多步 RL 是另一个研究阶段。旧 QASPER 的停止结论保持有效，新实验单独命名。

## 3. 统一计划表示

建议新增 `RetrievalPlan`，作为新动作的强类型表示；旧 `ConfigRef` 通过 adapter 转换，历史 ID 保持不变。

| 字段 | 约定 |
|---|---|
| `plan_id` | 稳定 ID；不能依赖自由文本名称区分执行语义 |
| `mode` | `collapsed` 或 `top_down` |
| `layer_selector` | collapsed 使用：`all`、`leaves`、`summaries` 或显式非空层集合 |
| `context_budget` | 最终序列化检索文本的 token 上限 |
| `start_layer` | traversal 使用：明确层号或 `highest` |
| `stop_layer` | traversal 终止层，首版固定为 L0 |
| `beam_width` | 每一层在整个当前候选集合中最多保留的节点数；不是每个父节点各保留 k 个 |
| `emit` | 首版 traversal 固定 `leaves`；摘要用于导航，不直接放入最终 context |
| `packing_version` | 上下文装包算法版本 |
| `executor_version` | 检索实现版本 |

区分 `requested_plan` 与按当前树解析后的 `resolved_plan`。记录实际层号、不可用原因与 fallback。显式请求不存在的层，不得悄悄忽略：离线 catalog 标为 unavailable，在线 planner 按预注册规则回退并记录。

新模式的语义、层集合、beam、输出规则、预算和算法版本都必须进入配置 hash。`[2,1]` 与 `[1,2]` 规范化为同一集合；不同模式不得共用 ID。

第一轮建议只有四个动作，在单一预算下比较：

- leaf collapsed；
- all-layer collapsed；
- 一个预先指定的 summary-layer subset，例如 `{L1,L2}`，适用条件固定；
- top-down，固定 beam，最终输出 leaves。

当最高摘要层只有 L2 时，`{L1,L2}` 等价于 `summary_only`，不是新的动作；只有 L1 时则不可用。必须报告按文档和按问题的有效动作覆盖率，去除等价动作，不能仅在深树子集上报告收益而省略全体结果。

此 catalog 是建议，不是已经选定的实验配置。先审计树深再冻结层集合；不要对所有层子集做穷举调参。旧 `summary_only` 保留为诊断 baseline。

## 4. 检索执行语义

### 4.1 层集合 collapsed

解析层集合 → 收集对应节点 → 按 query cosine similarity 排序 → 按预算装包。

需要明确：`all` 表示所有层参与检索，不是把全部节点都发送给生成器。`partial collapsed` 表示缩小候选层集合，仍然是统一排序，不沿树边遍历。

### 4.2 Top-down：首版采用摘要导航、叶子作证据

1. 从解析后的起始层收集候选节点。
2. 在摘要层用同一个 query embedding 排序，保留全局前 `beam_width` 个节点；L0 不再按 beam 截断。
3. 收集选中节点的 children，按 node ID 去重。
4. 在下一层重复，直到 L0；记录每步候选数、选中节点和边。
5. 将到达的叶子按 query similarity 重排，按最终预算装包。

必须处理共享子节点、空 frontier、单层树、非法边和重复节点。排序 ties 使用稳定 node ID。单层树可解析为 leaf 检索，但需记录退化行为。空 frontier 执行明确 fallback，不可把其他未到达节点补进结果而不记录。

首版不在遍历过程中调用 LLM，不改写 query，不动态学习停止策略。这样可以独立检验树导航的作用。

**该策略不同于上游原生 traversal。** 上游会累积沿途选中的摘要和叶子；本方案只输出到达的叶子。实现放在 `adaptive_raptor` adapter 内，保留 vendored RAPTOR 原样，并在 `UPSTREAM.md` 说明区别。如需评测原生 traversal，另设 baseline ID。

### 4.3 预算与公平比较

旧 collapsed 的装包规则是“首个放不下的节点出现就停止”，不是跳过大节点继续填充。旧执行路径保持此行为，避免历史结果改变。

新计划使用同一个、单独版本化的装包器：按排序逐个加入完整节点，若下一节点使序列化 context 超预算则停止；不截断节点。分隔符计入 context token。所有新固定 baseline 和新 router 使用同一装包器。

生成器额外包装、问题和输出 token 单独计费；context budget 不是总请求 token budget。新旧装包方式不同的结果不得直接混为同一配置。记录导航访问节点数和检索延迟，不能只比较最终 token 数就声称检索更快。

### 4.4 Beam 是额外的资源参数

beam 很小会使候选叶子不足、装不满预算；beam 足够大时可能退化为全叶子检索。必须记录实际 context tokens、预算利用率、到达叶子数、到达比例和与 leaf baseline 的节点重合率。

首版只比较预先固定的小集合（beam ∈ {1,2,4}）。若需要选择一个 beam，只能在 train 上按冻结的证据质量—实际 token 规则选择，ties 选更小 beam，然后在 dev 做一次诊断。只用 dev 缓存时，采用事先固定的单一 beam；展示多个 beam 的敏感性不等于独立验证，不能从 dev 挑最优值后报告为已验证方案。

主比较使用实际平均 context token 下的固定 leaf 策略混合前沿，不能比较相同名义预算就声称成本相同。诊断的前沿由证据指标构建，不得替代之后的答案 F1 前沿。混合概率按成本与冻结规则确定，不按每题 gold 选择动作；报告配对文档 bootstrap 不确定性。若候选成本超出 baseline 覆盖范围，应标记不可比或先补齐离线预算点，不能外推收益。

## 5. 树元数据：层号不等于语义粒度

沿用 `level=0` 为叶子，向上递增。不要把它命名为从根向下的 `depth`，也不要固定写成“L1=章节、L2=全文”。

### 5.1 离线、无 API 的元数据

为每个 index 保存版本化 sidecar，至少包含：

- `index_key_hash`、metadata schema version、tokenizer、构建来源；
- 各层节点数、文本 token 长度分布；
- 每个节点的 children、去重后的 descendant leaf IDs；
- 覆盖原文 token 数、压缩比例；注明重叠 chunk 可能重复计数；
- 原文 span / section 信息及其精度状态，若可用；
- 每层确定性选取的少量代表节点，明确选取规则和总 token 上限。

RAPTOR 允许共享子节点，应按 DAG 的边和去重后代计算覆盖，不能假设每个 child 只有一个 parent。

现有 builder 收到 `Chunk` 后只把文本传给上游；应保存 leaf-to-chunk provenance。已有缓存若缺少章节信息，允许标记 unknown；只有能按同版本 chunker 重建并验证文本一致时才回填。不得推断出虚假的精确 offsets。

先用现有文本和统计生成模板描述，例如“本层节点通常概括若干叶子，覆盖范围中位数为 X”。真正的语义描述必须基于节点内容；额外 LLM 描述属于另一个付费阶段。

### 5.2 在线输入

planner 只可读取 question、允许的文档元数据、上述树描述、可用动作及预算约束。禁止输入答案、gold evidence、最终 F1 或 oracle 动作。

代表节点首版固定选择；如果后来加入 query-dependent probe retrieval，则作为独立消融，计入延迟和成本。不要在一次实验中同时改变 metadata、检索动作和 probe，再把全部提升归因于画像。

### 5.3 成功预期与证据边界

基于现有 QASPER 结果，对“仅补充层规模、覆盖分布等静态 metadata”应采取低收益预期：增益可能接近零。E2 中 query-dependent 检索/边际上下文特征也未预测出有效收益；更弱的结构统计不能被默认视为解决方案。

但这不是跨数据集的零收益结论，也不能证明 LLM 利用问题与节点文本的交互必然失败。结构统计和代表节点文本应分开消融；语义文本与纯文档长度特征不是同一种输入。

## 6. LLM planner 与现有 router 并存

保留 `profile_v3`、A1/A2 作为旧 baseline。新建 `tree_plan_v1`，直接从经过可用性验证的有限 catalog 中返回：

```json
{"plan_id": "<one available plan ID>"}
```

LLM 只能选择已定义动作，不得输出任意 SQL、Python、层号或无限增长的预算。输出无效可有一次格式重试，仍失败使用冻结 fallback。首版不依赖 token logprob confidence。

新增 planner cache key 必须包含 question 内容、允许输入的文档内容 hash、index hash、metadata hash、prompt/version、model/settings、catalog hash。现有 `ProfileStore` 的 `(question_id,prompt_hash)` 不足以区分同一问题对应的不同树。

`QueryContext` 增加 tree metadata 和可用计划。当前 `ConfigPolicy.configs_for(question, embedding, document)` 没有 index 信息，应增加可选的结构化 retrieval context；`CatalogPolicy` 忽略不需要的字段，`RouterPolicy` 转发给 planner。

`pipeline.py` 当前只区分 `ConfigRef` 与字符串 baseline。新增 `RetrievalPlan` 后必须显式分派，不能让未知对象误入 `retrieve_original()`。policy trace 的键应包含 index/variant/seed 和 question，避免同题不同树的记录混用。

## 7. 修改文件清单（仅 P0b 通过后启用）

| 文件 | 修改职责 |
|---|---|
| `config/schema.py`、`config/validation.py` | 新计划字段、互斥约束、catalog 唯一性与版本 |
| `retrieval.py` | 层集合解析、top-down executor、公共装包器、执行轨迹 |
| `indexing/builder.py` | leaf provenance 与 metadata 生成入口 |
| `indexing/store.py` | sidecar 读写及版本校验；不强制重建树 |
| `indexing/metadata.py`（新增） | DAG 覆盖统计、代表节点与紧凑描述 |
| `optimizer/planner.py`（新增） | `tree_plan_v1`、解析、验证、缓存、fallback |
| `optimizer/__init__.py`、`policy.py` | context 传递及旧 router 兼容 |
| `pipeline.py` | 新计划分派与 index-aware policy 调用 |
| `evaluation/traces.py`、`profiling.py` | 计划/执行语义、labels 与完整成本 |
| `evaluation/research.py`、分析脚本 | 不再假设 config ID 恒为 `strategy@budget` |
| `experiments/runner.py` | 新 catalog 和 planner 实验入口 |
| `configs/experiments/` | 独立新配置；历史配置不覆盖 |
| `tests/`、`UPSTREAM.md` | 行为测试和与上游差异说明 |

优先在现有模块扩展，仅将 metadata 与 planner 各自独立为一个文件，避免创建大量单用途模块。

## 8. Trace、缓存和旧结果兼容

新增 trace：requested/resolved plan、可用动作、层集合、start/stop layer、beam、导航访问与选择节点、最终节点、fallback reason、metadata/planner hash、planner input/output tokens、planner/retrieval latency、执行版本。

保留历史 ID 和读取逻辑。新 checkpoint 使用独立格式版本；不自动把旧 A1 checkpoint 解释成新 planner。缓存必须区分树、动作语义、装包器和 metadata。旧 label 不得直接当作新 traversal 的监督标签。

新增 metadata 可从缓存树离线派生，写 sidecar；若树构建逻辑或摘要 prompt 未变，不应触发付费重建。

## 9. 分阶段实施与验收

### P0：离线结构审计

输出当前缓存树的层数、各层规模、覆盖分布、provenance 缺失率。验证是否存在可区分的层级，不把“多一层”当作语义有效的证明。

### P0b：独立离线 traversal 证据检查（新增投入门槛）

实施脚本：`scripts/check_tree_navigation.py`。读取现有可信缓存树、问题、QASPER gold evidence 和已存 trace；仅本地计算 embedding/similarity 和评估。不建新摘要，不调用 API，不下载模型或数据，不改 pipeline 与 vendored RAPTOR。运行前验证必需缓存齐全，缺少依赖则报告缺项并停止，不自动联网补齐。

`leaf_only@2000` trace 可用于核对基线排序，但单独的 trace 不足以定义任意预算的 baseline，也不包含完整 gold provenance。需要从缓存树重新得到完整 leaf ranking，复用同 tokenizer、序列化和装包规则，在实际成本覆盖范围内生成固定预算点。不得把新 traversal 结果套用旧 leaf trace 的生成答案或 F1。

本次冻结协议（先于结果）：

- QASPER 只测 top-down，不测 subset；在浅树上 `{L1,L2}` 可能不可用或等价于 summary_only，实际最大层号以 P0 审计为准。
- train 固定 `outputs/labels_train/20261109T195548Z`，dev 固定对应的 `labels_dev` run。每个预算分别从 beam {1,2,4} 中，按 train matched-cost highlighted recall 增量最大选择；精确并列取较小 beam。选定的 budget→beam 映射写入 frozen 文件后才读取 dev 结果。不把近似随机覆盖统计转化成“beam=1 必然装不满”或“2000 必须 beam=4”的硬规则。
- 主预算上限 1000，次要上限 2000；这不是实际消耗强制等于 1000/2000。比较点为各候选的实际平均 context tokens。
- 固定 leaf budget 网格为 0、100、200、400、600、800、1000、1200、1400、1600、1800、2000、2400；0 用于覆盖低成本端。两种方法采用相同序列化装包器。最佳固定混合是随机策略期望，不按题目 gold 挑动作。每个 split 的固定前沿作为事后较强 comparator，不能称为从 train 冻结的可部署基线；bootstrap 中重新计算该前沿与实际成本匹配。
- 每个点用 4000 次配对文档 bootstrap，97.5% 双侧 percentile CI（两个预算的 Bonferroni 校正）。只有主点增量>0 且 CI 下界>0 才触发后续投入；次要点即使显著也不挽救失败的主点。
- 直接 import `compression_feasibility.load_gold/evaluate`，复用其 highlighted recall、多标注者取最大及 extractive answer-in-context。该代码是内容词覆盖 proxy，并非精确 gold-to-leaf 标注；报告 all-leaf coverage 作为可达性检查，不能称其为精确匹配率。
- 用上游距离和排序函数，从完整缓存树重建 leaf ranking，按旧逐节点 token 装包复现 train/dev 全部 leaf@400/@1000/@2000 trace。任意不一致立即停止并输出错误，不能带差异继续选择或评估。
- 已有 utility feature cache 含本地 SBERT query embedding，可优先复用并通过上述逐题复现检验；不存在时停止，不自动调用模型或下载依赖。报告输入文件 hash 与冻结协议 hash。

检查内容：

1. 用独立实现执行第 4.2 节确定性 traversal，记录最终叶子、导航路径和实际 token。用小型人工 DAG 检查路径约束、共享叶子去重和预算后，再运行真实缓存。
2. 对相同问题生成 leaf baseline，在相同名义预算和实际平均成本匹配前沿下分别比较；前者描述容量，后者是主要成本比较。
3. 主指标为 highlighted-evidence recall，预先冻结与已有压缩分析一致的匹配规则、分母、多标注者合并方式，并记录匹配失败率。报告有非空 gold evidence 的问题覆盖率；无证据题单列，不把其 recall 记为 0 混入平均。
4. 同时报告 evidence precision 或非证据 token 比例、逐题改善/退化数、按树深分层结果与配对文档 bootstrap CI。保留同样 recall 但干扰更少的现象供解释，不能把其当作已经提升 F1。
5. `answer-in-context` 仅作为 extractive 答案的辅助 proxy；冻结归一化与多答案规则。yes/no、abstractive、unanswerable 不使用字符串出现率判断答案可答性。答案字符串存在也不是充分推理证据。
6. summary subset 的后代证据覆盖只衡量“导航能否到达证据”，不是摘要文本的 evidence recall。必须把 descendant coverage 和 summary-text evidence retention 分开；若无法可靠测后者，标记未评估，不能以高后代覆盖判定 summary-only 可用。

建议在运行前冻结的保守工程门槛：选定 traversal 在 matched-cost evidence recall 上超过 leaf 固定 mix，且配对文档 bootstrap 97.5% CI 下界大于 0，且为预定 1000-token 主点，才允许进入 P1/P2。点估计、CI、样本量和退化情况全部报告。门槛是课程项目控制投入的规则，不是答案质量提升的必要或充分条件；不通过可以停止投入，但不能写成“已证明 traversal 无用”。如果未来要采用“召回不劣、precision 提升”的替代门槛，必须事先定义，不在看到失败结果后补设。

QASPER dev 已被多次分析，本步骤只能称为已有设置上的探索性诊断，不是新的独立验证。QASPER 的负结果不排除更深长文档上的收益；若将来重新开启该方向，需要单独说明更换设置的理由与资源安排。

### P1：确定性检索能力（P0b 通过后）

实现新计划、subset collapsed 和 bounded top-down，不调用 planner API。用小型固定树验证：

- subset 仅返回允许层；
- 未展开分支的叶子不能出现在 traversal 结果；
- 共享后代不重复输出；
- 所有新策略遵守相同序列化 context budget；
- 空 frontier、浅树、缺层和超大单节点有明确行为；
- 旧配置的节点排序和输出保持兼容。

### P2：metadata 与 planner 接口（P0b 通过后）

通过 mock planner 验证 JSON、非法动作、fallback、缓存失效、不同 index 的 trace 隔离。此阶段不需要付费模型调用。

### P3：获准后的小规模研究验证

层选择/导航的主候选为 NarrativeQA 的全文设置，QuALITY 为备选；实际层数和语义覆盖仍需先审计，不能预设固定三层。按文档划分 train / calibration / evaluation，固定一个初始预算；补齐对应加载器和标准指标后再运行。MuSiQue 不纳入本实验：其复杂度路由正对照属于另一动作空间和研究问题。

先比较固定动作和交叉评分 oracle，检查动作差异是否稳定；同时独立报告预算选择与策略选择的 headroom。若 subset/traversal 没有稳定价值，停止扩大 planner。

随后比较：最佳固定动作、同成本固定 mix、question-only planner、question+结构统计 planner、question+结构统计+代表节点文本 planner，以及非 LLM 的 R-feature ridge router。后者沿用 E2 的方法，在新数据集 train 上拟合新动作收益，不能直接搬用旧 QASPER 权重；新增动作适用的特征集合事先冻结。LLM planner 除达到相对固定 mix 的门槛，还应在计入 planner 开销后优于该廉价 router，才支持 LLM 带来额外价值的论点。R 特征提取也记录实际计算成本，不称为绝对零成本。冻结 prompt、catalog、fallback、成本规则后再评估；重复用于选择的 dev 不再被描述为全新验证集。test 保留最终一次。

建议预注册 go/no-go：相对同成本固定 mix，主质量指标至少提高 2 个百分点且按文档配对 bootstrap 的 95% CI 下界大于 0。该门槛是建议，需在看新结果前确定；不同指标不能视为等价效应大小。

计费包括建索引、metadata、planner、probe（若有）和最终生成。索引成本按真实复用场景摊销，同时单列一次性成本。不要以历史 API 单价保证新费用；付费运行前单独估算和批准。

### P4：仅在 P3 通过后

评估监督式 action ranking / utility prediction 或 planner 蒸馏。只有引入“观察证据后继续选择展开/停止”的多步交互，才单独论证 RL 的必要性。一次选有限计划、且有全动作收益表时，不以 RL 作为默认方案。

## 10. PostgreSQL 后续方案

如系统规模确实需要持久查询服务，可使用 `indexes`、`nodes`、`edges`、`layer_metadata`。节点主键为 `(index_id,node_id)`，边表表达多父节点；保存 `level`、text、embedding、provenance 和版本。语义粒度描述独立于 level。

数据库 backend 必须与本地 executor 在相同距离、tie-break、过滤和预算规则下对齐；若启用近似向量检索，单独测召回和延迟变化。数据库迁移本身不作为 optimizer 有效性的证据。

## 11. 实施完成的定义

- 新计划能从 YAML 经 policy、executor 到 trace 完整运行，旧路径可复现。
- planner 看到的是当前树的实际信息，所有输出经过有限 catalog 校验。
- traversal 的导航和输出规则、预算及费用都有明确记录。
- 离线测试通过，不因补充 metadata 自动触发 API。
- 新实验结果与旧 QASPER 结果分开，正负结果均按预注册门槛报告。

本次修订采纳“先诊断、后集成”的评审意见，补充 beam 成本、指标局限、数据集范围和廉价 router 对照。未重新测量评审提到的树深比例；正式报告需由 P0 输出可复核统计。

本文依据 2026-11-11 的本地源码审阅；设计内容尚未实施。相关本地文件：`idea.md`、`NOTES.md`、`docs/OPTIMIZER_FLOW.md`、`src/adaptive_raptor/optimizer/`、`src/adaptive_raptor/retrieval.py`、`src/raptor/tree_retriever.py`。
