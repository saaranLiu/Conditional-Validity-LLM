# Predictor vs Sampler 假设检验：统计学原理、推导与实施规范

## 本次重评估实际结果与实现状态（2026-08-20）

正式实现统一先过滤 `schema_ok`，跨臂统计只在所需臂共同 ego 上重算。Network 主指标固定为 composition 字段的 `mean_tv_composition`，core 单列。N0 自身样本 N=1094：composition **0.5092**、core **0.2688**；N0–N3 四臂共同 N=846：composition **0.5073/0.4820/0.4532/0.2965**。T5 size/prior 四臂共同 N=307/289，interaction **0.0092/0.1223**。

T1 Eval2 为 153 pairs（Pearson **0.4478**、Spearman **0.6394**、overshoot **55.6%**）；Eval3 为 22 outcomes，Pearson **-0.1306**（p=0.562，不宣称显著）。T2 最新六字段 paired rho **-0.8857** 仅作探索性小样本结果；旧七字段及其他历史口径只作敏感性说明。T3 以 GSS/NSHAP 非退化靶量为主；UCNets 是 full-network bucket 对 core-roster target，estimand 不一致，只作探索性。T4 paired bootstrap：UCNets CI [-0.0053, 0.0159]、NSHAP [-0.0066, 0.0079] 跨 0，GSS 1985 [0.0081, 0.0320] 不跨 0；每个条件每 ego 仅一次生成，低个体相关不能识别 input-insensitivity。T6：mean Pearson **0.0671**、mean span ratio **0.3626**，family **0.6315**、friend **0.4499**。T5 只定位为工程可组合性。

## 0. 文档目的

本文档系统说明本项目中假设检验部分的统计对象、零假设、检验统计量、推导、抽样不确定性和推荐实施方式。它服务于以下研究问题：

> 给定个体信息 \(X\)，LLM 生成的状态或自我中心网络 \(Y_L\)，能否被视为人类条件分布 \(P_H(Y\mid X)\) 的校准样本？

需要首先区分三个层次：

1. **边际效度**：\(P_L(Y)\) 是否接近 \(P_H(Y)\)；
2. **条件效度**：\(P_L(Y\mid X)\) 是否接近 \(P_H(Y\mid X)\)；
3. **机制解释**：观察到的偏差是否由“默认模板”“信息忽略”“生成格式”等机制导致。

第一、二层可以通过统计检验直接评价。第三层通常只能通过专门设计的干预、重复生成或消融实验得到支持，不能仅凭一个相关系数被唯一识别。

本文对现有 T1–T6 做如下定位：

| 判据 | 主要统计目标 | 能否直接检验条件采样校准 |
|---|---|---|
| T1 关联结构 | 比较人类与 LLM 的联合分布结构 | 可以拒绝完全校准，但不能唯一识别 predictor |
| T2 信息使用 | 比较条件分布是否因给定信息而正确变化 | 需要改为直接条件校准；现有“相关性—TV 改善”分析仅为探索性 |
| T3 显式区间跟随 | 检验输出是否响应明确给定的条件 | 当前最直接的条件响应检验 |
| T4 State 替换 | 分离输入效应与生成随机性 | 当前缺少同输入重复生成，尚不能完成识别 |
| T5 干预组合 | 检验信息与格式效果是否存在交互 | 属于工程稳健性，不是 sampler 的必要条件 |
| T6 跨总体响应 | 检验国家、年份或人群差异是否被复现 | 可以检验总体响应校准，但需处理跨数据差异 |

---

## 1. 记号与统计对象

对个体 \(i=1,\ldots,n\)，定义：

- \(X_i\)：已观测条件，如年龄、性别、教育、婚姻、国家或给定的目标分箱；
- \(Y_i^H\)：人类调查中的状态或网络统计量；
- \(Y_{ir}^L\)：在条件 \(X_i\) 下，第 \(r\) 次 LLM 生成结果；
- \(P_H(Y\mid X)\)：人类条件分布；
- \(P_L(Y\mid X)\)：LLM 所诱导的条件分布；
- \(S_H\)、\(S_L\)：真实 State 与 LLM 生成 State；
- \(N_H\)、\(N_L\)：真实 Network 与 LLM 生成 Network。

理想的条件采样校准定义为：

\[
H_{\mathrm{cal}}:\quad P_L(Y\mid X=x)=P_H(Y\mid X=x)
\quad \text{对研究支持集中的所有 }x.
\]

实际数据不可能证明两个高维分布完全相等。因此实验检验的是它的一组必要条件：

\[
\theta_j(P_L)=\theta_j(P_H),\qquad j=1,\ldots,J,
\]

其中 \(\theta_j\) 可以是：

- 类别概率；
- 条件均值、条件方差；
- 关联强度；
- 回归解释力；
- 组间跨度；
- 跨总体差异。

如果某个必要条件被明确拒绝，就可以拒绝“在该统计量、该数据集和该 prompt 条件下完全校准”。但不能自动推出唯一的生成机制。

---

## 2. 假设检验的基本原则

### 2.1 差异检验、等效性检验与校准检验

传统差异检验写作：

\[
H_0:\theta_L=\theta_H,\qquad
H_1:\theta_L\neq\theta_H.
\]

若 \(p<\alpha\)，可以拒绝相等。若 \(p\geq\alpha\)，只能说“没有足够证据拒绝相等”，不能说二者已经相等。

本项目常常真正关心的是“误差是否小到可以接受”。这更适合**等效性检验**：

\[
H_0:|\theta_L-\theta_H|\geq \delta,\qquad
H_1:|\theta_L-\theta_H|<\delta,
\]

其中 \(\delta\) 是事先规定的科学容忍阈值。可采用 two one-sided tests（TOST）：

\[
\begin{aligned}
H_{01}&:\theta_L-\theta_H\leq-\delta,\\
H_{02}&:\theta_L-\theta_H\geq \delta.
\end{aligned}
\]

只有两个单侧零假设都被拒绝，才可以声明在 \(\delta\) 范围内等效。等价地，如果差异的 \(100(1-2\alpha)\%\) 置信区间完全落入 \((-\delta,\delta)\)，即可判为等效。

因此论文中建议使用三分类结论：

1. **不等效**：区间完全或部分超出容忍范围，并有证据表明偏差较大；
2. **等效**：TOST 通过；
3. **不确定**：既不能拒绝相等，也不能证明等效。

### 2.2 统计显著性与实际显著性

样本量很大时，小偏差也可能显著；样本量很小时，大偏差也可能不显著。因此每个结果必须同时报告：

- 点估计；
- 置信区间；
- 效应量；
- 样本量；
- 科学容忍阈值；
- 经多重比较校正后的 \(p\) 值（若使用显著性检验）。

论文不能在只有点估计时使用“显著”“不显著”“不存在”等措辞。应写成“点估计显示”“观察到”“未发现证据”。

### 2.3 分析单位必须是 ego

同一个 ego 的多个 alter、多个字段和多次生成不是相互独立的。bootstrap 或置换必须尽量以 ego 为重抽样单位：

\[
\{i_1^*,\ldots,i_n^*\}
\sim \text{sample with replacement from }\{1,\ldots,n\}.
\]

每次抽到 ego \(i\) 时，应整体带走其：

- 人类真值；
- 所有 alter；
- 所有实验臂；
- 所有重复生成。

这样可以保留 ego 内部相关性和实验臂配对结构。

### 2.4 配对设计优先

如果同一个 ego 在多个实验臂中出现，应对每个 ego 计算差值：

\[
D_i=m(Y_{i,\mathrm{treat}}^L,Y_i^H)
-m(Y_{i,\mathrm{base}}^L,Y_i^H),
\]

再检验：

\[
H_0:E[D_i]=0.
\]

配对设计消除了 ego 组成差异，通常比比较两个独立总体均值更有统计效率。

---

## 3. 基础统计量的定义与推导

## 3.1 总变差距离（Total Variation Distance）

对离散变量 \(Y\) 的两个分布 \(P,Q\)，定义：

\[
\mathrm{TV}(P,Q)
=\frac12\sum_{k\in\mathcal K}|P(Y=k)-Q(Y=k)|.
\]

性质：

\[
0\leq \mathrm{TV}(P,Q)\leq1.
\]

TV 还等于两个分布在所有事件上的最大概率差：

\[
\mathrm{TV}(P,Q)
=\sup_{A\subseteq\mathcal K}|P(A)-Q(A)|.
\]

因此 TV=0 表示边际分布完全相同，TV=1 表示分布支持集完全分离。

对连续字段，本项目先按固定切点分箱，再计算经验分布：

\[
\widehat p_k=\frac{1}{n_H}\sum_i\mathbf 1(Y_i^H\in B_k),\qquad
\widehat q_k=\frac{1}{n_L}\sum_i\mathbf 1(Y_i^L\in B_k).
\]

于是：

\[
\widehat{\mathrm{TV}}
=\frac12\sum_k|\widehat p_k-\widehat q_k|.
\]

注意：

1. TV 依赖分箱方案；
2. TV 是边际指标，不直接测量 \(P(Y\mid X)\)；
3. 总体 TV 很小不代表同一个人的输出正确；
4. TV 改善不一定与条件变量的解释力单调相关，因为它同时受基线误差和可改善空间影响。

推荐用 ego bootstrap 得到：

\[
\widehat{\mathrm{SE}}_{\mathrm{boot}}(\widehat{\mathrm{TV}})
=\sqrt{\frac{1}{B-1}
\sum_{b=1}^{B}
\left(\widehat{\mathrm{TV}}^{*(b)}
-\overline{\mathrm{TV}}^*\right)^2}.
\]

### 3.2 熵与熵比

Shannon 熵定义为：

\[
H(P)=-\sum_k p_k\log p_k.
\]

熵比：

\[
R_H=\frac{H(P_L)}{H(P_H)}.
\]

解释：

- \(R_H<1\)：LLM 分布相对压缩；
- \(R_H>1\)：LLM 分布相对铺开；
- \(R_H=1\)：两者熵相同。

但熵相同不代表分布相同。例如 \((0.9,0.1)\) 与 \((0.1,0.9)\) 熵完全相同。因此熵比只能与 TV/JSD 联合使用。

### 3.3 Jensen–Shannon 距离

令：

\[
M=\frac12(P+Q).
\]

Jensen–Shannon divergence 为：

\[
\mathrm{JSD}(P,Q)
=\frac12\mathrm{KL}(P\|M)
+\frac12\mathrm{KL}(Q\|M),
\]

其中：

\[
\mathrm{KL}(P\|M)=\sum_k p_k\log\frac{p_k}{m_k}.
\]

如果使用以 2 为底的对数，则：

\[
0\leq\mathrm{JSD}(P,Q)\leq1.
\]

常把 \(\sqrt{\mathrm{JSD}}\) 称为 Jensen–Shannon distance；它满足度量性质。

### 3.4 Cramér's \(V\)

对两个类别变量的列联表，先计算 Pearson 卡方统计量：

\[
\chi^2
=\sum_{a=1}^{r}\sum_{b=1}^{c}
\frac{(O_{ab}-E_{ab})^2}{E_{ab}},
\]

其中：

\[
E_{ab}
=\frac{O_{a+}O_{+b}}{n}.
\]

Cramér's \(V\) 定义为：

\[
V=\sqrt{\frac{\chi^2}{n\min(r-1,c-1)}}.
\]

其值域为 \([0,1]\)，用于比较类别变量间的关联强度。

当前代码 `evaluate_state_eval2.py` 采用：

\[
|\widehat V_H-\widehat V_L|\leq0.10
\]

作为通过规则。该规则本质上是科学容忍阈值，而不是传统显著性检验。更规范的方式是对差值做 ego bootstrap，并实施：

\[
H_0:|V_L-V_H|\geq\delta_V,\qquad
H_1:|V_L-V_H|<\delta_V.
\]

### 3.5 \(\eta^2\)：组间方差占比

对类别分组 \(G\) 和数值结果 \(Y\)，总平方和为：

\[
\mathrm{SS}_{\mathrm{total}}
=\sum_{i=1}^{n}(Y_i-\bar Y)^2.
\]

组间平方和为：

\[
\mathrm{SS}_{\mathrm{between}}
=\sum_{g=1}^{G}n_g(\bar Y_g-\bar Y)^2.
\]

定义：

\[
\eta^2
=\frac{\mathrm{SS}_{\mathrm{between}}}
{\mathrm{SS}_{\mathrm{total}}}.
\]

因为：

\[
\mathrm{SS}_{\mathrm{total}}
=\mathrm{SS}_{\mathrm{between}}
+\mathrm{SS}_{\mathrm{within}},
\]

所以 \(\eta^2\in[0,1]\)，可解释为分组变量解释的样本方差比例。

需要注意：

- \(\eta^2\) 是描述性效应量；
- 样本内 \(\eta^2\) 有向上偏差；
- 分组数不同的 \(\eta^2\) 不应无条件直接比较；
- 比较 \(\eta_H^2\) 与 \(\eta_L^2\) 时应保持同一分组定义和同一分析样本。

推荐检验：

\[
\Delta_{\eta^2}=\eta_L^2-\eta_H^2,
\]

并以 ego bootstrap 估计 \(\Delta_{\eta^2}\) 的置信区间。

### 3.6 Pearson 相关与 Fisher \(z\) 变换

样本 Pearson 相关系数：

\[
r
=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_i(x_i-\bar x)^2}
\sqrt{\sum_i(y_i-\bar y)^2}}.
\]

当数据近似二元正态时，Fisher 变换：

\[
z(r)=\operatorname{atanh}(r)
=\frac12\log\frac{1+r}{1-r}
\]

近似满足：

\[
z(r)\sim N\left(
\operatorname{atanh}(\rho),
\frac{1}{n-3}
\right).
\]

若比较两个独立相关系数：

\[
Z
=\frac{z(r_1)-z(r_2)}
{\sqrt{\frac1{n_1-3}+\frac1{n_2-3}}}.
\]

当前 `evaluate_state_eval2.py` 使用这一公式比较人类与 LLM 的 numeric–numeric 相关。但两组相关系数来自同一批 ego，并共享变量结构，严格来说并非独立。上述标准误会忽略配对相关性。

推荐改用 ego bootstrap：

1. 重抽 ego；
2. 同时计算 \(r_H^*\) 与 \(r_L^*\)；
3. 保存 \(z(r_L^*)-z(r_H^*)\)；
4. 用其经验分布构造区间。

### 3.7 Spearman 等级相关

Spearman 相关是秩变量的 Pearson 相关：

\[
\rho_s
=\operatorname{Corr}(\operatorname{rank}(X),
\operatorname{rank}(Y)).
\]

无并列值时也可写为：

\[
\rho_s
=1-\frac{6\sum_i d_i^2}{n(n^2-1)}.
\]

它检验的是单调关系，而不是 \(Y=X\) 的校准关系。即使 \(\rho_s=1\)，也可能存在：

\[
Y=10X,
\]

这在排序上完美，但并未校准。因此 T1 中必须同时报告：

- 45 度线偏差；
- 截距与斜率；
- Pearson/Spearman；
- 绝对差异；
- 不确定性。

### 3.8 OLS \(R^2\)

线性模型：

\[
Y=X\beta+\varepsilon.
\]

最小二乘估计：

\[
\widehat\beta=(X^\top X)^{-1}X^\top Y.
\]

定义：

\[
R^2
=1-\frac{\sum_i(Y_i-\widehat Y_i)^2}
{\sum_i(Y_i-\bar Y)^2}.
\]

本项目对相同人口学锚点 \(X\)，分别拟合人类结果 \(Y_H\) 与 LLM 结果 \(Y_L\)，得到：

\[
R_H^2,\qquad R_L^2.
\]

差异：

\[
\Delta R^2=R_L^2-R_H^2
\]

衡量 LLM 是否制造了过强或过弱的人口学决定性。

注意当前代码使用样本内 \(R^2\)。类别水平较多时会向上偏。推荐使用：

- 调整 \(R^2\)；
- 交叉验证 \(R^2\)；
- 或 bootstrap optimism correction。

### 3.9 McFadden 伪 \(R^2\)

对二分类 logistic 模型，模型对数似然为：

\[
\ell_{\mathrm{model}}
=\sum_i
\left[
y_i\log p_i
+(1-y_i)\log(1-p_i)
\right].
\]

仅含截距的模型对数似然为 \(\ell_{\mathrm{null}}\)。McFadden 伪 \(R^2\) 定义为：

\[
R^2_{\mathrm{McF}}
=1-\frac{\ell_{\mathrm{model}}}
{\ell_{\mathrm{null}}}.
\]

当前实现把“人类数据中的众数类别”编码为 1，其余类别编码为 0，并对人类与 LLM 使用相同正类定义。这保证了编码可比，但也把多分类结构压缩为“众数 vs 其余”。

因此：

- 不同 outcome 的伪 \(R^2\) 可用于粗略比较；
- 不宜把它解释成普通 OLS 的方差解释比例；
- 可以补充 multinomial log-loss 或 cross-validated deviance。

### 3.10 Cohen's \(\kappa\) 的正确解释

对人类标签与模型标签的一致性：

\[
\kappa
=\frac{p_o-p_e}{1-p_e},
\]

其中：

- \(p_o\)：实际一致率；
- \(p_e\)：由两个边际分布计算的机会一致率。

重要限制：

> 条件 sampler 不应被要求与人类的一次观察 realization 高度一致。

即使：

\[
Y_H,Y_L\overset{\mathrm{iid}}{\sim}P(Y\mid X),
\]

二者仍可能不同。给定 \(X\) 后的理论一致率是：

\[
P(Y_H=Y_L\mid X)
=\sum_kP(Y=k\mid X)^2,
\]

通常小于 1。

因此，低 \(\kappa\) 可以说明“LLM 不是准确的个体点预测器”，但不能单独否定“LLM 是条件 sampler”。若要用 agreement 评价 sampler，需要：

- 人类重复测量基线；
- 或已知 \(P_H(Y\mid X)\) 后推导 sampler 的期望一致率；
- 或改用 proper scoring rule 评价条件概率预测。

---

## 4. T1：关联结构校准

## 4.1 统计问题

T1 问的是：

> 人类数据中存在的变量关联，是否在 LLM 生成数据中以相近强度被复现？

设第 \(j\) 个变量对的关联统计量为：

\[
\theta_{Hj},\qquad \theta_{Lj}.
\]

完全校准要求：

\[
\theta_{Lj}=\theta_{Hj},
\qquad j=1,\ldots,J.
\]

对应统计假设：

\[
H_{0j}:|\theta_{Lj}-\theta_{Hj}|\geq\delta_j,
\qquad
H_{1j}:|\theta_{Lj}-\theta_{Hj}|<\delta_j.
\]

这是等效性形式。若只做差异检验，则：

\[
H_{0j}:\theta_{Lj}=\theta_{Hj}.
\]

### 4.2 跨变量对校准回归

可进一步拟合：

\[
\widehat\theta_{Lj}
=\alpha+\beta\widehat\theta_{Hj}+u_j.
\]

理想校准要求：

\[
\alpha=0,\qquad \beta=1.
\]

三个不同问题分别对应：

- \(\alpha\neq0\)：整体偏移；
- \(\beta\neq1\)：信号被放大或压缩；
- \(R^2\) 或相关低：不同变量对的偏差不一致。

单独报告 Pearson/Spearman 不足以证明校准，因为高相关不等于落在 45 度线上。

### 4.3 当前项目实现

当前分析汇总 Cramér's \(V\)、\(|r|\) 与 \(\eta^2\)，计算：

- 人类与 LLM 关联强度的 Pearson；
- Spearman；
- OLS 斜率；
- 平均 inflation；
- LLM 超过人类的比例。

混合不同统计量时取绝对值可以统一为“关联强度”，但会丢失 Pearson 相关的方向。建议正文中：

1. 主图按统计量类型分面；
2. signed Pearson 单独报告；
3. 不同 task/dataset 用层级模型或 cluster bootstrap；
4. 避免把 153 个变量对当作完全独立样本。

### 4.4 推荐 bootstrap

对每个 dataset × task：

1. 按 ego 重抽；
2. 重算所有 \(\theta_{Hj}^{*(b)}\)、\(\theta_{Lj}^{*(b)}\)；
3. 重算校准回归 \((\alpha^{*(b)},\beta^{*(b)})\)；
4. 得到每个 \(\Delta_j^{*(b)}\)；
5. 对变量对层面的 \(p\) 值做 Benjamini–Hochberg 校正。

稳健结论应写为：

> LLM 生成数据拒绝了关联结构的完全校准；偏差在题型与条件之间高度异质。该结果与默认模板或刻板关联解释一致，但不唯一识别其生成机制。

---

## 5. T2：条件信息是否被正确利用

## 5.1 现有分析为什么只能是探索性的

现有分析将：

\[
x_j=\eta_H^2(Y_j\sim B)
\]

与：

\[
g_j
=\mathrm{TV}_{\mathrm{baseline},j}
-\mathrm{TV}_{\mathrm{conditioned},j}
\]

做 Spearman 相关。

这个分析有直觉意义，但“\(\eta_H^2\) 越大，边际 TV 改善必然越大”不是校准 sampler 的数学必要条件。原因包括：

1. 不同字段基线误差不同；
2. 可改善空间（headroom）不同；
3. TV 与条件信息量之间不是线性关系；
4. 分箱方式不同；
5. 条件变量可能改善条件结构，但边际分布几乎不动。

例如 `size_bucket` 与 `num_alters` 高度相关，但如果 N0 的 `num_alters` 边际已经不算太差，N1 的边际 TV 改善仍可能很小；与此同时，它的条件一致率可以从 0.819 提升到 0.999。

### 5.2 更直接的条件校准检验

令 \(B\) 表示给定的 bucket。对每个字段 \(Y_j\) 和 bucket \(b\)，比较：

\[
P_H(Y_j\mid B=b)
\quad\text{与}\quad
P_L(Y_j\mid B=b).
\]

定义条件 TV：

\[
\mathrm{CTV}_j
=\sum_b w_b
\mathrm{TV}
\left(
P_H(Y_j\mid B=b),
P_L(Y_j\mid B=b)
\right),
\]

其中：

\[
w_b=P_H(B=b)
\]

或使用共同分析样本中的经验权重。

对应假设：

\[
H_0:\mathrm{CTV}_j\geq\delta_j,
\qquad
H_1:\mathrm{CTV}_j<\delta_j.
\]

还可以分解为：

\[
\Delta_{jb}
=E_L[Y_j\mid B=b]
-E_H[Y_j\mid B=b].
\]

对类别字段比较条件概率，对连续字段比较条件均值、方差和分位数。

### 5.3 交互回归表达

把人类与 LLM 数据合并，引入来源变量：

\[
Z=
\begin{cases}
0,&\text{Human},\\
1,&\text{LLM}.
\end{cases}
\]

拟合：

\[
Y
=\alpha+\beta B+\gamma Z+\delta(B\times Z)+\varepsilon.
\]

其中：

- \(\gamma\)：总体位置偏差；
- \(\delta\)：条件响应斜率是否被 LLM 改变。

关键检验为：

\[
H_0:\delta=0.
\]

如果 \(Y\) 是类别变量，可使用 logistic/multinomial 模型。

### 5.4 当前数字口径

当前正式与历史敏感性口径如下：

- 最新正式六字段 paired 产物得到 \(\rho=-0.8857\)，但只有 6 个字段，作为探索性结果；
- 历史七字段及旧脚本口径（曾报告 \(-0.286\)、\(-0.771\)）保留为敏感性说明，不作为当前主结果。

正式写作必须：

1. 明确字段纳入规则；
2. 以一个口径为主；
3. 同时报告另一口径作为敏感性分析；
4. 不把小样本、非显著相关写成“强烈负相关”的确定结论。

建议结论：

> 现有探索性分析未发现“真实相关性越强、边际 TV 改善越大”的稳定证据；更直接的条件分布校准检验仍需补充。

---

## 6. T3：显式区间跟随

## 6.1 设计逻辑

对每个 ego，真实目标量 \(T_i^H\) 被离散为：

\[
B_i\in\{\mathrm{low},\mathrm{medium},\mathrm{high}\}.
\]

该 bucket 被明确写入 prompt。LLM 生成网络后，重新从输出计算目标量：

\[
T_i^L.
\]

如果模型使用了这个条件，则：

\[
E[T_L\mid B=\mathrm{high}]
>
E[T_L\mid B=\mathrm{medium}]
>
E[T_L\mid B=\mathrm{low}],
\]

并且组间差异应与人类侧具有相近量级。

### 6.2 组间跨度

定义：

\[
\mathrm{span}_H
=\max_b\bar T_b^H-\min_b\bar T_b^H,
\]

\[
\mathrm{span}_L
=\max_b\bar T_b^L-\min_b\bar T_b^L.
\]

响应压缩比：

\[
C_{\mathrm{span}}
=\frac{\mathrm{span}_L}{\mathrm{span}_H}.
\]

理想校准下：

\[
C_{\mathrm{span}}\approx1.
\]

固定模板或弱响应下：

\[
C_{\mathrm{span}}\approx0.
\]

但跨度只使用最大、最小两组，可能对极端样本均值敏感。应同时报告：

- 每个 bucket 的均值及 CI；
- \(\eta_L^2\) 与 \(\eta_H^2\)；
- 组别×来源交互；
- 单调趋势检验。

### 6.3 来源×bucket 交互检验

把 Human 与 LLM 结果堆叠：

\[
T_{is}
=\alpha
+\sum_b\beta_b\mathbf 1(B_i=b)
+\gamma Z_s
+\sum_b\delta_b
\mathbf 1(B_i=b)Z_s
+u_i+\varepsilon_{is},
\]

其中：

- \(s\in\{H,L\}\)；
- \(u_i\) 是 ego 随机截距或配对固定效应；
- \(\delta_b\) 表示 LLM 与人类的 bucket 响应差。

联合零假设：

\[
H_0:\delta_{\mathrm{medium}}
=\delta_{\mathrm{high}}=0.
\]

拒绝该假设表示 LLM 的条件响应曲线与人类不同。

如果目标量在 \([0,1]\)，可考虑：

- fractional logit；
- beta regression；
- ego bootstrap 的非参数均值差。

### 6.4 单调趋势

把 bucket 编码为 \(0,1,2\)，拟合：

\[
T_i^L=\alpha+\beta_L B_i+\varepsilon_i.
\]

检验：

\[
H_0:\beta_L=0,\qquad H_1:\beta_L>0.
\]

但 \(\beta_L>0\) 只说明方向上有弱跟随，不代表幅度校准。必须同时比较：

\[
\frac{\beta_L}{\beta_H}
\]

或组间跨度比。

### 6.5 分箱有效性

当前项目规定任一 bucket 占比低于 15% 时视为退化。这是合理的设计诊断，因为极不平衡的 bucket 会导致：

- 组均值方差很大；
- 模型忽略小组时总体指标几乎不受影响；
- 单调趋势受单个小组支配。

UCNets 同种族 bucket 有 90.6% 位于 high，只能作为失败示例。GSS 1985/2004、NSHAP 的非退化闭合度/同性别结果应作为主证据。

### 6.6 推荐 bootstrap

对每个 dataset × target：

1. 在三个 bucket 内按 ego 分层 bootstrap，保持组样本量；
2. 每次重算三个组均值；
3. 重算 span、span ratio、\(\eta^2\) 和来源×bucket 交互；
4. 报告 percentile 或 BCa 置信区间；
5. 对多个 target 的交互检验做 FDR。

T3 可支持的准确结论是：

> 对已测试的显式条件，LLM 输出虽然部分呈现正确排序，但条件响应幅度远小于人类数据，拒绝了这些目标量上的完全条件校准。

---

## 7. T4：State 替换、个体响应与随机性分解

## 7.1 当前设计

当前比较：

\[
R0_i=N_L(S_i^H),
\]

\[
R1_i=N_L(S_i^L).
\]

现有统计量包括：

- R0、R1 相对 \(N_H\) 的 mean TV；
- \(A=\mathrm{TV}(R1,N_H)-\mathrm{TV}(R0,N_H)\)；
- 同一 ego 的 R0/R1 Pearson；
- size bucket 一致率；
- 平均绝对差。

### 7.2 为什么低 R0/R1 相关不能直接证明输入不敏感

R0 和 R1 是两次独立生成。即使模型是正确 sampler：

\[
R0_i\sim P_H(N\mid S_i^H),
\qquad
R1_i\sim P_H(N\mid S_i^L),
\]

单次 realization 仍可能差异很大。

观察到的差异可以分解为：

\[
\underbrace{R0_i-R1_i}_{\text{总差异}}
=
\underbrace{
E[R0_i\mid S_i^H]
-E[R1_i\mid S_i^L]
}_{\text{输入变化造成的系统差异}}
+
\underbrace{\varepsilon_{i0}-\varepsilon_{i1}}_{\text{生成随机性}}.
\]

只有一轮 R0/R1 时，两个部分不可分离。

### 7.3 必需的重复生成设计

对每个 ego、每种输入重复生成 \(K\) 次：

\[
Y_{icr}^L,\qquad
c\in\{S_H,S_L\},
\quad r=1,\ldots,K.
\]

推荐至少 \(K=5\)，理想情况下 \(K=10\) 或更多。

定义同输入距离：

\[
D_{\mathrm{within}}
=E\left[
d(Y_{icr}^L,Y_{icr'}^L)
\mid r\neq r'
\right].
\]

定义换输入距离：

\[
D_{\mathrm{between}}
=E\left[
d(Y_{i,S_H,r}^L,Y_{i,S_L,r'}^L)
\right].
\]

关键效应：

\[
\Delta_D
=D_{\mathrm{between}}
-D_{\mathrm{within}}.
\]

检验：

\[
H_0:\Delta_D=0,
\qquad
H_1:\Delta_D>0.
\]

如果 \(\Delta_D\approx0\)，说明换输入带来的变化没有超过模型自身重抽噪声；这才是“输入不敏感”的直接证据。

### 7.4 方差分解

对连续网络统计量可拟合：

\[
Y_{icr}
=\mu+\alpha_i+\beta_c
+(\alpha\beta)_{ic}
+\varepsilon_{icr}.
\]

方差分解为：

\[
\operatorname{Var}(Y)
=\sigma_{\mathrm{ego}}^2
+\sigma_{\mathrm{condition}}^2
+\sigma_{\mathrm{ego\times condition}}^2
+\sigma_{\mathrm{generation}}^2.
\]

输入敏感性可以用：

\[
\mathrm{ICC}_{\mathrm{input}}
=\frac{
\sigma_{\mathrm{ego}}^2
+\sigma_{\mathrm{ego\times condition}}^2
}{
\sigma_{\mathrm{total}}^2
}
\]

或按研究目的报告 condition 解释的方差比例。

### 7.5 当前数据可以支持什么

在没有重复生成前，当前 T4 稳健支持：

> 用 LLM State 替换真实 State 后，网络总体边际误差几乎没有变化；在当前 N0 协议下，修复 State 不是降低 network-level mean TV 的高杠杆手段。

当前数据尚不能单独支持：

> 个体输出不响应输入，因此模型必然是 predictor。

---

## 8. T5：信息 × 格式交互

## 8.1 现有代数分解

设损失 \(L\) 越低越好：

- \(L_{00}\)：基线 N0；
- \(L_{10}\)：仅信息；
- \(L_{01}\)：仅格式；
- \(L_{11}\)：信息+格式。

现有可加预测为：

\[
\widehat L_{11}^{\mathrm{add}}
=L_{00}
 +(L_{10}-L_{00})
 +(L_{01}-L_{00}),
\]

即：

\[
\widehat L_{11}^{\mathrm{add}}
=L_{10}+L_{01}-L_{00}.
\]

交互项：

\[
I
=L_{11}
-\widehat L_{11}^{\mathrm{add}}
=L_{11}-L_{10}-L_{01}+L_{00}.
\]

如果损失越低越好：

- \(I>0\)：组合比加性预测更差，表现为抵消或反噬；
- \(I<0\)：组合优于加性预测；
- \(I=0\)：在该损失尺度上可加。

### 8.2 为什么可加性不是 sampler 的必要条件

即使模型正确采样，信息与格式仍可能在真实生成过程中交互；此外 TV 是非线性函数：

\[
\mathrm{TV}(P,Q)
=\frac12\sum_k|p_k-q_k|,
\]

其改善量不具有天然线性可加性。因此：

\[
I\neq0
\]

只能说明“干预在该指标尺度上存在交互”，不能单独区分 sampler 与 predictor。

### 8.3 正确的 factorial 分析

对 ego-level 损失 \(L_i\) 拟合：

\[
L_i
=\alpha+\beta_I I_i+\beta_F F_i
+\beta_{IF}I_iF_i+\varepsilon_i.
\]

关键检验：

\[
H_0:\beta_{IF}=0.
\]

由于四个臂来自相同 ego，应使用：

- ego 固定效应；
- cluster-robust SE；
- 或配对 ego bootstrap。

如果只有总体 mean TV，而没有可加的 ego-level TV，需要在每次 ego bootstrap 中重算四臂总体 TV 和交互：

\[
I^{*(b)}
=L_{11}^{*(b)}
-L_{10}^{*(b)}
-L_{01}^{*(b)}
+L_{00}^{*(b)}.
\]

报告 \(I\) 的 CI，而不是仅报告点估计。

### 8.4 推荐定位

T5 应命名为：

> 干预可组合性与生成协议稳健性检验。

它支持：

> 在 oneshot 中有效的信息干预不能直接外推到多步 roster 格式；格式会改变干预效果。

它不直接支持：

> 交互不为零，所以模型不是 sampler。

---

## 9. T6：跨总体响应

## 9.1 ISSP 跨国校准

对国家 \(c=1,\ldots,C\) 和字段 \(j\)，定义：

\[
\mu_{Hcj}=E_H[Y_j\mid C=c],
\]

\[
\mu_{Lcj}=E_L[Y_j\mid C=c].
\]

理想校准要求：

\[
\mu_{Lcj}=\mu_{Hcj}.
\]

可拟合国家均值校准回归：

\[
\widehat\mu_{Lcj}
=\alpha_j+\beta_j\widehat\mu_{Hcj}+u_{cj}.
\]

理想状态：

\[
\alpha_j=0,\qquad \beta_j=1.
\]

Pearson 相关只检验国家排序是否近似一致，不检验幅度。跨度比：

\[
C_j
=\frac{
\max_c\widehat\mu_{Lcj}
-\min_c\widehat\mu_{Lcj}
}{
\max_c\widehat\mu_{Hcj}
-\min_c\widehat\mu_{Hcj}
}
\]

衡量跨国异质性是否被压缩。

### 9.2 不确定性的两层来源

ISSP 同时存在：

1. 国家内个体抽样误差；
2. 国家间异质性估计误差。

推荐 hierarchical bootstrap：

1. 保持国家集合固定；
2. 每个国家内按 ego 重抽；
3. 重算各国均值；
4. 重算 \(\alpha_j,\beta_j,r_j,C_j\)；
5. 如果研究目标是外推到国家总体，可再对国家做第二层 bootstrap。

### 9.3 GSS 跨年差异

对字段 \(j\)：

\[
\Delta_{Hj}
=\mu_{H,2004,j}-\mu_{H,1985,j},
\]

\[
\Delta_{Lj}
=\mu_{L,2004,j}-\mu_{L,1985,j}.
\]

响应压缩：

\[
C_j
=\frac{|\Delta_{Lj}|}{|\Delta_{Hj}|}.
\]

方向一致：

\[
\operatorname{sign}(\Delta_{Lj})
=\operatorname{sign}(\Delta_{Hj}).
\]

应进一步检验 difference-in-differences：

\[
D_j
=\Delta_{Lj}-\Delta_{Hj}.
\]

零假设：

\[
H_0:D_j=0.
\]

由于 1985 与 2004 是独立样本，可分别在年份内 bootstrap。

### 9.4 跨 survey 比较限制

NSHAP 与 UCNets 同时改变了：

- 年龄总体；
- 地理总体；
- 名单上限；
- name generator；
- 网络定义。

因此观察到的差异不能全部归因于“人群标签”。该比较应定位为描述性外部稳健性，而不是干净的因果或条件响应检验。

T6 推荐结论：

> 在 ISSP 跨国和 GSS 跨年比较中，LLM 对多数总体差异的响应幅度明显小于人类数据；少数字段保持了正确方向。跨 survey 结果仅作描述性补充。

---

## 10. 多重比较

项目包含：

- 多个 dataset；
- 多个 task；
- 多个字段；
- 多个变量对；
- 多个 T3 target；
- 多个 T6 国家字段。

若对每项都使用 \(\alpha=0.05\)，至少出现一个假阳性的概率会快速增加。

推荐使用 Benjamini–Hochberg（BH）控制 FDR。对排序后的 \(p\) 值：

\[
p_{(1)}\leq p_{(2)}\leq\cdots\leq p_{(m)},
\]

寻找最大 \(k\)，满足：

\[
p_{(k)}
\leq\frac{k}{m}q.
\]

则拒绝：

\[
H_{(1)},\ldots,H_{(k)}.
\]

建议按“同一个理论家族”分组校正：

- T1 pairwise association 为一族；
- T3 target 交互为一族；
- T6 ISSP 字段为一族；
- T5 多指标交互为一族。

不建议把所有论文检验混成一个超大 FDR 家族，也不建议完全不校正。

---

## 11. 缺失与 schema failure

评估只使用 schema_ok 的模型输出。如果 schema failure 与 ego 特征或目标量相关，则 complete-case 分析会产生选择偏差。

定义：

\[
R_i=
\begin{cases}
1,&\text{输出 schema 合格},\\
0,&\text{输出失败}.
\end{cases}
\]

至少应检验：

\[
P(R=1\mid X,Y_H).
\]

可以拟合：

\[
\operatorname{logit}P(R_i=1)
=\alpha+X_i^\top\beta+Y_i^{H\top}\gamma.
\]

如果 \(\beta\) 或 \(\gamma\) 明显不为 0，则失败并非完全随机。推荐：

1. 报告各臂 schema_ok 比例；
2. 比较纳入/排除 ego 的人类 gold 分布；
3. 做 inverse probability weighting 敏感性分析；
4. 在配对实验中只用共同 schema_ok 交集，同时报告样本损失。

---

## 12. 推荐的统一 bootstrap 流程

对大多数统计量，可采用以下统一流程：

### 输入

- 共同 ego ID；
- Human gold；
- 所有相关实验臂；
- 分层变量（dataset、bucket、country）；
- bootstrap 次数 \(B\)，建议至少 1000，主结果建议 2000–5000。

### 算法

对 \(b=1,\ldots,B\)：

1. 按 ego 重抽；
2. 保留同一个 ego 的所有臂和所有 alter；
3. 对 T3 按 bucket 分层重抽；
4. 对 ISSP 在国家内重抽；
5. 重算目标统计量；
6. 保存臂间差、交互、跨度比或校准斜率。

### 区间

简单 percentile interval：

\[
\mathrm{CI}_{1-\alpha}
=\left[
Q_{\alpha/2}(\theta^*),
Q_{1-\alpha/2}(\theta^*)
\right].
\]

如果统计量偏斜明显，可使用 BCa interval。

### 配对差

所有臂间比较都应在同一次 bootstrap 中使用同一组重抽 ego，从而保持配对。

---

## 13. 推荐的正文报告模板

每条主结果建议按以下顺序报告：

1. **Estimand**：检验的统计对象；
2. **Hypothesis**：零假设与备择；
3. **Statistic**：点估计；
4. **Uncertainty**：95% CI；
5. **Multiplicity**：原始与 FDR-adjusted \(p\)；
6. **Effect size**：与科学阈值比较；
7. **Scope**：可支持与不可支持的解释。

示例：

> 在 GSS 1985 情感闭合度实验中，人类三个条件桶的均值跨度为 0.756，LLM 为 0.032，跨度比为 0.042。按 ego 分层 bootstrap 的 95% CI 为 \([\,\cdot,\cdot\,]\)。来源×bucket 交互拒绝完全条件校准。该结果说明 llama3.1 8B 对该显式条件的响应幅度严重压缩，但不唯一识别造成压缩的内部机制。

---

## 14. T1–T6 最终统计定位

### T1

可检验：

\[
\theta_L=\theta_H
\]

或等效性范围。

可得结论：

> 关联结构是否校准。

不能单独得到：

> 偏差一定来自 predictor。

### T2

推荐检验：

\[
P_L(Y\mid B)=P_H(Y\mid B).
\]

现有 relevance–TV gain 相关只作探索性结果。

### T3

推荐检验：

\[
\text{Source}\times\text{Bucket}=0.
\]

以及：

\[
\mathrm{span}_L/\mathrm{span}_H\approx1.
\]

这是当前最直接的条件响应校准检验。

### T4

推荐在重复生成后检验：

\[
D_{\mathrm{between}}
-D_{\mathrm{within}}>0.
\]

当前只宜作总体误差归因。

### T5

检验：

\[
I=L_{11}-L_{10}-L_{01}+L_{00}=0.
\]

它描述干预交互，不是 sampler 的必要条件。

### T6

检验：

\[
\alpha=0,\quad\beta=1
\]

的总体均值校准回归，以及：

\[
\Delta_L-\Delta_H=0.
\]

---

## 15. 与仓库实现的对应关系

| 内容 | 文件 |
|---|---|
| State pairwise association、Cramér's V、\(\eta^2\)、Fisher \(z\) | `Testing/scripts/evaluate_state_eval2.py` |
| OLS \(R^2\)、McFadden pseudo-\(R^2\) | `Testing/scripts/evaluate_state_eval3.py` |
| T1 跨任务关联校准、T2 六字段分析 | `Reports/cross_phase_analysis/analysis_a_info_sufficiency.py` |
| T2 七字段敏感性结果 | `output/conditioning_leverage_summary.json` |
| T3 分箱跟随 | `Testing/T3_interval_following/{dataset}/eval/t3_interval_following.csv` |
| T4 R0/R1 评估 | `Testing/scripts/evaluate_t4.py` |
| T5 信息×格式交互 | `Reports/cross_phase_analysis/analysis_c_info_format_interaction.py` |
| T6 跨国家、年份、人群 | `Testing/scripts/evaluate_t6.py` |
| 论文当前结果 | `Paper_LLM_Ego_Network_Estimand_Validity.md` |

---

## 16. 总结

本项目的统计问题本质上不是“LLM 的某个平均数是否接近人类”，而是：

\[
P_L(Y\mid X)
\stackrel{?}{\approx}
P_H(Y\mid X).
\]

边际 TV、熵比和总体均值只能评价这一问题的一部分。最有识别力的设计应直接操纵或分层 \(X\)，比较完整的条件响应曲线，并通过 ego-level bootstrap 给出不确定性。

基于当前设计：

- T1 可以检验关联结构校准；
- T3 可以直接检验显式条件响应；
- T6 可以检验跨总体响应压缩；
- T2 需要改为条件分布校准；
- T4 需要同输入重复生成以分离输入效应与生成噪声；
- T5 应作为干预交互和工程稳健性，而非 sampler/predictor 的识别条件。

因此最严谨的总体表述是：

> 现有实验在多个数据集和目标量上发现 LLM 生成分布的边际、关联和条件响应未达到人类数据的校准水平，其中显式条件下的响应幅度压缩是最直接的证据。部分现象与默认模板式生成一致，但在缺少重复生成和更强机制对照时，不应把所有偏差唯一归因于 predictor。
