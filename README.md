# federated-rdd：无损联邦断点回归（精确 RD 与模糊 RD）

**English summary.** A Python implementation of federated regression discontinuity design (RDD). Several data holders (hospitals, regions, firms) estimate a treatment effect at a cutoff together without pooling individual records: each site uploads only kernel-weighted sums (Gram and "meat" matrices, counts), and a sponsor solves the normal equations. With double-double summation the federated estimate is **bit-for-bit identical** to the estimate on the pooled data. Supports sharp and fuzzy RD, CCT (MSE-optimal) and IK bandwidths, robust bias-corrected confidence intervals, covariates, clustered variance, site fixed effects and a federated McCrary density test. Results agree with the official `rdrobust` 2.1.1 to < 1e-10.

---

## 解决什么问题

断点回归（RDD）是评估政策效果的常用方法：比如年满 60 岁才能领养老金，就可以比较 60 岁两侧的人群来估计养老金的效果。实际中数据往往分散在多家机构（多家医院、多个地区的统计部门、多家企业），出于隐私和合规不能汇总到一起。

这个库让各机构**不交出个体数据**也能共同完成断点回归：

- 每家机构（站点）只上传求和后的统计量，例如核加权的 Gram 矩阵、残差的夹心矩阵和计数；
- 协调方（sponsor）只看到所有站点的总和（模拟安全聚合），据此解正规方程、算带宽和置信区间；
- 联邦估计与把数据合并后的集中式估计**逐位相同**，不是近似。

## 方法概要

| 环节 | 做法 |
|---|---|
| 无损聚合 | 站点对每条记录的贡献做双双精度（double-double）求和，以两个 float64 上传，sponsor 用同一方法合并，联邦与集中式结果舍入到同一个 double |
| 带宽选择 | CCT MSE 最优带宽（逐步转写 rdrobust 的 `rdbwselect(mserd)`，方差改用可联邦的 HC 形式）；IK (2012) 带宽 |
| 推断 | 稳健偏差校正置信区间（CCT 2014）；HC0–HC3、站点内嵌套聚类、按公共网格聚类 |
| 扩展 | 协变量调整（CCFT 2019）、模糊 RD（delta 方法）、断点边际效应（deriv = 1）、离散驱动变量的质点调整、站点固定效应、小单元抑制规则 |
| 诊断 | 联邦 McCrary (2008) 密度检验 |
| 通信 | CCT 全流程 10 轮，每站上传量与站点样本量无关（仿真中约 6–47 KB） |
| 威胁模型 | sponsor 诚实但好奇，站点之间不合谋；驱动变量的取值范围用公开的支撑边界，站点不上报自己的最小、最大值 |

## 目录结构

```
federated-rdd/
├─ fsrd/                  核心库
│  ├─ federation.py       站点 Site、聚合通道 Federation、通信账本 Ledger、双双精度求和
│  ├─ local.py            站点端计算（只返回求和后的统计量）
│  ├─ lprd.py             sponsor 端：正规方程、偏差校正、HC/聚类夹心方差、模糊 RD
│  ├─ bandwidth.py        联邦 CCT（mserd）带宽与 IK 带宽
│  ├─ density.py          联邦 McCrary 密度检验
│  ├─ api.py              fsrd() / centralized() / site_estimates() / meta_ivw() / meta_mean()
│  ├─ kernels.py          核函数与边界核常数
│  └─ reference.py        仅供测试：rdrobust 估计步骤的观测级逐行转写，作为独立对照
├─ examples/quickstart.py 模拟数据上的最小示例
├─ tests/                 单元测试（test_fsrd.py）与无需 pytest 的运行器（run_tests.py）
├─ simulation/            蒙特卡洛仿真：数据生成（dgp.py）、仿真、通信基准、表图
├─ charls/                CHARLS 实证案例：数据构建、分析、合作价值分析、多重检验、表图
├─ reporting.py           表格输出（CSV / Markdown / LaTeX）
├─ run_all.py             一键运行
└─ requirements.txt
```

运行后会生成 `data/`、`results/`、`output/`，均已在 `.gitignore` 中排除。

## 环境

Python 3.12（3.10 及以上应可运行），依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
# 可选：与官方 rdrobust 逐项对照的测试需要
pip install rdrobust
```

## 快速开始

```bash
python examples/quickstart.py
```

在 10 个站点的模拟数据上分别跑精确 RD（真实效应 1）和模糊 RD（真实效应 2），比较联邦估计、集中式估计和各站点单独估计后的逆方差加权平均。输出示例：

```
Sharp RD, 10 sites, true effect 1.0
  federated   : tau = 1.0078  robust 95% CI [0.9882, 1.0274]  h = 0.2896
  centralized : tau = 1.0078  robust 95% CI [0.9882, 1.0274]  h = 0.2896
  IVW of per-site fits: tau = 1.0087  (sites used: 10)
  max |federated - centralized| = 0.0e+00
  communication: 10 rounds, at most 16.1 KB uploaded per site
```

在自己的数据上使用：

```python
from fsrd import Site, fsrd, centralized

# 每个站点一份数据：x 为已按断点中心化的驱动变量 (X - c)，y 为结局，d 为处理接受情况（模糊 RD 时提供），w 为协变量
sites = [Site(x_k, y_k, d=d_k, w=w_k) for (x_k, y_k, d_k, w_k) in site_data]
res = fsrd(sites, bw='mserd', support=(lo, hi))   # support：驱动变量公开的取值范围
print(res['tau_bc'], res['ci_rb'], res['h'], res['comm'])
```

## 测试

```bash
python tests/run_tests.py
```

测试覆盖：无噪声数据上精确还原真值；联邦与集中式逐位相同（精确/模糊 RD、有无协变量、各种方差选项、随机切分站点）；与 rdrobust 观测级转写及官方 `rdrobust` 2.1.1 一致（给定带宽时差 < 1e-10，mserd 带宽相对差 < 1e-8）；IK 带宽公式；密度检验；站点固定效应；通信量与站点样本量无关。

`test_c_published_rdrobust_senate_example` 需要 rdrobust 自带的参议院示例数据 `rdrobust_senate.dta`（可从 rdrobust 的官方仓库获取），放到 `tests/data/` 或用环境变量 `RDROBUST_SENATE_DTA` 指定路径；没有这个文件时自动跳过。

## 仿真

```bash
python simulation/run_simulation.py --reps 500     # 6 类设计、36 个配置，20 核约 40 分钟
python simulation/run_comm.py                      # 通信量与运行时间
python simulation/make_sim_tables.py               # 表格与图
```

设计包括：基准线性设计、带协变量和方差异质性的设计、站点效应异质且样本量不等、Lee (2008) 曲率设计、二次曲率设计、站点构成在断点处跳跃。比较对象为集中式估计、各站点估计的逆方差加权平均和简单平均。

## CHARLS 实证案例与数据获取

`charls/` 用中国健康与养老追踪调查（CHARLS）2015 和 2018 两期数据，以 60 岁为断点估计城乡居民养老金对农村慢性病老人健康的影响，按 7 个地理区域划分站点。

**本仓库不包含任何 CHARLS 数据。** CHARLS 数据需要在 [CHARLS 官网](https://charls.pku.edu.cn/) 注册并申请，按其数据使用协议不得再分发。获得数据后：

```bash
python charls/extract.py --raw /path/to/CHARLS          # 2018 年（第 4 期）
python charls/extract_2015.py --raw /path/to/CHARLS     # 2015 年（第 3 期）
python charls/run_charls.py                             # 分析（--wave 2015 跑 2015 年）
python charls/make_charls_tables.py
```

`--raw` 指向包含 `2018 CHARLS Wave 4` 等子目录的原始 .dta 数据文件夹，所需变量见 `charls/extract.py` 和 `charls/extract_2015.py` 的说明。

一键运行全部步骤：

```bash
python run_all.py --skip-charls             # 没有 CHARLS 数据：测试 + 仿真 + 表图
python run_all.py --raw /path/to/CHARLS     # 完整运行
python run_all.py --quick --skip-charls     # 冒烟检查：每个仿真配置 40 次重复
```

## 局限

- rdrobust 默认的最近邻方差（`vce='nn'`）需要跨站点找近邻，无法联邦计算，本库只提供 HC 和聚类方差。
- 连续驱动变量的四分位距由 128 个公共分箱插值得到，初始带宽与 rdrobust 可能有极小差异。
- 使用站点固定效应时，聚合后的统计量会暴露各站点窗口内的总和，安全聚合不再能隐藏站点级信息，只在站点构成于断点处跳跃时使用。

## 作者

王楠（Nan Wang），大连理工大学。相关论文为合作研究，正在准备中，正式发表后补充引用信息。

## 许可证

[MIT](LICENSE)
