const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageBreak, Footer, PageNumber } = require('docx');

const FONT = { ascii: 'Calibri', hAnsi: 'Calibri', eastAsia: 'Microsoft YaHei' };
const EN_COLOR = '4A5568';
const CW = 9026; // A4 content width with 1" margins

const cn = (t, opts = {}) => new Paragraph({
  spacing: { after: 80 },
  ...opts.para,
  children: [new TextRun({ text: t, font: FONT, size: 21, ...opts.run })],
});
const en = (t, opts = {}) => new Paragraph({
  spacing: { after: 160 },
  ...opts.para,
  children: [new TextRun({ text: t, font: FONT, size: 20, color: EN_COLOR, italics: false, ...opts.run })],
});
const bullet = (t, color) => new Paragraph({
  numbering: { reference: 'bullets', level: 0 },
  spacing: { after: 60 },
  children: [new TextRun({ text: t, font: FONT, size: color ? 20 : 21, color: color || '000000' })],
});
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });

const border = { style: BorderStyle.SINGLE, size: 1, color: 'BBBBBB' };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (t, w, head = false) => new TableCell({
  borders, width: { size: w, type: WidthType.DXA },
  shading: head ? { fill: 'DCE9F5', type: ShadingType.CLEAR } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 19, bold: head })] })],
});
const table = (widths, rows) => new Table({
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: widths,
  rows: rows.map((r, i) => new TableRow({ children: r.map((c, j) => cell(c, widths[j], i === 0)) })),
});

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: '数字孪生驱动的刀具磨损预测论文项目 · 讲解文档', font: FONT, size: 36, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: 'M-DRDC: Mechanics-Anchored Monotone Residual Diffusion for Tool-Wear Path Correction — A Bilingual Explainer', font: FONT, size: 24, color: EN_COLOR })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: '2026-08 · 方案评审 9.1/10 READY（GPT-5.4 三轮对抗评审） · 目标期刊 JIM / Measurement / EAAI（中科院二区）', font: FONT, size: 18, color: '777777' })] }),

  h1('0. 一句话总结 / One-Sentence Summary'),
  cn('用教科书级的切削机理模型（数字孪生）先给出刀具磨损曲线的"物理初稿"，再训练一个很小的扩散模型专门学习"初稿与真实之间的误差规律"来修正它——修正过程从结构上保证磨损只增不减；最后用共形预测给出换刀决策的可信区间。'),
  en('We let a textbook cutting-mechanics model (the digital twin) draft a physically sound "first version" of the tool-wear curve, then train a small diffusion model that learns only the systematic error between this draft and reality — with wear monotonicity guaranteed by construction — and finally wrap tool-change decisions with conformal prediction intervals.'),

  h1('1. 背景与动机 / Background & Motivation'),
  h2('1.1 要解决的实际问题 / The Practical Problem'),
  cn('铣削加工中刀具持续磨损，磨损超限（如后刀面磨损 VB 达到阈值）会导致工件报废甚至断刀。理想做法是用传感器信号（切削力等）实时预测磨损值和剩余寿命（RUL）。但深度学习预测器需要大量"全寿命+磨损标注"数据，而每条这样的数据都要把一把刀磨报废、并反复用显微镜测量——在新的切削工况下，实测标注数据几乎总是稀缺的（本项目设定：目标工况只有不超过 20% 的标注可用）。'),
  en('In milling, tools wear continuously; once flank wear VB exceeds a threshold, parts are scrapped or the tool breaks. Ideally, sensor signals (cutting forces, etc.) drive real-time prediction of wear and remaining useful life (RUL). But deep predictors need abundant run-to-failure data with wear labels, and every such trajectory costs a sacrificed tool plus repeated microscope measurements — so under any NEW cutting condition, labeled data is almost always scarce (this project assumes at most 20% of target-condition labels).'),
  h2('1.2 为什么现有方法不够 / Why Existing Approaches Fall Short'),
  cn('主流补救思路是"生成式数据增强"：用物理仿真或生成模型（GAN、扩散模型）造出虚拟训练样本。2026 年 7 月的一篇预印本已经把"切削力学+扩散模型直接生成信号"做了。但这类直接生成有一个共同缺陷：生成的信号缺乏退化一致性——模型可能先生成一段"磨损严重"的信号、紧接着又生成"崭新刀具"的信号；磨损轨迹可能永远不越过报废阈值。用这种数据训练预测器，反而会学到错误规律（负迁移）。此外，现有不确定性量化多是"事后套壳"，一旦工况变化（分布偏移），置信区间的覆盖率就失效。'),
  en('The mainstream remedy is generative augmentation: create virtual training samples via physics simulation or generative models (GANs, diffusion). A July-2026 preprint already combines cutting mechanics with a diffusion model to directly generate signals. But direct generation shares one flaw: no degradation consistency — the generator may emit a "heavily worn" window followed by a "fresh tool" window, and trajectories may never cross the failure threshold. Predictors trained on such data learn wrong patterns (negative transfer). Meanwhile, existing uncertainty quantification is an afterthought whose interval coverage collapses once the operating condition shifts.'),
  h2('1.3 选题如何炼成 / How the Topic Was Forged'),
  cn('本选题经过完整的"联网调研 → 深度查新 → 撞车规避 → 三轮对抗评审"流程：初版方案（物理引导扩散+共形预测的直接组合）在查新中被判 3.5/10——因为与上述预印本正面撞车；随后重构为"残差修正+结构化单调"路线，经 GPT-5.4 三轮评审从 6.3 分迭代到 9.1 分（READY）。关键教训：卖点不是"用了扩散模型"，而是"扩散模型学什么对象、带什么结构"。'),
  en('This topic survived a full pipeline of web survey, deep novelty check, collision avoidance, and three rounds of adversarial review: the initial framing (physics-guided diffusion + conformal, naively combined) scored 3.5/10 for colliding head-on with the preprint; the reframed residual-correction + structural-monotonicity route was then iterated 6.3 → 8.1 → 9.1 (READY) under GPT-5.4 review. Key lesson: the selling point is not "we used diffusion", but WHAT object the diffusion models and WHAT structure it carries.'),

  h1('2. 核心思想的三步直觉 / The Core Idea in Three Intuitions'),
  cn('比喻：让物理学家先写初稿，让 AI 只做校对，且校对规则里内置常识。'),
  en('Analogy: let the physicist write the first draft, let AI act only as the copy-editor, and hard-wire common sense into the editing rules.'),
  h2('第一步：机理孪生 = 物理初稿 / Step 1: The Mechanistic Twin as a Physics Draft'),
  cn('经典切削理论（Taylor 寿命方程、Usui 磨损模型、铣削力模型）能在只观测刀具寿命前 15% 数据的情况下标定参数，然后外推整条磨损曲线。这条曲线大方向正确（物理保证），但细节有系统性偏差（仿真与现实的差距，sim-to-real gap）。'),
  en('Classical cutting theory (Taylor tool-life equation, Usui wear model, milling force models) can be calibrated from only the first 15% of a tool’s life and then extrapolate the whole wear curve. This curve is directionally right (physics guarantees it) but systematically biased in the details — the sim-to-real gap.'),
  h2('第二步：扩散模型只学"误差规律" / Step 2: Diffusion Learns Only the Error Pattern'),
  cn('与其让生成模型从零造信号（要学的东西太多、太容易造假），不如让它只学"孪生初稿与真实曲线之间的残差"。残差是低维、平滑、规律性强的对象——一个约 80 万参数的一维扩散模型就够了（对比：直接生成信号通常要千万级参数）。这就是"残差化"（residualization）的价值：把难学的问题变成好学的问题。'),
  en('Rather than asking a generative model to build signals from scratch (too much to learn, too easy to hallucinate), we ask it to learn only the residual between the twin’s draft and the real curve. The residual is low-dimensional, smooth, and highly structured — a ~0.8M-parameter 1-D diffusion model suffices (direct signal generation typically needs tens of millions). That is the value of residualization: it converts a hard learning problem into an easy one.'),
  h2('第三步：单调性写进结构而非损失函数 / Step 3: Monotonicity by Construction, Not by Penalty'),
  cn('刀具磨损永远不会自己变浅——这是物理常识。我们不用惩罚项"劝"模型遵守（惩罚项可以被违反），而是改变参数化方式：扩散模型输出的是增量序列 u，磨损路径 = 初始磨损 + cumsum(softplus(u))。softplus 保证每一步增量非负，累加保证单调。违反单调的样本在数学上不可能被生成。这是评审从 6.3 到 9.1 的最关键一步。'),
  en('Tool wear never spontaneously decreases — that is physical common sense. Instead of "persuading" the model via a penalty loss (penalties can be violated), we change the parameterization: the diffusion model outputs an increment sequence u, and the wear path = initial wear + cumsum(softplus(u)). Softplus makes every increment non-negative; the cumulative sum makes the path monotone. Non-monotone samples are mathematically impossible. This single move drove the review score from 6.3 to 9.1.'),

  new Paragraph({ children: [new PageBreak()] }),
  h1('3. 方法流水线（五个阶段） / The Five-Stage Pipeline'),
  table([900, 2500, 3300, 2326], [
    ['阶段', '模块 / Module', '做什么 / What it does', '关键设计 / Key design'],
    ['S1', '机理孪生 Twin', '磨损 ODE + 铣削力模型，MAP 标定（源工况全寿命 + 目标工况寿命前 15%）', '输出孪生磨损曲线、力映射、失配标量 m'],
    ['S2', 'M-DRDC 核心', '约 0.8M 参数一维 DDPM，学习残差增量序列（T′=64 样条重采样网格）', '单调性结构保证；条件=[孪生增量, 工况 p, 失配 m]；端点引导权重主表取 0'],
    ['S3', '力特征映射', '机理力模型 F(w,p) + 闭式岭回归残差（约 20 个系数）', '刻意用线性修正，防止审稿人把增益归功于它'],
    ['S4', '预测器 TCN', '纯力信号时间卷积网络，磨损回归 + RUL', '合成/实测特征模式完全一致，防止域线索泄漏'],
    ['S5', '共形包装层', '加权 split 共形区间 + 换刀决策', '权重来自显式工况描述符（非深度嵌入）；定位为部署包装层，不是新颖性声明'],
  ]),
  cn(''),
  cn('损失函数极简：扩散部分只有标准 ε-预测损失（演化过程中三个代理惩罚项全部删除）；TCN 用 Huber + RUL 评分损失；岭回归闭式解，无需训练。', { para: { spacing: { before: 120, after: 80 } } }),
  en('The loss design is deliberately minimal: the diffusion uses only the standard ε-prediction loss (all three proxy penalty terms were deleted during refinement); the TCN uses Huber + RUL-score loss; the ridge correction is closed-form.'),

  h1('4. 与竞品的差异 / How This Differs from Prior Work'),
  table([2800, 3100, 3126], [
    ['对象 / Prior work', '它做了什么 / What it does', '我们的差异 / Our delta'],
    ['2026-07 预印本（切削力学+FiLM 扩散）', '物理量作为条件，直接生成信号窗口', '生成对象不同（磨损路径 vs 信号窗）；监督目标不同（残差 vs 直接）；一致性不同（结构保证 vs 无）'],
    ['轴承 DT+扩散系列（Measurement 2025/26 等）', '仿真驱动扩散做故障分类', '我们是退化轨迹级回归+决策不确定性，非分类'],
    ['MSSP 2026 分段共形刀具监测', '共形区间，无生成、无偏移校正', '我们有生成式孪生+偏移感知加权'],
    ['CycleGAN 虚实映射系列', '无约束信号映射', '单调结构+残差化+更小模型'],
  ]),
  cn(''),
  cn('论文写作要点：必须正面引用预印本，并用 B5/B5+ 公平基线正面对比（同骨干、同参数量、同条件预算），这是审稿人最关注的对照。', { para: { spacing: { before: 120, after: 80 } } }),
  en('Writing note: the preprint MUST be cited and confronted head-on via the fair baselines B5/B5+ (same backbone, same parameter and conditioning budget) — this is the comparison reviewers will scrutinize first.'),

  h1('5. 实验设计 / Experimental Design'),
  h2('5.1 声明→证据 / Claims to Evidence'),
  table([3300, 5726], [
    ['声明 / Claim', '证据 / Minimum convincing evidence'],
    ['C1 主声明：残差化+单调结构在小样本下优于直接生成', 'ours > B5、B5+、A1（磨损 MAE / RUL 分数，3 轮换 × 5 种子，误差棒不重叠，k=5/10%）'],
    ['C2 包装层：加权共形在工况偏移下保覆盖率', 'PICP@90 距名义值 ≤2 点；普通 CP / MC-dropout 下降 ≥5 点'],
    ['反声明 1：增益其实来自力修正项 h_psi', '隔离实验：ours（h_psi=0）仍胜；B3+h_psi 追不上'],
    ['反声明 2：增益来自孪生端点引导的偏置', '主表引导权重=0；敏感性曲线 {0, 0.1, 0.5, 1.0}'],
    ['反声明 3：64 长度序列用扩散是杀鸡用牛刀', 'GP 增量采样器对照（nice-to-have，附录）'],
  ]),
  h2('5.2 基线矩阵 / Baseline Matrix'),
  table([1200, 4300, 3526], [
    ['编号', '系统 / System', '回答什么问题 / Question answered'],
    ['B1', '全标注 TCN', '性能上界'],
    ['B2', '仅 k% 标注 TCN', '稀缺下界'],
    ['B3', '纯孪生增强（+岭修正）', '到底需不需要生成模型？'],
    ['B5', '直接生成磨损路径的 DDPM（同预算，条件 [p,m]）', '残差化本身值多少？（对预印本的杀招）'],
    ['B5+', 'B5 + 孪生曲线作为条件通道', '信息预算对齐后的公平对照'],
    ['A1', '非单调原始残差扩散', '单调结构值多少？'],
    ['OURS', 'M-DRDC 完整方法', '头条结果'],
  ]),
  cn(''),
  cn('数据：PHM2010（c1/c4/c6 三工况轮换，留一工况为目标）+ Nature SciData 2024 涂层立铣刀数据集（并入源域训练池）。指标：磨损 MAE/RMSE、RUL PHM 评分、穿越时间 CRPS 与覆盖率（证明生成对象本身有用）、PICP/区间宽度/换刀延迟率。', { para: { spacing: { before: 120, after: 80 } } }),
  en('Data: PHM2010 (c1/c4/c6, leave-one-condition-out rotations) plus the Nature SciData 2024 coated end-mill dataset merged into the source training pool. Metrics: wear MAE/RMSE, RUL PHM score, threshold-crossing-time CRPS and coverage (showing the generative object is useful in itself), PICP / interval width / late tool-change rate.'),

  new Paragraph({ children: [new PageBreak()] }),
  h1('6. 12 周执行路线 / 12-Week Execution Roadmap'),
  table([1000, 1400, 4200, 2426], [
    ['周', '里程碑', '任务 / Tasks', '通过门槛 / Gate'],
    ['W1', 'M0 数据与孪生', '数据下载、标签解析、指标对账；孪生 MAP 标定', '孪生前 15% 拟合 R² ≥ 0.8；指标复现文献数字'],
    ['W2', 'M1 基线', 'B1 / B2 / B3 全网格（TCN 分钟级）', 'B1 达文献水平；B2 随 k 单调变差'],
    ['W3–4', 'M2 核心模型', 'M-DRDC 训练+采样；本阶段冻结全部超参', '修正路径在留出源域上 CRPS 优于原始孪生'],
    ['W5', 'M2.5 首战', 'OURS 下游全网格', 'ours 在 k=10% 显著优于 B3；失败则先修生成器'],
    ['W6', 'M3 决胜周', 'B5 / B5+ / A1 杀招对比', '头条声明判定；若 B5+≈ours 启动应急叙事'],
    ['W7', 'M3.5 UQ', '加权共形 vs 普通共形 vs MC-dropout', '覆盖率达标'],
    ['W8', 'M4 防御', 'h_psi 隔离、引导敏感性（+GP 对照）', '三个反声明关闭'],
    ['W9', 'M4.5 收尾', '标定比例扫描、SciData 迁移、失效分析', '附录材料齐'],
    ['W10–12', 'M5 写作投稿', '论文写作 + 内审循环 + 投 JIM', '内审 ≥ 8/10 再投'],
  ]),
  cn(''),
  cn('算力预算：约 30 GPU 时（单张消费级显卡）。扩散训练占约 20 时，TCN 与共形均为分钟级。', { para: { spacing: { before: 120, after: 80 } } }),
  en('Compute budget: ~30 GPU-hours on a single consumer GPU. Diffusion training dominates (~20 h); TCN and conformal steps run in minutes.'),

  h1('7. 风险与预案 / Risks & Contingencies'),
  bullet('最大风险：W6 中 B5+ 追平 ours（残差化增益不显著）。预案：叙事转向"单调结构增益（vs A1）+ 早期可辨识性"，目标期刊降为 Measurement/EAAI，论文不死。'),
  bullet('Biggest risk: B5+ matches ours in Week 6 (residualization gain insignificant). Contingency: re-center the narrative on the structure delta (vs A1) plus early-cycle identifiability; retarget Measurement/EAAI. The paper survives.', EN_COLOR),
  bullet('小数据方差：源域全寿命轨迹仅 2 条 PHM2010 + 约 6 条 SciData。缓解：裁剪段训练、0.8M 小模型、早停、5 种子、按轮换配对报告。'),
  bullet('Small-data variance: only 2 PHM2010 + ~6 SciData full-life source paths. Mitigations: random-crop training, 0.8M capacity, early stopping, 5 seeds, paired per-rotation reporting.', EN_COLOR),
  bullet('预印本中途见刊：引用期刊版即可，B5/B5+ 对比就是为它设计的正面回应。'),
  bullet('If the preprint gets published mid-project: cite the journal version; the B5/B5+ comparison is the designed answer.', EN_COLOR),

  h1('8. 术语表 / Glossary'),
  table([2900, 2200, 4026], [
    ['English', '中文', '一句话解释 / One-line explanation'],
    ['Digital Twin (DT)', '数字孪生', '物理设备的可标定仿真副本，此处指机理磨损+力模型'],
    ['Flank wear (VB)', '后刀面磨损', '刀具报废判据，显微镜测量，单位微米'],
    ['RUL', '剩余使用寿命', '距离磨损超限还能切多少刀/多久'],
    ['DDPM (diffusion model)', '去噪扩散概率模型', '先加噪再学去噪的生成模型，本文用于生成残差增量序列'],
    ['Residualization', '残差化', '只学"物理初稿与真实的差"，不从零生成'],
    ['Monotone by construction', '结构化单调', '用 softplus+累加的参数化让磨损不可能下降'],
    ['Sim-to-real gap', '仿真-现实差距', '仿真数据与实测数据的系统性分布差异'],
    ['Negative transfer', '负迁移', '用失真的合成数据训练反而降低真实性能'],
    ['FiLM conditioning', 'FiLM 条件注入', '预印本 A 的条件机制：用缩放+平移调制网络特征'],
    ['Conformal prediction (CP)', '共形预测', '分布无关的区间构造方法，有限样本覆盖率保证'],
    ['Weighted split-CP', '加权分割共形', '用似然比权重校正分布偏移下的共形区间'],
    ['PICP / NMPIW', '区间覆盖率 / 归一化区间宽度', 'UQ 质量的两个标准指标：既要盖得住，又要不太宽'],
    ['Crossing-time CRPS', '穿越时间连续分级概率评分', '衡量"预测何时到达报废阈值"这个分布的准确度'],
    ['MAP calibration', '最大后验标定', '带文献先验的参数估计，防止小数据过拟合'],
    ['TCN', '时间卷积网络', '轻量时序回归骨干，非本文创新点'],
    ['PHM2010', 'PHM2010 铣刀数据集', '公开基准：3 把刀全寿命，315 次切削，显微磨损标注'],
    ['Leave-one-condition-out', '留一工况', '两工况做源域、一工况做目标域的轮换协议'],
  ]),

  h1('9. 文件索引 / File Index'),
  bullet('FINAL_PROPOSAL.md — 最终方案（评审 9.1/10 READY）/ Final proposal (READY)'),
  bullet('REFINEMENT_REPORT.md — 三轮评审演化报告 / Three-round refinement report'),
  bullet('EXPERIMENT_PLAN.md — 声明驱动实验计划 / Claim-driven experiment plan'),
  bullet('EXPERIMENT_TRACKER.md — 18 个 Run 的执行跟踪表 / Run tracker (18 runs)'),
  bullet('目录 / Directory: /Users/jixia/Downloads/dt-toolwear-paper/refine-logs/'),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: '1F4E79' },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: '2E5F8F' },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 620, hanging: 320 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: '888888' })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync('/Users/jixia/Downloads/dt-toolwear-paper/M-DRDC项目讲解文档_Bilingual_Explainer.docx', b); console.log('written', b.length, 'bytes'); });
