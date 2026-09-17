const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ExternalHyperlink,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageBreak, Footer, PageNumber } = require('docx');

const FONT = { ascii: 'Calibri', hAnsi: 'Calibri', eastAsia: 'Microsoft YaHei' };
const CW = 9026;

const p = (t, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts.para,
  children: [new TextRun({ text: t, font: FONT, size: 21, ...opts.run })] });
const eq = (t, note) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  children: [ new TextRun({ text: t, font: { ascii: 'Cambria Math', hAnsi: 'Cambria Math', eastAsia: 'Microsoft YaHei' }, size: 22 }),
              ...(note ? [new TextRun({ text: '    ' + note, font: FONT, size: 18, color: '777777' })] : []) ] });
const bullet = (t) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 60 },
  children: [new TextRun({ text: t, font: FONT, size: 21 })] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const link = (label, url) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 60 },
  children: [ new TextRun({ text: label + '：', font: FONT, size: 21 }),
    new ExternalHyperlink({ link: url, children: [new TextRun({ text: url, font: FONT, size: 20, color: '0563C1', underline: {} })] }) ] });

const border = { style: BorderStyle.SINGLE, size: 1, color: 'BBBBBB' };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (t, w, head = false) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
  shading: head ? { fill: 'DCE9F5', type: ShadingType.CLEAR } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 19, bold: head })] })] });
const table = (widths, rows) => new Table({ width: { size: widths.reduce((a,b)=>a+b,0), type: WidthType.DXA },
  columnWidths: widths, rows: rows.map((r,i)=>new TableRow({ children: r.map((c,j)=>cell(c,widths[j],i===0)) })) });

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: 'M-DRDC 项目：理论基础与数据资源', font: FONT, size: 36, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
    children: [new TextRun({ text: 'Theoretical Foundations & Data Resources · 2026-08 · 配套文档：项目讲解文档 / FINAL_PROPOSAL.md', font: FONT, size: 20, color: '4A5568' })] }),

  p('本文档覆盖方法五阶段流水线（孪生→扩散→力映射→预测器→共形）所依赖的全部理论模块，每节给出核心公式、直觉解释、以及在本项目中的具体用法；末尾附经过联网核实的数据集获取渠道与关键参考文献。', { run: { color: '444444' } }),

  h1('第一部分：理论基础 / Part I: Theoretical Foundations'),

  h2('1. 刀具磨损机理与寿命模型（S1 孪生的物理内核）'),
  p('刀具后刀面磨损 VB 随切削时间呈典型三阶段：初期磨合（快）、稳定磨损（近似线性）、急剧磨损（加速直至失效）。孪生需要一个能外推整条曲线的参数化模型。'),
  p('Taylor 寿命方程（1907）及其扩展形式给出寿命与切削参数的幂律关系：'),
  eq('Vc · Tⁿ = C', '基本形式：切速 Vc 与寿命 T 的幂律'),
  eq('T = C / (Vc^α · fz^β · ap^γ)', '扩展形式：加入每齿进给 fz 与切深 ap'),
  p('Usui 磨损率模型从粘着磨损机理出发，把磨损率与接触正应力 σn、滑动速度 vs、切削温度 θ 联系起来：'),
  eq('dW/dt = A · σn · vs · exp(−B/θ)', 'A、B 为材料相关常数'),
  p('本项目的孪生采用工程化的磨损率 ODE（Taylor/Usui 的可标定折中）：磨损增速由切削参数的幂律项与磨损阶段修正函数 Φ(VB) 共同决定，参数向量 θ_twin = (k₁, a, b, …) 需要从数据标定：'),
  eq('dVB/dt = k₁ · Vc^a · fz^b · Φ(VB)', 'Φ 刻画三阶段形状（如 sigmoid 加速项）'),
  p('直觉：这类模型"方向永远正确"（磨损单调增、参数越激进磨得越快），但绝对数值有系统偏差——这正是后续残差扩散要修正的对象。'),

  h2('2. 铣削力模型（S3 力特征映射的物理内核）'),
  p('线性刃力模型（Altintas）把每齿瞬时切削力分解为剪切项与刃口项，未变形切屑厚度 h 随刀齿转角 φ 变化：'),
  eq('dFt = Ktc · h(φ) · dz + Kte · dz ,    h(φ) = fz · sinφ', '切向力微元；径向/轴向同构'),
  p('关键物理事实：切削力系数随磨损增大（后刀面摩擦面积增大），常用一阶近似：'),
  eq('K(VB) = K₀ · (1 + κ · VB)', 'κ 为磨损敏感系数'),
  p('这给出"磨损 → 力特征"的单调映射，是两件事的依据：① 力信号可作为磨损的可观测代理（S4 预测器只用力特征）；② 孪生可以由磨损曲线合成力特征轨迹 F_twin(w, p)。本项目在此机理映射上再加一个约 20 系数的闭式岭回归残差项，吸收机理模型的系统偏差。'),

  h2('3. MAP 参数标定（S1 的统计内核）'),
  p('小样本下用最大后验（MAP）而非最小二乘标定孪生参数，先验来自切削手册/文献的参数范围，防止早期数据过拟合：'),
  eq('θ* = argmax_θ [ log p(D_early | θ) + log p(θ) ]', 'D_early = 目标工况寿命前 15% 的磨损测量'),
  p('多工况时可用层次贝叶斯池化（partial pooling）：各工况参数共享一个总体先验，单工况数据不足时自动向总体收缩。标定质量诊断：前 15% 拟合 R²（实验计划门槛 ≥ 0.8）。'),

  h2('4. 去噪扩散概率模型 DDPM（S2 的生成内核）'),
  p('DDPM（Ho et al., 2020）定义一个逐步加噪的前向过程和一个学习去噪的反向过程。前向过程闭式表达：'),
  eq('q(xₜ | x₀) = N( √ᾱₜ · x₀ , (1−ᾱₜ) I )', 'ᾱₜ = ∏ (1−βₛ)，βₛ 为噪声日程'),
  eq('xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε ,   ε ~ N(0, I)', '任意时刻一步采样'),
  p('训练目标是让网络 ε_θ 从带噪样本中预测噪声（等价于学习分数函数），这是全项目唯一的扩散损失（无任何附加惩罚项）：'),
  eq('L = E_{x₀, t, ε} ‖ ε − ε_θ( xₜ , t , c ) ‖²', 'c = 条件向量 [孪生增量序列, 工况 p, 失配 m]'),
  p('条件生成用无分类器引导（Classifier-Free Guidance）：训练时随机丢弃条件，采样时用带条件与无条件预测的外插控制条件强度：'),
  eq('ε̃ = (1 + w) · ε_θ(xₜ, c) − w · ε_θ(xₜ, ∅)', 'w 为引导强度'),
  p('直觉：扩散模型学的是"给定物理初稿与工况，真实磨损增量序列的分布长什么样"。采样 N=50 条即可得到生成式集合，天然给出轨迹族的不确定性扩展。'),

  h2('5. 残差化与单调结构参数化（本文的方法核心）'),
  p('残差化：不让扩散模型直接生成磨损路径 w(t)，而是学孪生与真实的差。残差 r(t) = w_real(t) − w_twin(t) 是低维、平滑、均值近零的对象，学习难度远低于原始路径——这是"把难问题变成好学问题"的关键一步。'),
  p('单调结构：扩散模型的输出是无约束增量表征 u ∈ R^T，磨损路径由确定性变换构造：'),
  eq('w_corr(t) = w_early + Σ_{τ≤t} softplus(u_τ) ,   softplus(x) = log(1 + eˣ)', 'softplus ≥ 0 ⇒ 增量非负 ⇒ 路径单调'),
  p('为什么结构优于惩罚：惩罚项只是把违反约束"变贵"，可行域仍包含非物理样本，且引入权重超参与训练冲突；结构化参数化直接把可行域缩小到物理可行集合，违反单调在数学上不可能。审稿术语：constraint by construction vs. constraint by penalty。'),
  p('端点处理：报废阈值穿越时刻的真实分布由源域真实路径经 DDPM 似然自然学得；孪生预测寿命 L_twin(p) 只在采样时作为可选引导先验（主实验权重取 0），避免"修正器反而保留孪生偏差"的漂移。'),

  h2('6. 时间卷积网络 TCN 与损失（S4 预测器）'),
  p('TCN 用因果空洞卷积堆叠获得指数级感受野，参数少、训练稳，是时序回归的标准轻量骨干（非本文创新点）。磨损回归用 Huber 损失（对显微测量的离群点稳健）；RUL 用 PHM 竞赛的不对称评分——晚预测（危险）罚得比早预测（保守）重：'),
  eq('score = Σ [ exp(−d/a₁) − 1 ]  (d<0) ;  Σ [ exp(d/a₂) − 1 ]  (d≥0)', 'd = 预测RUL − 真实RUL；a₁>a₂ ⇒ 晚预测更贵'),

  h2('7. 共形预测与偏移校正（S5 包装层）'),
  p('分割共形（split conformal）：留出校准集，计算非一致性分数并取分位数，构造的区间带有限样本、分布无关覆盖保证：'),
  eq('sᵢ = | yᵢ − ŷ(xᵢ) | ,   q̂ = Quantile( {sᵢ} ; ⌈(n+1)(1−α)⌉ / n )', ''),
  eq('C(x) = [ ŷ(x) − q̂ , ŷ(x) + q̂ ] ,   P( y ∈ C(x) ) ≥ 1 − α', '对任意预测器成立，无分布假设'),
  p('该保证的前提是校准与测试数据可交换（exchangeable）。跨工况部署破坏这一前提（协变量偏移），需用加权共形（Tibshirani et al., 2019）：以密度比 w(x) = dP_target/dP_source 对校准分数加权后取加权分位数。本项目的密度比用显式工况描述符 + 孪生失配标量的逻辑回归估计（刻意不用深度嵌入），并做权重截断与有效样本量（ESS）检查。'),
  p('评价指标：PICP（区间覆盖率，应 ≈ 名义值 1−α）、NMPIW（归一化平均区间宽度，越窄越好）、CRPS（连续分级概率评分，衡量整个预测分布的质量——本项目用于"阈值穿越时刻"分布）：'),
  eq('CRPS(F, y) = ∫ [ F(z) − 1{z ≥ y} ]² dz', 'F = 预测累积分布；对点预测退化为 MAE'),

  new Paragraph({ children: [new PageBreak()] }),
  h1('第二部分：数据资源 / Part II: Data Resources'),

  h2('8. 数据集总览与用途分工'),
  table([2200, 3300, 3526], [
    ['数据集', '内容', '在本项目中的角色'],
    ['PHM 2010（主数据）', '高速 CNC 铣削，6 把刀（c1/c4/c6 带磨损标注），每把约 315 次切削；三向力 + 三向振动 + 声发射，50 kHz；每次切削后显微测量三刃 VB', '主实验：三工况留一轮换；所有主表（K1–K3）在此产生'],
    ['Nature SciData 2024 涂层立铣刀全寿命数据（辅数据）', '涂层立铣刀全寿命多特征数据；力/振动等多传感器；含数据清洗脚本 data_processing.py 与 tool_wear.csv 四刃磨损记录', '并入源域训练池缓解"仅 2 条源轨迹"的数据饥饿；附录角色互换迁移实验'],
    ['Mendeley 变磨损铣削多元时序（备选）', '不同磨损状态与不同机床的铣削多元时序', '备用：若 SciData 工况覆盖不足时的第二源域'],
    ['NASA PCoE Milling（备选）', '经典铣削磨损老数据集（Matlab 格式）', '备用/鲁棒性检查，不进主表'],
  ]),

  h2('9. 获取渠道（2026-08-01 联网核实）'),
  p('PHM 2010 Data Challenge（三个渠道，任选其一）：'),
  link('IEEE DataPort 官方存档（需 IEEE 账号，机构订阅可免费）', 'https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge'),
  link('Kaggle 公开镜像（CC0 许可，免费直下，搜索 "PHM data challenge 2010"）', 'https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010'),
  link('PHM Society 官网数据挑战页（含任务原始说明）', 'https://phmsociety.org/'),
  p('Nature SciData 2024 涂层立铣刀数据集：'),
  link('论文页（Data Records 节含 figshare 数据 DOI 直链）', 'https://www.nature.com/articles/s41597-024-04345-2'),
  link('PMC 免费全文（论文被墙时的替代入口）', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11704274/'),
  p('备选数据集：'),
  link('Mendeley Data：Multivariate time series of milling with varying tool wear', 'https://data.mendeley.com/datasets/zpxs87bjt8/3'),
  link('NASA PCoE 数据仓库（页内搜索 Milling Data Set）', 'https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/'),
  p('注意事项：① Kaggle 镜像下载后务必与文献报告的切削次数（c1=315 等）对账；② SciData 数据请从论文 Data Records 节的 figshare DOI 进入，避免第三方转载版本缺文件；③ PHM2010 的磨损标签是三刃分别测量，需决定用最大值还是平均值作为回归目标（文献两种都有，论文中需声明）。', { run: { size: 19, color: '8B4513' } }),

  h2('10. 数据准备清单（对应实验计划 W1 / R001–R003）'),
  bullet('下载 PHM2010，解析 c1/c4/c6 的 wear 文件与逐切削信号文件，核对切削次数与文献一致'),
  bullet('实现每次切削的力特征提取（RMS/峰值/均值等逐刀特征），生成 (特征序列, 磨损序列) 对'),
  bullet('下载 SciData 数据集，运行其自带 data_processing.py，对齐到与 PHM2010 相同的特征模式'),
  bullet('把全部磨损路径样条重采样到 T′=64 增量网格（M-DRDC 的统一输入格式）'),
  bullet('用 3 篇已发表论文的 PHM2010 基线数字校验自己的指标实现（防指标 bug 污染全部后续实验）'),

  h2('11. 关键参考文献 / Key References'),
  table([600, 5200, 3226], [
    ['#', '文献', '支撑的模块'],
    ['1', 'Ho, Jain & Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020', 'DDPM 理论（§4）'],
    ['2', 'Ho & Salimans. Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop', '条件引导 CFG（§4）'],
    ['3', 'Tibshirani, Barber, Candès & Ramdas. Conformal Prediction Under Covariate Shift. NeurIPS 2019', '加权共形（§7）'],
    ['4', 'Angelopoulos & Bates. A Gentle Introduction to Conformal Prediction. FnT ML 2023', '共形预测入门（§7）'],
    ['5', 'Altintas. Manufacturing Automation (2nd ed.). Cambridge Univ. Press 2012', '铣削力模型（§2）'],
    ['6', 'Usui, Shirakashi & Kitagawa. Analytical prediction of cutting tool wear. Wear 1984', 'Usui 磨损模型（§1）'],
    ['7', 'Bending Moment Synthesis by Integrating Cutting Mechanics with Diffusion Model for Tool Wear Prediction. Research Square 预印本 2026-07', '最近竞品 = B5 基线原型，必须正面引用'],
    ['8', 'PHM Society. 2010 PHM Data Challenge（官方任务说明）', '主数据集协议（§8–9）'],
    ['9', 'A multi-feature dataset of coated end milling cutter tool wear whole life cycle. Scientific Data 2024/2025', '辅数据集（§8–9）'],
    ['10', 'Bai, Kolter & Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks. arXiv 2018', 'TCN 骨干（§6）'],
  ]),
  p('注：写论文时相关工作还需覆盖查新阶段发现的 2025–2026 竞品（Measurement DT+DDPM、MSSP 分段共形、AEI 物理引导 UDA 等），完整清单见 REFINEMENT_REPORT.md 与查新记录。', { run: { size: 19, color: '777777' } }),
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

Packer.toBuffer(doc).then(b => { fs.writeFileSync('/Users/jixia/Downloads/dt-toolwear-paper/M-DRDC理论基础与数据资源.docx', b); console.log('written', b.length, 'bytes'); });
