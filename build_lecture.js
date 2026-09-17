const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  ImageRun,
  AlignmentType,
  LevelFormat,
  HeadingLevel,
  BorderStyle,
  WidthType,
  ShadingType,
  PageBreak,
  Footer,
  PageNumber,
} = require("docx");

const FONT = {
  ascii: "Calibri",
  hAnsi: "Calibri",
  eastAsia: "Microsoft YaHei",
};
const MONO = {
  ascii: "Consolas",
  hAnsi: "Consolas",
  eastAsia: "Microsoft YaHei",
};
const FIGD = "/Users/jixia/Downloads/dt-toolwear-paper/code/paper/lecture_figs";

const p = (t, o = {}) =>
  new Paragraph({
    spacing: { after: 120 },
    ...o.para,
    children: [new TextRun({ text: t, font: FONT, size: 21, ...o.run })],
  });
const tip = (t) =>
  new Paragraph({
    spacing: { after: 140 },
    indent: { left: 280 },
    shading: { fill: "FFF7E6", type: ShadingType.CLEAR },
    children: [
      new TextRun({
        text: "讲课提示：" + t,
        font: FONT,
        size: 19,
        color: "8C5A00",
      }),
    ],
  });
const qa = (t) =>
  new Paragraph({
    spacing: { after: 140 },
    indent: { left: 280 },
    shading: { fill: "E8F1FB", type: ShadingType.CLEAR },
    children: [
      new TextRun({
        text: "预判提问：" + t,
        font: FONT,
        size: 19,
        color: "1F4E79",
      }),
    ],
  });
const code = (lines) =>
  lines
    .map(
      (l) =>
        new Paragraph({
          spacing: { after: 0 },
          shading: { fill: "F4F4F4", type: ShadingType.CLEAR },
          children: [new TextRun({ text: l || " ", font: MONO, size: 17 })],
        }),
    )
    .concat([p("")]);
const h1 = (t) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text: t, font: FONT })],
  });
const h2 = (t) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text: t, font: FONT })],
  });
const bullet = (t) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text: t, font: FONT, size: 21 })],
  });
const img = (name, w, h, caption) => [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(`${FIGD}/${name}.png`),
        transformation: { width: w, height: h },
        altText: { title: name, description: caption, name },
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [
      new TextRun({ text: caption, font: FONT, size: 18, color: "666666" }),
    ],
  }),
];

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (t, w, head = false) =>
  new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "DCE9F5", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [
      new Paragraph({
        children: [new TextRun({ text: t, font: FONT, size: 19, bold: head })],
      }),
    ],
  });
const table = (widths, rows) =>
  new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map(
      (r, i) =>
        new TableRow({
          children: r.map((c, j) => cell(c, widths[j], i === 0)),
        }),
    ),
  });

const children = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 80 },
    children: [
      new TextRun({
        text: "刀具磨损预测项目讲义（第一部分）",
        font: FONT,
        size: 34,
        bold: true,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [
      new TextRun({
        text: "数据管线 · 机理孪生 · 基线实验 —— 代码逻辑与结果",
        font: FONT,
        size: 24,
        color: "4A5568",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({
        text: "时长 60 分钟 ｜ 代码：dt-toolwear-paper/code ｜ 2026-08",
        font: FONT,
        size: 18,
        color: "777777",
      }),
    ],
  }),

  h1("课程地图与时间分配"),
  table(
    [1200, 5200, 1400, 1226],
    [
      ["节", "内容", "时间", "对应代码"],
      ["§1", "问题设定与数据认识", "10 min", "data/phm2010.py"],
      [
        "§2",
        "数据管线的代码逻辑与验收",
        "10 min",
        "data/features.py, scripts/r001",
      ],
      [
        "§3",
        "磨损路径编码器与评价指标",
        "8 min",
        "data/resample.py, metrics.py",
      ],
      ["§4", "机理孪生与 MAP 标定", "15 min", "twin/*.py, scripts/r002"],
      ["§5", "基线实验协议与结果", "12 min", "experiments/*, scripts/w2"],
      ["§6", "小结与第二部分预告", "5 min", "—"],
    ],
  ),
  p(""),

  h1("§1 问题设定与数据认识（10 min）"),
  p(
    "任务：铣削加工中刀具持续磨损，后刀面磨损 VB 超过阈值（本项目取 165 μm）即报废。我们要在目标刀具只有少量磨损标注（k≤20%）的条件下，预测其寿命末段（最后 20% 切削）的磨损值与剩余寿命 RUL。为什么标注贵：每个标注点都要停机、用显微镜测量三条刃的磨损再取最大值。",
  ),
  p(
    "数据：PHM2010 公开数据集，6 把刀中 c1/c4/c6 带标注，每把 315 次切削，每次切削记录 7 通道 50 kHz 信号（三向力、三向振动、声发射），共 17 GB。",
  ),
  ...img(
    "L1_data_overview",
    620,
    220,
    "图 1  (a) 三把刀全寿命磨损曲线；(b) Fx RMS 特征随磨损单调上升——力信号是磨损的可观测代理",
  ),
  bullet(
    '看图 1a：三把刀都有"磨合快—平台稳—末段加速"三阶段形状，但幅度差异大（c1 到 173，c6 到 235 μm）——跨刀差异就是后面一切困难的来源。',
  ),
  bullet(
    '看图 1b：力特征与磨损强相关（物理：磨损→接触面积→摩擦力↑），这是"用力信号回归磨损"的依据；注意 c1 在 255 刀附近的尖峰是传感器毛刺——真实数据永远需要清洗意识。',
  ),
  tip(
    '开场 3 分钟先讲"为什么标注贵"，学生对问题的稀缺性有体感后，后面所有设计选择都顺理成章。',
  ),

  h1("§2 数据管线的代码逻辑与验收（10 min）"),
  h2("2.1 解析器：防御式编程"),
  p(
    "phm2010.py 的设计原则：所有外部数据先验证再使用，失败时报出可操作的错误信息。核心逻辑：",
  ),
  ...code([
    "@dataclass(frozen=True)          # 不可变容器：解析结果不许被偷偷改",
    "class ToolRecord:",
    "    tool: str",
    "    wear: np.ndarray             # 回归目标 = 三刃最大值（论文中声明）",
    "    signal_files: tuple          # 按切削序号数值排序，非字典序",
    "",
    "def load_tool(root, tool):",
    "    wear = load_wear(...)        # 校验 3 个 flute 列、数值有限",
    "    files = _find_signal_files() # 正则提取序号排序",
    "    if len(files) != len(wear):  # 315 刀对账，不符即报错",
    "        raise ValueError(...)",
  ]),
  bullet(
    "两个易错点：① 信号文件名 c_1_100.csv 按字符串排序会排在 c_1_99.csv 前面，必须按数字排序；② 磨损目标取三刃 max 还是 mean 文献两种都有，代码保留两者、论文声明用 max（保守的报废判据）。",
  ),
  h2("2.2 特征提取与验收门槛"),
  p(
    '每刀提取 12 维纯力特征（Fx/Fy/Fz × RMS/峰值/绝对均值/标准差）。为什么只用力：虚实一致性——后面孪生只能合成力特征，真实与合成样本必须同模式，否则模型会学到"哪个是合成的"这种捷径。',
  ),
  p("R001 验收结果（真实数据）：三把刀 315/315 全通过，磨损范围与文献吻合。"),
  qa(
    '学生可能问"为什么不用振动和声发射？信息不是更多吗"——答：第一部分只做基线，第二部分的对照实验要求虚实特征同构，这是实验公平性设计，附录会讨论多模态。',
  ),

  h1("§3 磨损路径编码器与评价指标（8 min）"),
  h2("3.1 单调编码器：结构保证而非事后修补"),
  p(
    "磨损物理上只增不减，但测量有噪声。我们的编码链：等张化（累计最大值）→ PCHIP 单调插值重采样到 64 点 → 增量取 softplus 逆变换得到无约束表征 u。解码时 softplus 保证每步增量非负，累加保证路径单调——任何 u 解码出的路径都物理合法。",
  ),
  ...img(
    "L2_codec",
    620,
    186,
    "图 2  编码三步曲：重采样 → u 表征 → 单调解码（往返误差 <1e-4）",
  ),
  ...code([
    "w_corr(t) = w0 + Σ softplus(u_τ)     # 单调性由构造保证",
    "softplus(x) = log(1 + e^x) ≥ 0",
    "# 工程细节：增量下限 0.05 μm（ENC_EPS），否则平台段",
    "# softplus⁻¹(≈0) → -13.8 的尖峰会污染任何下游学习",
  ]),
  tip(
    'ENC_EPS 这个细节值得讲 2 分钟：它是第二部分调试十轮中真实踩过的坑，"数学正确≠数值健康"。',
  ),
  h2("3.2 指标：不对称的代价"),
  p(
    "磨损用 MAE/RMSE；RUL 用 PHM 不对称评分——预测晚了（刀已坏还说没坏）罚得比预测早了重，因为代价不对称（报废工件 vs 提前换刀）。共形区间用 PICP（覆盖率）与 NMPIW（区间宽度）。",
  ),

  h1("§4 机理孪生与 MAP 标定（15 min，本讲核心）"),
  h2("4.1 三阶段磨损率 ODE"),
  ...code([
    "dVB/dn = g(p) · [ β_brk·exp(−VB/VB_brk)  ← 磨合期（递减）",
    "                + 1                        ← 稳定期（常数）",
    "                + β_acc·(VB/VB_acc)²  ]   ← 加速期（递增）",
    "g(p) = k1·(Vc/Vc0)^a·(fz/fz0)^b           ← 工况因子",
  ]),
  p(
    "参数在对数空间拟合保证正性。PHM2010 三把刀共享同一切削工况，所以 (a,b) 在此不可辨识，冻结在先验均值——这不是缺陷而是诚实：单一工况数据里就没有这个信息。",
  ),
  h2("4.2 MAP 标定：先验的作用"),
  ...code([
    "θ* = argmin  Σ(VB_sim − VB_meas)²/σ²   ← 数据项（σ=5μm 测量噪声）",
    "           + Σ((θ − μ_prior)/τ)²        ← 先验项（文献参数范围）",
    "# L-BFGS-B 多起点（8 个）防局部极小",
  ]),
  p(
    "先验来自切削手册的参数范围。它的作用在标注少时最明显：15 个含噪点足以把 7 维参数拉到物理合理区域，而纯最小二乘会追着噪声跑。",
  ),
  ...img(
    "L3_twin_calibration",
    620,
    200,
    "图 3  只用寿命前 15%（蓝色窗）标定，外推整条寿命。R² 全部 ≥0.88（门槛 0.8）",
  ),
  table(
    [1100, 1800, 1800, 2000],
    [
      ["刀具", "前15% 标定 R²", "全寿命外推 MAE", "解读"],
      ["c1", "0.882", "11.1 μm", "外推最好"],
      ["c4", "0.940", "12.9 μm", "标定好、末段藏雷（第二部分揭晓）"],
      ["c6", "0.990", "29.8 μm", "磨损范围最大、外推最难"],
    ],
  ),
  p(""),
  bullet(
    '这张表是全项目的伏笔：孪生"方向对、细节有系统偏差"（10–30 μm），这个偏差正是后续所有方法要去缩小的对象。',
  ),
  h2("4.3 力映射：从磨损合成特征"),
  p(
    "机理部分 K(VB)=K₀(1+κ·VB)（磨损增大接触摩擦），加闭式岭回归吸收残差：每个输出特征 7 个基函数系数 [1, vb, vb², vb·vc, vb·fz, vc, fz]，12 个特征共 84 个（单工况下 vc/fz 列退化，有效约 36 个）。λ=0.01 的岭项不可省——单工况时 vc/fz 是常数列，与截距完全共线，无岭则矩阵奇异。刻意低容量（比扩散模型小四个数量级）：将来对照实验的增益不能被归功于这个部件（审稿防御，附录有 h_psi=0 消融）。",
  ),

  h1("§5 基线实验协议与结果（12 min）"),
  h2("5.1 协议设计的三个决定"),
  bullet("轮换：三把刀轮流当目标（留一工况），其余两把当源域——3 组独立评估。"),
  bullet(
    '标注预算：目标刀前 80% 范围内均匀抽 k%（k=5/10/20）。教训：最初设计成"连续前 k%"，导致稀缺下界 B2 变成纯外推任务、结果被归一化伪影支配（B2@20% 反而比 @5% 差）——协议本身也要接受实验检验。',
  ),
  bullet(
    "测试窗：所有系统统一用最后 20% 切削（高磨损、决策关键区），跨 k 可比。",
  ),
  h2("5.2 基线结果全景"),
  ...img(
    "L4_baselines",
    580,
    232,
    "图 4  W2 基线：B1 上界 21 μm；B2 随标注减少单调恶化；B3 纯孪生增强在低 k 时无增益甚至负迁移",
  ),
  table(
    [2100, 3300, 3826],
    [
      ["发现", "数字", "含义"],
      [
        "门槛 A：文献对账",
        "LIT 38.3 μm ∈ 跨刀文献区间 20–50",
        "实现无 bug（注意：门槛最初错用了同刀文献口径 8–25，已留档修正）",
      ],
      [
        "门槛 B：稀缺性成立",
        "B2: 50→50→20 μm (5/10/20%)",
        "主战场在 k=5–10%；k=20% 时稀缺问题基本消失",
      ],
      [
        "B3 ≈ B2（关键）",
        "低 k 时 53.8 vs 49.9 μm",
        "未经修正的孪生合成数据存在负迁移——引出第二部分的全部工作",
      ],
    ],
  ),
  p(""),
  tip(
    'B3≈B2 是第一部分留给第二部分的钩子，收尾时重读一遍图 4 的粉色柱：如果直接增强没用，"修正孪生输出"就是自然的下一步——第二部分讲这条路怎么走、以及它教会我们什么。',
  ),
  qa(
    "“训练时磨损目标为什么要 z-score 归一化？”——答：不归一化时 Huber 损失全程线性段收敛慢，且稀缺模型外推会无界爆炸（我们实测过 5000 μm 的预测），这是 W2 冒烟测试抓到的第一个真 bug。",
  ),

  h1("§6 小结与第二部分预告（5 min）"),
  bullet(
    "第一部分交付：可复现的数据管线（26→31 个单测）、验收制脚本（每步有 PASS/FAIL 门槛）、机理孪生（R²≥0.88）、四条基线与两个协议教训。",
  ),
  bullet(
    "方法论要点：防御式解析、不可变数据结构、结构化约束优于惩罚项、门槛先于实验预登记、错了留档不删记录。",
  ),
  bullet(
    '第二部分预告：残差扩散修正孪生的十轮攻防（为什么失败）、换帅为经验贝叶斯池化孪生（为什么简单反而赢）、以及 c4 那把"任何方法都预见不了"的刀。',
  ),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 27, bold: true, font: FONT, color: "1F4E79" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "2E5F8F" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 620, hanging: 320 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({
                  children: [PageNumber.CURRENT],
                  font: FONT,
                  size: 18,
                  color: "888888",
                }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(
    "/Users/jixia/Downloads/dt-toolwear-paper/讲义-第一部分-数据管线与机理孪生.docx",
    b,
  );
  console.log("written", b.length, "bytes");
});
