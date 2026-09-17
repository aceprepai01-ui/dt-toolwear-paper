const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
        ShadingType, Footer, PageNumber } = require('docx');

const FONT = { ascii: 'Calibri', hAnsi: 'Calibri', eastAsia: 'Microsoft YaHei' };
const MONO = { ascii: 'Consolas', hAnsi: 'Consolas', eastAsia: 'Microsoft YaHei' };
const FIGD = '/Users/jixia/Downloads/dt-toolwear-paper/code/paper/lecture_figs';

const p = (t, o = {}) => new Paragraph({ spacing: { after: 120 }, ...o.para,
  children: [new TextRun({ text: t, font: FONT, size: 21, ...o.run })] });
const tip = (t) => new Paragraph({ spacing: { after: 140 }, indent: { left: 280 },
  shading: { fill: 'FFF7E6', type: ShadingType.CLEAR },
  children: [new TextRun({ text: '讲课提示：' + t, font: FONT, size: 19, color: '8C5A00' })] });
const qa = (t) => new Paragraph({ spacing: { after: 140 }, indent: { left: 280 },
  shading: { fill: 'E8F1FB', type: ShadingType.CLEAR },
  children: [new TextRun({ text: '预判提问：' + t, font: FONT, size: 19, color: '1F4E79' })] });
const code = (lines) => lines.map(l => new Paragraph({
  spacing: { after: 0 }, shading: { fill: 'F4F4F4', type: ShadingType.CLEAR },
  children: [new TextRun({ text: l || ' ', font: MONO, size: 17 })] })).concat([p('')]);
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (t) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 60 },
  children: [new TextRun({ text: t, font: FONT, size: 21 })] });
const img = (name, w, h, caption) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(`${FIGD}/${name}.png`),
      transformation: { width: w, height: h },
      altText: { title: name, description: caption, name } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: caption, font: FONT, size: 18, color: '666666' })] })];

const border = { style: BorderStyle.SINGLE, size: 1, color: 'BBBBBB' };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (t, w, head = false) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
  shading: head ? { fill: 'DCE9F5', type: ShadingType.CLEAR } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 19, bold: head })] })] });
const table = (widths, rows) => new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: widths, rows: rows.map((r, i) => new TableRow({ children: r.map((c, j) => cell(c, widths[j], i === 0)) })) });

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
    children: [new TextRun({ text: '刀具磨损预测项目讲义（第二部分）', font: FONT, size: 34, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: '残差扩散攻防 · 预登记仲裁 · 换帅与信息边界', font: FONT, size: 24, color: '4A5568' })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: '时长 60 分钟 ｜ 承接第一部分 ｜ 2026-09', font: FONT, size: 18, color: '777777' })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: '图注标注说明：[存档数字] = 按实验存档结果重绘；[合成示意] = 用合成数据演示已实测的现象（代码与数据文件于 8 月下旬被清理，所有数字取自实验跟踪表与运行报告存档）', font: FONT, size: 16, color: '999999' })] }),

  h1('课程地图与时间分配'),
  table([1100, 5700, 1400, 826], [
    ['节', '内容', '时间', '图'],
    ['§0', '衔接：上一部分留下的钩子', '3 min', '—'],
    ['§1', 'M-DRDC 设计：残差扩散修正孪生', '10 min', 'L5'],
    ['§2', '十轮攻防实录：每一轮修掉一个真 bug', '14 min', 'L6, L7'],
    ['§3', '外部评审处方与最后一搏', '8 min', '—'],
    ['§4', '下游仲裁：预登记规则否决 K1', '8 min', 'L8'],
    ['§5', '换帅：经验贝叶斯池化孪生', '10 min', 'L9'],
    ['§6', '信息边界与全项目方法论', '7 min', 'L10'],
  ]),
  p(''),

  h1('§0 衔接（3 min）'),
  p('第一部分结尾的钩子：B3（纯孪生增强）≈ B2（少样本），低标注时甚至更差——未经修正的仿真数据存在负迁移。自然的下一步：让生成模型学会"修正"孪生的输出。这一部分讲这条路怎么走、为什么最终没走通、以及失败如何变成论文里最硬的两个发现。'),
  tip('开场重放第一部分图 4 的粉色柱（B3），10 秒建立连续性。'),

  h1('§1 M-DRDC 设计（10 min）'),
  ...img('L5_mdrdc_pipeline', 640, 218, '图 L5  M-DRDC 流水线：源刀标定变体库 → 条件残差扩散 → 结构单调解码'),
  h2('1.1 三个设计决策'),
  bullet('残差化：扩散不直接生成磨损路径，只学"孪生与真实的差" r = u实 − u孪。残差低维、平滑、近零均值——把难学的问题变成好学的问题。'),
  bullet('结构单调：解码 w = w₀ + Σsoftplus(u_twin + r̂)，无论 r̂ 是什么，路径必单调——第一部分讲的编码器在这里兑现价值。'),
  bullet('标定变体库：每轮换只有 2-3 条源刀全寿命，不够训练生成模型。对每条源刀做 16-32 次"随机标定比例+测量抖动"的变体标定，把孪生标定的不确定性本身当成数据增强。'),
  ...code([
    '# residual_bank.py 核心（骨架）',
    'for v in range(n_variants):',
    '    jittered = isotonize(real + noise(1.5um))',
    '    idx = label_indices(len, k~U(0.05,0.25))   # 与部署同协议！',
    '    twin = calibrate(jittered[idx], idx).extrapolate(...)',
    '    bank.append(residual = u(jittered) − u(twin), cond = [misfit, ...])']),
  qa('“为什么不直接生成信号波形？”——2026 年 7 月的竞品预印本就是直接生成（FiLM 条件注入），它没有退化一致性保证；我们的差异化正是残差+结构。这也是查新逼出来的设计。'),

  h1('§2 十轮攻防实录（14 min，本讲核心）'),
  ...img('L6_campaign', 640, 265, '图 L6  [存档数字] 攻防轨迹：聚合穿越 CRPS 从 40.5 修到 27.0，但始终未跨过孪生基准 24.3'),
  h2('2.1 四个代表性 bug（每个都值得单讲）'),
  table([1500, 3700, 3826], [
    ['轮次', '症状', '修复与教训'],
    ['v1 删失计分', 'c1 孪生"误差 315 刀"——其实是没穿越阈值被记成整个寿命', '删失路径按"寿命末端穿越"计。教训：对照双方必须同一套计分规则'],
    ['v2 协议错配', '训练库残差比部署所需大 2 倍，修正器把好孪生改坏', '库的标定协议改成与部署一致（均匀 k%）。教训：训练分布要模拟部署分布'],
    ['v3 u 空间尖峰', '63 个位置里 51 个方差异常，残差学习被伪影支配', 'ENC_EPS=0.05 μm 下限。教训：数学正确 ≠ 数值健康'],
    ['v3 逐位置归一化', '残差强异方差（位置 std 相差 5 倍）被全局标量抹平', '均值/方差改为逐位置向量。教训：归一化的粒度是建模决策'],
  ]),
  p(''),
  ...img('L7_uspike', 620, 205, '图 L7  [合成示意] u 尖峰现象：实测中 u_real 最小达 −13.8（存档），下限修复后有界于 ≈−3'),
  h2('2.2 每轮修复后的诊断纪律'),
  bullet('先无训练诊断（分布统计、几何检查），再花 GPU——u 尖峰就是纯诊断脚本抓到的。'),
  bullet('单种子结论不算数：c1 曾在两次运行间从 4.3 摆到 32.8，之后所有判定一律 3 种子起。'),
  bullet('修复必须留档在代码注释里——审稿人问"为什么这样设计"时，答案是踩坑记录不是事后编造。'),
  tip('这一节的讲法是"侦探故事"：先展示症状让学生猜原因，再揭诊断数据。删失 bug 和 u 尖峰两个案例互动效果最好。'),

  h1('§3 外部评审处方与最后一搏（8 min）'),
  p('v3 三种子稳定后仍差基准 5 刀，我们把完整档案（含失败数据）交给外部模型评审（GPT-5.4，xhigh 推理），它的诊断一针见血：96 个库样本本质只是 3 个残差家族的扰动，隐式条件学习在学一个噪声全局插值器。处方四条：'),
  bullet('① 按尾段斜率分域检索（平台/爬升/陡升），只用同域样本训练；'),
  bullet('② 核加权：让"像目标"的样本主导损失；'),
  bullet('③ 条件砍到 2 维 [失配, 斜率]，删掉欠定维度；'),
  bullet('④ 拒绝选项：修正量不显著（|中位穿越偏移| < max(5刀, 1.5×IQR)）就原样输出孪生——保护好孪生。'),
  p('结果（v4，三种子稳定）：26.3–27.6，仍未过 24.3。而且拒绝选项开反了火：它拒了最该修的 c4/c6，却对最不该动的 c1 施加了修正。根因浮出水面：**源刀只有 3-4 把时，任何"按相似度找同类"的策略，每个目标能找到的同类只有 1-2 把**——小 N 墙，不是工程 bug。'),
  qa('“为什么不再多试几种生成模型（flow matching、VAE）？”——评审原话：换求解器解决不了支撑集问题。这是"方法学花样"与"数据体制极限"的区别，识别这个区别本身是研究能力。'),

  h1('§4 下游仲裁：预登记规则否决 K1（8 min）'),
  p('关键认知：穿越 CRPS 只是内部分段门槛，论文的真声明（K1）是下游——修正路径当训练数据，能否让 TCN 在稀缺标注下学得更好。仲裁前先把判定规则写死在脚本头部（预登记）："OURS 必须在 k=5% 和 10% 两档同时以 ±1 SEM 不重叠击败 B3 和 B2"。'),
  ...img('L8_arbitration', 600, 264, '图 L8  [存档数字] 仲裁结果：三系统两档全部纠缠在误差棒内，K1 否决'),
  bullet('唯一的火种：c4@10% 某种子上修正被施加时 MAE 24.1（全表最佳）——"弱孪生可救"有孤证，但 9 格里只有 1 格。'),
  bullet('为什么预登记重要：结果出来后人总能找到"换个指标就赢了"的角度；规则先行让否决无可辩驳，也让论文的负结果具有可信度。'),
  tip('这里放慢：预登记是整门课最想传递的科研习惯。可以现场问学生"如果没有预登记，你会怎么给这组数字找补？"'),

  h1('§5 换帅：经验贝叶斯池化孪生（10 min）'),
  p('复盘所有证据后的洞察：物理孪生贡献了全部可提取信号，瓶颈是"单刀标定看不到别的刀的晚期知识"。最小对症机制不是生成模型，而是把源刀的标定结果池化成先验：'),
  ...code([
    '# hierarchical.py 核心（骨架）',
    'fits = [calibrate(w) for w in source_full_lives]   # 每刀独立 MAP',
    'hyper.mu  = mean(θ_fits);  hyper.tau = std(θ_fits) # 经验超先验',
    'θ_target = MAP(target_k%_labels, prior=hyper)       # 目标在人群先验下标定',
    '# 确定性、零 GPU、无种子——整条主方法是闭式的']),
  ...img('L9_reframe', 640, 260, '图 L9  [存档数字] 换帅后头条：池化孪生 22.7/25.0 μm，贴近 80% 标注 DL 上界 21.0，约两倍优于全部 DL 系统'),
  table([2400, 3300, 3326], [
    ['防线', '存档结果', '含义'],
    ['留一源刀 (H3)', 'c1: 6.2–6.6；c4: 41.3–45.9；c6: 22.3–24.3', '去掉任何一把源刀结论不变'],
    ['先验尺度 ×0.5/1/2 (H4)', '24.83 / 25.01 / 24.73', '对超先验宽度不敏感'],
    ['零样本对照 (H1)', '人群均值参数直接外推 → 发散', '目标标定不可省，池化是先验不是替身'],
  ]),
  p(''),
  bullet('十轮扩散攻防没有浪费：它变成论文第二贡献（受控负结果），扩散基线的全部基础设施变成对照实验。'),
  qa('“这不就是普通的分层贝叶斯吗？”——是。查新确认 2026 年 1 月（通用 PHM）和 6 月（钻削）已有 HB 先例，所以论文里 HB 定位为"借用的工具"，声明的是证据架构：公开数据+匹配预算的物理 vs DL 正面对决+预登记负结果+失效边界。'),

  h1('§6 信息边界与全项目方法论（7 min）'),
  ...img('L10_boundary', 640, 211, '图 L10  (a) [合成示意] c4 型状态切换发生在标注池之后；(b) [存档数字] 两个部署可观测探测器均无区分度'),
  p('c4 的末段 78 μm 暴涨（实测 132.7→210.9 μm）在任何 ≤20% 标注预算内原理性不可观测：孪生拟合、标注窗失配旗标、先验敏感度分歧三重探测全部失败。共形区间在这类刀上必然欠覆盖（存档 PICP 0.16–0.29 vs 名义 0.9）。这不是方法缺陷，是标签仅有方法的信息边界——论文据此论证"保守区间或最低限度传感监测"的必要性。'),
  h2('全项目方法论清单（结课带走的东西）'),
  bullet('门槛先行、预登记判定，负结果照单全收并写进论文；'),
  bullet('对照实验的公平性架构（同骨干、同预算、同协议）比模型本身更难设计也更值钱；'),
  bullet('每次失败先做无训练诊断，修复留档在代码注释；单种子不下结论；'),
  bullet('"最小对症机制"哲学：十轮攻防的终点是承认 0.19M 参数的扩散不如一个闭式先验——简单赢了，且我们有完整证据链说明为什么；'),
  bullet('外部对抗评审（查新、方案、诊断处方）贯穿全程——自己的判断需要制度化的挑战者。'),
  tip('结课回到第一部分图 1a 那三条曲线："我们从三条曲线出发，最后学到的是——这三条曲线里到底有多少信息、以及信息不够时诚实的科研长什么样。"'),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 27, bold: true, font: FONT, color: '1F4E79' },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: '2E5F8F' },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
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

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync('/Users/jixia/Downloads/dt-toolwear-paper/讲义-第二部分-扩散攻防与换帅.docx', b);
  console.log('written', b.length, 'bytes');
});
