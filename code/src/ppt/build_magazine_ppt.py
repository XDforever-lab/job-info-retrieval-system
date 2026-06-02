"""Build PPT with Magazine template - only uses template-defined CSS variables"""
import os, re

TEMPLATE = os.path.join(os.path.expanduser("~"), ".claude/skills/guizang-ppt-skill/assets/template.html")
OUT = "C:/Users/ZhangQichen/Desktop/recruit/code/ppt/index.html"

with open(TEMPLATE, encoding="utf-8") as f:
    html = f.read()

# Fix title
html = re.sub(r'<title>.*?</title>', '<title>上市公司招聘信息检索系统 · 开题汇报</title>', html)

# Remove module script block (won't work with file://)
cut = html.find('<!-- ============ Motion One')
html = html[:cut]

# CSS fix: add --accent variable for consistency
accent_css = """
    --accent:#c75117;
    --sp-3:8px;--sp-4:12px;--sp-5:16px;--sp-6:24px;--sp-7:32px;--sp-8:40px;--sp-9:48px;--sp-10:64px;
"""
html = html.replace("--paper-tint:#e8e5de;", "--paper-tint:#e8e5de;" + accent_css)

# ── All 10 slides ──
A = "var(--accent)"  # shortcut for readability
P = "var(--paper)"
PR = "var(--paper-rgb)"

slides = f"""
<!-- ====== P1: Cover ====== -->
<section class="slide hero dark" data-theme="dark">
  <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;height:100%;padding:48px;text-align:center">
    <div style="font-family:var(--mono);font-size:13px;letter-spacing:.22em;text-transform:uppercase;color:{A};margin-bottom:24px">信息检索系统实验 · 期末大作业</div>
    <h1 style="font-family:var(--serif-zh);font-weight:700;font-size:min(8vw,14vh);line-height:1.12;color:{P};margin-bottom:32px">上市公司招聘<br>信息检索系统</h1>
    <p style="font-family:var(--sans-zh);font-size:min(2vw,20px);color:rgba({PR},.5);max-width:44ch;line-height:1.6">中国 A 股上市公司 2026 年招聘发布数据<br>面向求职者与行业研究者的垂直领域检索系统</p>
    <div style="margin-top:48px;font-family:var(--mono);font-size:11px;letter-spacing:.2em;color:rgba({PR},.25)">南京农业大学 · 信息检索系统实验 · 2026.06</div>
  </div>
</section>

<!-- ====== P2: Corpus Scale ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">02 / 10 · 语料特征与规模</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(4vw,7vh);color:{P};margin-bottom:32px">语料特征与规模</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:32px">
      <div style="background:rgba({PR},.05);padding:28px 24px;border-left:2px solid {A}"><div style="font-family:var(--serif-zh);font-size:min(4.4vw,6.4vh);color:{P};font-weight:300">5,000</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">清洗后记录数</div></div>
      <div style="background:rgba({PR},.05);padding:28px 24px;border-left:2px solid {A}"><div style="font-family:var(--serif-zh);font-size:min(4.4vw,6.4vh);color:{P};font-weight:300">14</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">保留字段</div></div>
      <div style="background:rgba({PR},.05);padding:28px 24px;border-left:2px solid {A}"><div style="font-family:var(--serif-zh);font-size:min(4.4vw,6.4vh);color:{P};font-weight:300">9</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">行业大类</div></div>
      <div style="background:rgba({PR},.05);padding:28px 24px;border-left:2px solid {A}"><div style="font-family:var(--serif-zh);font-size:min(4.4vw,6.4vh);color:{P};font-weight:300">965</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">停用词</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:32px;flex:1">
      <div><h3 style="font-family:var(--sans-zh);font-size:18px;font-weight:600;color:{A};margin-bottom:12px">数据来源与处理</h3>
        <p style="font-family:var(--sans-zh);font-size:15px;color:rgba({PR},.65);line-height:1.8">原始数据：中国 A 股上市公司 2026 年招聘数据 (macrodatas.cn)<br>原始规模：93,602 条 · 81 个行业 · 8,796 家企业<br>清洗后：<strong style="color:{P}">5,000 条</strong>（GBK→UTF-8，去广告/URL/空白）<br>分词方案：CRF 条件随机场（金标准200条评测 F1=74.77%）<br>停用词表：Top-500人工审查 + 哈工大 + 百度 → 965词</p></div>
      <div><h3 style="font-family:var(--sans-zh);font-size:18px;font-weight:600;color:{A};margin-bottom:12px">9 大行业分类</h3>
        <p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.9">互联网/软硬件技术 · 销售/市场/客服<br>人力/财务/行政 · 生产制造/汽车<br>金融 · 房地产/建筑 · 物流/贸易/采购<br>医疗/能源/环保/农业<br>文化传媒/教育/生活服务</p></div>
    </div>
  </div>
</section>

<!-- ====== P3: N-gram & Zipf ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">03 / 10 · 语料分析</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(4vw,7vh);color:{P};margin-bottom:32px">N-Gram 统计与齐普夫定律验证</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:40px">
      <div style="background:rgba({PR},.08);padding:24px;text-align:center"><div style="font-family:var(--serif-zh);font-size:min(5vw,7vh);color:{P};font-weight:200">1,049,179</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">总词次 (Tokens)</div></div>
      <div style="background:rgba({PR},.08);padding:24px;text-align:center"><div style="font-family:var(--serif-zh);font-size:min(5vw,7vh);color:{P};font-weight:200">32,365</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">唯一词型 (Types)</div></div>
      <div style="background:rgba({PR},.08);padding:24px;text-align:center"><div style="font-family:var(--serif-zh);font-size:min(5vw,7vh);color:{P};font-weight:200">965</div><div style="font-family:var(--mono);font-size:11px;color:{A};margin-top:8px">停用词数量</div></div>
      <div style="background:{A};padding:24px;text-align:center"><div style="font-family:var(--serif-zh);font-size:min(4.2vw,6vh);color:{P};font-weight:400">R² > 0.85</div><div style="font-family:var(--mono);font-size:11px;color:rgba({PR},.7);margin-top:8px">Zipf 拟合优度</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:28px;flex:1">
      <div style="border-top:1px solid rgba({PR},.12);padding-top:20px"><span style="font-family:var(--mono);font-size:28px;color:{A}">01</span><h3 style="font-family:var(--sans-zh);font-size:17px;color:{P};margin:8px 0">Unigram 词频统计</h3><p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.7">5,000条职位描述 CRF 分词后统计。Top-500词中高频词以标点(11.4%)、虚词(的/及/与)、招聘套话(工作/负责/岗位)为主。</p></div>
      <div style="border-top:1px solid rgba({PR},.12);padding-top:20px"><span style="font-family:var(--mono);font-size:28px;color:{A}">02</span><h3 style="font-family:var(--sans-zh);font-size:17px;color:{P};margin:8px 0">齐普夫定律验证</h3><p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.7">词频与排名双对数坐标下呈近似线性，R² > 0.85。招聘文本词频分布符合 Zipf 定律：词频 × 排名 ≈ 常数。</p></div>
      <div style="border-top:1px solid rgba({PR},.12);padding-top:20px"><span style="font-family:var(--mono);font-size:28px;color:{A}">03</span><h3 style="font-family:var(--sans-zh);font-size:17px;color:{P};margin:8px 0">停用词构建</h3><p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.7">Top-500人工审查 + 哈工大通用词表 + 百度词表 → 合并去重 965 词。高频虚词(具备/负责/熟悉)对检索无区分价值。</p></div>
    </div>
  </div>
</section>

<!-- ====== P4: User Needs (3 Personas) ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">04 / 10 · 战略层</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(4vw,7vh);color:{P};margin-bottom:32px">战略层：用户需求分析</h2>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;flex:1">
      <div style="background:rgba({PR},.04);padding:28px;display:flex;flex-direction:column">
        <div style="font-family:var(--mono);font-size:11px;color:{A};text-transform:uppercase;letter-spacing:.14em;margin-bottom:16px">Persona 1</div>
        <h3 style="font-family:var(--serif-zh);font-size:min(2.2vw,26px);color:{P};margin-bottom:12px">应届生小李</h3>
        <p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.7;flex:1">22 岁 · 计算机本科应届<br>目标：寻找 IT/互联网行业校招岗位<br>核心需求：<br>① 按行业筛选 IT 相关岗位<br>② 按城市筛选工作地点<br>③ 关键词搜索(Python、机器学习)<br>④ 按薪资范围排序</p>
        <div style="font-family:var(--mono);font-size:10px;color:rgba({PR},.25);margin-top:12px">使用场景：每晚浏览新发布岗位</div>
      </div>
      <div style="background:rgba({PR},.04);padding:28px;display:flex;flex-direction:column">
        <div style="font-family:var(--mono);font-size:11px;color:{A};text-transform:uppercase;letter-spacing:.14em;margin-bottom:16px">Persona 2</div>
        <h3 style="font-family:var(--serif-zh);font-size:min(2.2vw,26px);color:{P};margin-bottom:12px">求职者小赵</h3>
        <p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.7;flex:1">28 岁 · 3年经验 · 在职看机会<br>目标：关注薪资+稳定性+成长空间<br>核心需求：<br>① 薪资区间精确筛选<br>② 学历+经验多维度过滤<br>③ 相似岗位推荐<br>④ 上市公司偏好（福利规范）</p>
        <div style="font-family:var(--mono);font-size:10px;color:rgba({PR},.25);margin-top:12px">使用场景：周末系统搜索</div>
      </div>
      <div style="background:rgba({PR},.04);padding:28px;display:flex;flex-direction:column">
        <div style="font-family:var(--mono);font-size:11px;color:{A};text-transform:uppercase;letter-spacing:.14em;margin-bottom:16px">Persona 3</div>
        <h3 style="font-family:var(--serif-zh);font-size:min(2.2vw,26px);color:{P};margin-bottom:12px">分析师张工</h3>
        <p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.55);line-height:1.7;flex:1">35 岁 · 行业研究分析师<br>目标：通过招聘数据洞察行业趋势<br>核心需求：<br>① 按行业浏览招聘规模与结构<br>② 跨行业薪资水平对比<br>③ 学历/经验要求差异分析<br>④ 关键词趋势分析</p>
        <div style="font-family:var(--mono);font-size:10px;color:rgba({PR},.25);margin-top:12px">使用场景：季度行业趋势报告</div>
      </div>
    </div>
  </div>
</section>

<!-- ====== P5: Product Goals ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">05 / 10 · 战略层</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(4vw,7vh);color:{P};margin-bottom:32px">战略层：产品目标</h2>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px;flex:1;align-items:start">
      <div><div style="font-family:var(--mono);font-size:56px;color:{A};font-weight:200;line-height:1">01</div><h3 style="font-family:var(--serif-zh);font-size:min(2vw,24px);color:{P};margin:12px 0">学术目标</h3><p style="font-family:var(--sans-zh);font-size:15px;color:rgba({PR},.6);line-height:1.7">完成信息检索课程综合实验要求，完整展现<strong style="color:{P}">信息组织、检索、展示</strong>三阶段以及六次实验的全部知识应用</p></div>
      <div><div style="font-family:var(--mono);font-size:56px;color:{A};font-weight:200;line-height:1">02</div><h3 style="font-family:var(--serif-zh);font-size:min(2vw,24px);color:{P};margin:12px 0">能力目标</h3><p style="font-family:var(--sans-zh);font-size:15px;color:rgba({PR},.6);line-height:1.7">掌握从<strong style="color:{P}">数据预处理</strong>到<strong style="color:{P}">系统开发</strong>到<strong style="color:{P}">评测迭代</strong>的完整信息检索系统构建流程</p></div>
      <div><div style="font-family:var(--mono);font-size:56px;color:{A};font-weight:200;line-height:1">03</div><h3 style="font-family:var(--serif-zh);font-size:min(2vw,24px);color:{P};margin:12px 0">展示目标</h3><p style="font-family:var(--sans-zh);font-size:15px;color:rgba({PR},.6);line-height:1.7">产出<strong style="color:{P}">可演示、可交互</strong>的 Web 检索系统<br>遵循<strong style="color:{P}">五层产品设计模型</strong><br>战略→范围→结构→框架→表现</p></div>
    </div>
  </div>
</section>

<!-- ====== P6: Scope Layer Pt.1 (Functions 1-6) ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">06 / 10 · 范围层 (1/2)</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(3.6vw,6vh);color:{P};margin-bottom:24px">范围层：功能规格 — 核心检索功能</h2>
    <p style="font-family:var(--sans-zh);font-size:15px;color:rgba({PR},.5);margin-bottom:32px">10 大功能模块 · 覆盖信息存储与组织、信息检索、信息展示全链路</p>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;flex:1">
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">F1 · 多字段联合检索</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">VSM + TF-IDF</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">用户输入关键词 → 倒排索引 → TF-IDF余弦相似度排序<br>标题字段 2.0x 权重加成（"Python"匹配标题>正文）</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">F2 · 结构化筛选</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">布尔逻辑检索</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">行业(多选) × 城市 × 学历 × 经验 × 薪资范围(滑块) × 日期 → AND 组合过滤</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">F3 · 全文高亮 + F4 · 排序</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">信息展示与排序</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">查询词黄色高亮标记 · 四种排序：相关度(TF-IDF) / 发布日期 / 最低月薪 / 最高月薪</p></div>
      <div style="background:{A};padding:22px"><div style="font-family:var(--mono);font-size:11px;color:rgba({PR},.7);margin-bottom:8px">F5 · 行业标签浏览</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">NB 文本分类</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.7);line-height:1.6">400条人工标注9类 → 320训练/80测试 → NB (Micro F1=25%) vs KNN (16%) → NB 对5k全量分类 → 首页9行业标签入口</p></div>
      <div style="background:{A};padding:22px"><div style="font-family:var(--mono);font-size:11px;color:rgba({PR},.7);margin-bottom:8px">F6 · 技能族群聚类</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">K-Means 词汇聚类</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.7);line-height:1.6">800高频词 × 5000文档词矩阵 → L2归一化 → K-Means/等级聚类 → 10个技能族群 → 词云矩阵+PCA散点图</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">检索评测</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">P/R/F1/MAP</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">20条测试查询 · 人工标注相关文档 · P@10 · MAP · NDCG · 2×2表格 · 11点插值P/R曲线</p></div>
    </div>
  </div>
</section>

<!-- ====== P7: Scope Layer Pt.2 (Functions 7-10 + Experiments) ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">07 / 10 · 范围层 (2/2)</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(3.6vw,6vh);color:{P};margin-bottom:24px">范围层：功能规格 — 智能服务与评测</h2>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;flex:1">
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">F7 · 相似岗位推荐</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">TF-IDF 余弦相似度</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">5,000×32,365 TF-IDF 稀疏矩阵<br>逐文档批量 cos 计算 → Top-5<br>相似度 0.27~0.34（短文本正常）<br>本质即 VSM 检索模型</p></div>
      <div style="background:{A};padding:22px"><div style="font-family:var(--mono);font-size:11px;color:rgba({PR},.7);margin-bottom:8px">F8 · 关键词抽取 ★ 实验四</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">TF-IDF + TextRank</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.7);line-height:1.6">TF-IDF: F1=44.61% (P=32.97% R=76.11%)<br>TextRank: F1=41.92% (窗口=5,阻尼=0.85)<br>100条金标准评测Pooling融合<br>详情页Top-15关键词标签可点击搜索</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">F9 · 数据统计仪表盘</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">多维度可视化</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">行业招聘数量排行榜(柱状图)<br>城市招聘热度Top-10<br>薪资分布直方图(5k区间)<br>学历/经验要求饼图</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><div style="font-family:var(--mono);font-size:11px;color:{A};margin-bottom:8px">F10 · 分词效果展示</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">7 种分词器对比</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">FMM · BMM · BiMM · MinSeg<br>HMM · Ngram · CRF<br>BIES 字符级 + 集合元素化双评测<br>详情页模态弹窗对比</p></div>
      <div style="background:{A};padding:22px"><div style="font-family:var(--mono);font-size:11px;color:rgba({PR},.7);margin-bottom:8px">实验五 · 文本分类</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">NB vs KNN · 9类</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.7);line-height:1.6">400条标注 8:2 分层抽样<br>手写余弦KNN (最优K=3)<br>手写多项式NB (最优α=0.1)<br>宏/微平均 P/R/F1 + BEP<br>NB 微平均 F1=25% 较优</p></div>
      <div style="background:{A};padding:22px"><div style="font-family:var(--mono);font-size:11px;color:rgba({PR},.7);margin-bottom:8px">实验六 · 词汇聚类</div><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin-bottom:8px">10个技能族群</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.7);line-height:1.6">K-Means (K=6~12网格搜索)<br>等级聚类 (Ward/Single/Complete)<br>簇间分离度 0.78<br>簇内紧密度 0.15<br>词云矩阵 + PCA 散点图</p></div>
    </div>
  </div>
</section>

<!-- ====== P8: Key Data Results ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">08 / 10 · 核心实验数据</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(3.6vw,6vh);color:{P};margin-bottom:24px">核心实验结果汇总</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;flex:1">
      <div>
        <h3 style="font-family:var(--sans-zh);font-size:16px;color:{A};margin-bottom:16px">分词评测 (200条金标准)</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;font-family:var(--mono);font-size:11px;color:rgba({PR},.5)">
          <div style="color:{A}">模型</div><div style="color:{A}">P(%)</div><div style="color:{A}">R(%)</div><div style="color:{A}">F1(%)</div>
          <div>FMM</div><div>70.98</div><div>62.49</div><div style="color:{P}">66.46</div>
          <div>BMM</div><div>71.23</div><div>62.85</div><div style="color:{P}">66.78</div>
          <div>BiMM</div><div>71.87</div><div>63.20</div><div style="color:{P}">67.26</div>
          <div>CRF</div><div>78.76</div><div>71.16</div><div style="color:{A};font-weight:600">74.77</div>
          <div>Ngram</div><div>78.43</div><div>69.04</div><div style="color:{P}">73.43</div>
        </div>
        <div style="margin-top:24px"><h3 style="font-family:var(--sans-zh);font-size:16px;color:{A};margin-bottom:12px">10 个技能族群 (800词 K-Means)</h3>
        <p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.5);line-height:1.7">设备制造(85词) · 采购管理(132词) · HR招聘(26词) · 生产质量(330词) · 研发投资(49词) · 财务审计(22词) · 电商运营(49词) · 食品安全(12词) · 数据AI(52词) · 客户销售(43词)</p></div>
      </div>
      <div>
        <h3 style="font-family:var(--sans-zh);font-size:16px;color:{A};margin-bottom:16px">关键词评测 (100条金标准)</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;font-family:var(--mono);font-size:11px;color:rgba({PR},.5)">
          <div style="color:{A}">算法</div><div style="color:{A}">P(%)</div><div style="color:{A}">R(%)</div><div style="color:{A}">F1(%)</div>
          <div>TF-IDF</div><div>32.97</div><div>76.11</div><div style="color:{A};font-weight:600">44.61</div>
          <div>TextRank</div><div>30.87</div><div>72.82</div><div style="color:{P}">41.92</div>
        </div>
        <div style="margin-top:24px"><h3 style="font-family:var(--sans-zh);font-size:16px;color:{A};margin-bottom:12px">分类评测 (400条标注 9类)</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-family:var(--mono);font-size:11px;color:rgba({PR},.5)">
          <div style="color:{A}">模型</div><div style="color:{A}">宏 F1</div><div style="color:{A}">微 F1</div>
          <div>KNN(K=3)</div><div>15.28%</div><div style="color:{P}">16.25%</div>
          <div>NB(α=0.1)</div><div>18.36%</div><div style="color:{A};font-weight:600">25.00%</div>
        </div>
        <p style="font-family:var(--sans-zh);font-size:12px;color:rgba({PR},.35);margin-top:8px">注：9类×320训练≈36条/类，随机基线11%，NB 微平均显著领先</p></div>
      </div>
    </div>
  </div>
</section>

<!-- ====== P9: Structure Layer ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">09 / 10 · 结构层</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(3.6vw,6vh);color:{P};margin-bottom:32px">结构层：页面架构与交互设计</h2>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;flex:1">
      <div style="background:rgba({PR},.04);padding:22px"><span style="font-family:var(--mono);font-size:32px;color:{A};font-weight:200">01</span><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin:8px 0">首页 · 搜索门户</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">搜索框 + 快捷筛选(行业/城市/学历/薪资) + 9大行业标签入口卡片 + 最新招聘滚动展示</p></div>
      <div style="background:rgba({PR},.04);padding:22px"><span style="font-family:var(--mono);font-size:32px;color:{A};font-weight:200">02</span><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin:8px 0">检索结果页</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">左：筛选面板(布尔过滤) · 右：结果列表(VSM排序) · 排序切换 · 分页器 · "找到N条结果，用时X秒"</p></div>
      <div style="background:rgba({PR},.04);padding:22px"><span style="font-family:var(--mono);font-size:32px;color:{A};font-weight:200">03</span><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin:8px 0">职位详情页</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">基本信息区 · 完整描述(高亮) · 关键词标签(可点搜索) · 分类预测 · 分词效果弹窗 · 相似推荐</p></div>
      <div style="background:rgba({PR},.04);padding:22px"><span style="font-family:var(--mono);font-size:14px;color:{A}">04-07</span><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin:8px 0">辅助页面</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.6">行业浏览(9大类→职位列表) · 数据统计(5类图表) · 聚类视图(词云+PCA散点) · 关于(AI声明+组员)</p></div>
      <div style="background:{A};padding:22px;grid-column:span 2"><span style="font-family:var(--mono);font-size:14px;color:rgba({PR},.7)">交互设计</span><h3 style="font-family:var(--serif-zh);font-size:18px;color:{P};margin:8px 0">用户旅程</h3><p style="font-family:var(--sans-zh);font-size:14px;color:rgba({PR},.7);line-height:1.6">首页搜索 → 筛选条件 → 浏览结果(排序+分页) → 点击卡片 → 详情页(高亮+标签+分类预测+分词效果+相似推荐) → 返回继续浏览 / 修改条件重新搜索</p></div>
    </div>
  </div>
</section>

<!-- ====== P10: Framework + Surface ====== -->
<section class="slide dark" data-theme="dark">
  <div style="padding:48px 56px;height:100%;display:flex;flex-direction:column">
    <div style="font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:{A};margin-bottom:20px">10 / 10 · 框架层 & 表现层</div>
    <h2 style="font-family:var(--serif-zh);font-weight:600;font-size:min(3.6vw,6vh);color:{P};margin-bottom:32px">框架层 & 表现层</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;flex:1">
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><h3 style="font-family:var(--serif-zh);font-size:17px;color:{P};margin-bottom:12px">框架层 · 线框设计</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.7">首页搜索居中<br>结果页左筛选右列表<br>详情页上信息下描述<br>Flask + Jinja2 模板</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><h3 style="font-family:var(--serif-zh);font-size:17px;color:{P};margin-bottom:12px">框架层 · 导航设计</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.7">顶部固定导航栏<br>搜索|行业浏览|统计|聚类|关于<br>面包屑返回<br>页脚统一</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><h3 style="font-family:var(--serif-zh);font-size:17px;color:{P};margin-bottom:12px">表现层 · 视觉设计</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.7">主色 #1a73e8 蓝<br>辅色 #34a853 绿<br>背景 #f8f9fa<br>字体 微软雅黑</p></div>
      <div style="background:rgba({PR},.04);padding:22px;border-top:2px solid {A}"><h3 style="font-family:var(--serif-zh);font-size:17px;color:{P};margin-bottom:12px">表现层 · 一致性</h3><p style="font-family:var(--sans-zh);font-size:13px;color:rgba({PR},.55);line-height:1.7">全局统一配色/字体<br>卡片模板统一<br>所有页面继承 base.html<br>底部版权+AI声明</p></div>
    </div>
  </div>
</section>
"""

html = html.replace('<!-- SLIDES_HERE -->', slides)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Done: {OUT} ({len(html):,} bytes, {slides.count('<section class=')} slides)")
