"""生成 PPT HTML —— 严格使用 Swiss 模板定义的 CSS 类名"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *  # noqa: F403

TEMPLATE = os.path.join(os.path.expanduser("~"), ".claude/skills/guizang-ppt-skill/assets/template-swiss.html")
PPT_OUT = "C:/Users/ZhangQichen/Desktop/recruit/code/ppt/index.html"

# Load data
kw_eval = json.load(open(os.path.join(DIR_10_KW, "eval_report.json"), encoding="utf-8"))

# Read template
with open(TEMPLATE, encoding="utf-8") as f:
    html = f.read()

# Fix title
import re
html = re.sub(r'<title>.*?</title>', '<title>上市公司招聘信息检索系统 · 开题汇报</title>', html)

# ── Build slides using ONLY template-defined classes ──
slides = ""

# ===== P1: Hero Cover (S01) =====
# Copy exact template structure, replace content only
slides += """
<section class="slide accent" data-animate="hero">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    <div class="chrome-min">
      <div class="l">上市公司招聘信息检索系统</div>
      <div class="r">SS · 2026.06 · 01 / 10</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.6vh">
      <div data-anim="kicker" class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.22em">信息检索系统实验 · 期末大作业</div>
      <h1 data-anim="title" style="align-self:center;font-family:var(--sans),var(--sans-zh);font-weight:200;font-size:min(11.6vw,19vh);line-height:.94;letter-spacing:-.025em;color:#fff">上市公司招聘<br>信息检索系统</h1>
      <div data-anim="bottom" style="display:grid;grid-template-rows:auto auto;gap:1.6vh;border-top:1px solid rgba(255,255,255,.22);padding-top:2vh">
        <div data-anim="lead" class="lead" style="max-width:52ch;color:rgba(255,255,255,.86)">中国 A 股上市公司 2026 年招聘数据 · 垂直领域检索系统</div>
        <div style="display:flex;justify-content:space-between;align-items:end">
          <div class="t-meta" style="color:rgba(255,255,255,.6)">南京农业大学 · 06/2026</div>
          <div class="t-meta" style="color:rgba(255,255,255,.6)">← → 翻页 · ESC 索引</div>
        </div>
      </div>
    </div>
  </div>
</section>
"""

# ===== P2: Corpus Scale (S21 Tech Spec) =====
slides += """
<section class="slide light" data-animate="spec-in">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">语料特征与规模</div><div class="r">02 / 10</div></div>
    <div style="padding-top:3.2vh">
      <h1 class="h-xl" style="margin-bottom:1.6vh">语料特征与规模</h1>
      <p class="lead" style="max-width:56ch;margin-bottom:4vh">中国 A 股上市公司 2026 年招聘发布数据，原始 93,602 条经清洗抽样后保留 5,000 条</p>
      <div class="grid-12" style="gap:var(--sp-6);margin-bottom:var(--sp-8)">
        <div class="card-outlined span-3"><span class="num-mega">5,000</span><span class="t-cat">清洗后记录数</span></div>
        <div class="card-outlined span-3"><span class="num-mega">14</span><span class="t-cat">保留字段</span></div>
        <div class="card-outlined span-3"><span class="num-mega">9</span><span class="t-cat">行业大类</span></div>
        <div class="card-outlined span-3"><span class="num-mega">965</span><span class="t-cat">停用词</span></div>
      </div>
      <div class="spec-bars" style="grid-template-columns:140px 1fr;gap:1.2vh 2vw">
        <span class="t-cat">数据来源</span><span>macrodatas.cn 上市公司招聘发布数据 (GBK → UTF-8)</span>
        <span class="t-cat">原始规模</span><span>93,602 条 · 81 个行业 · 8,796 家企业</span>
        <span class="t-cat">核心字段</span><span>企业名称 / 股票简称 / 行业 / 城市 / 月薪 / 学历 / 经验 / 职位描述 / 发布日期</span>
        <span class="t-cat">9大行业</span><span>互联网软硬件 · 销售市场客服 · 人力财务行政 · 生产制造汽车 · 金融 · 房地产建筑 · 物流贸易采购 · 医疗能源环保农业 · 文化传媒教育生活</span>
      </div>
    </div>
  </div>
</section>
"""

# ===== P3: N-gram + Zipf (S02 Timeline) =====
slides += """
<section class="slide light" data-animate="pipeline">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">语料分析</div><div class="r">03 / 10</div></div>
    <h1 class="h-xl" style="margin-bottom:4vh">N-Gram 统计 · 齐普夫定律</h1>
    <div class="grid-12" style="margin-bottom:var(--sp-7)">
      <div class="card-ink span-3"><span class="num-mega" style="font-size:6vw">1.05M</span><span class="t-cat" style="color:var(--paper)">总词次</span></div>
      <div class="card-ink span-3"><span class="num-mega" style="font-size:6vw">32k</span><span class="t-cat" style="color:var(--paper)">唯一词型</span></div>
      <div class="card-ink span-3"><span class="num-mega" style="font-size:6vw">965</span><span class="t-cat" style="color:var(--paper)">停用词</span></div>
      <div class="card-accent span-3"><span class="num-mega" style="font-size:5.2vw">Zipf ✓</span><span class="t-cat" style="color:var(--accent-on)">齐普夫验证</span></div>
    </div>
    <div class="timeline-v">
      <div class="tl-node"><div class="dot"></div><div class="tl-axis"><span class="t-cat">01</span><h3 class="h-md" style="margin:0">Unigram 词频统计</h3><p class="body-sm">对全部 5,000 条职位描述分词后统计词频。Top-500 中高频词以标点 (11.4%)、虚词 (的/及/与)、招聘套话 (工作/负责/岗位) 为主。</p></div></div>
      <div class="tl-node"><div class="dot"></div><div class="tl-axis"><span class="t-cat">02</span><h3 class="h-md" style="margin:0">齐普夫定律验证</h3><p class="body-sm">词频与排名在双对数坐标下呈线性关系，斜率和 R² 均表明招聘文本符合 Zipf 定律——词频 × 排名 ≈ 常数。</p></div></div>
      <div class="tl-node"><div class="dot"></div><div class="tl-axis"><span class="t-cat">03</span><h3 class="h-md" style="margin:0">停用词构建</h3><p class="body-sm">Top-500 人工审查 + 哈工大通用词表 + 百度词表 → 965 个停用词。高频虚词对检索无区分价值，已全部过滤。</p></div></div>
    </div>
  </div>
</section>
"""

# ===== P4: Personas (S04 Brief Grid as substitute) =====
slides += """
<section class="slide dark" data-animate="grid-reveal">
  <div class="canvas-card" style="background:var(--ink);color:var(--paper)">
    <div class="chrome-min"><div class="l" style="color:var(--paper)">战略层 · 用户需求</div><div class="r" style="color:rgba(255,255,255,.4)">04 / 10</div></div>
    <h1 class="h-xl" style="color:var(--paper);margin-bottom:4vh">战略层：用户需求分析</h1>
    <div class="grid-12" style="gap:var(--sp-5)">
      <div class="card-outlined span-4" style="border-color:rgba(255,255,255,.2)">
        <span class="t-cat" style="color:var(--accent)">Persona 1</span>
        <h3 class="h-md" style="color:var(--paper);margin:0.8vh 0">应届生小李</h3>
        <p class="body-sm" style="color:rgba(255,255,255,.7)">22 岁 · 计算机本科<br>寻找 IT/互联网校招岗位<br>核心：行业筛选 + 关键词搜索 + 薪资排序</p>
      </div>
      <div class="card-outlined span-4" style="border-color:rgba(255,255,255,.2)">
        <span class="t-cat" style="color:var(--accent)">Persona 2</span>
        <h3 class="h-md" style="color:var(--paper);margin:0.8vh 0">求职者小赵</h3>
        <p class="body-sm" style="color:rgba(255,255,255,.7)">28 岁 · 3年经验 · 在职看机会<br>关注薪资+稳定性+成长空间<br>核心：多维度筛选 + 相似岗位推荐</p>
      </div>
      <div class="card-outlined span-4" style="border-color:rgba(255,255,255,.2)">
        <span class="t-cat" style="color:var(--accent)">Persona 3</span>
        <h3 class="h-md" style="color:var(--paper);margin:0.8vh 0">分析师张工</h3>
        <p class="body-sm" style="color:rgba(255,255,255,.7)">35 岁 · 行业研究分析师<br>撰写季度招聘趋势报告<br>核心：行业对比 + 薪资分布 + 趋势分析</p>
      </div>
    </div>
    <div style="margin-top:var(--sp-7);padding-top:var(--sp-5);border-top:1px solid rgba(255,255,255,.12)">
      <span class="t-cat" style="color:var(--accent)">用户旅程</span>
      <p class="lead" style="color:var(--paper);max-width:60ch;margin-top:1vh">首页搜索 → 行业筛选 → 浏览结果列表 → 点击详情 (关键词标签 + 分类预测 + 相似推荐) → 返回继续浏览</p>
    </div>
  </div>
</section>
"""

# ===== P5: Product Goals (S03 Split Statement) =====
slides += """
<section class="slide split light" data-animate="split-statement">
  <div class="canvas-card" style="padding:0">
    <div class="chrome-min" style="margin:5.6vh 3.6vw 0"><div class="l">战略层 · 产品目标</div><div class="r">05 / 10</div></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;flex:1">
      <div style="display:flex;align-items:center;justify-content:center;background:var(--accent);padding:var(--sp-8)">
        <h1 class="h-statement" style="color:var(--accent-on);font-size:min(9vw,15vh)">目标</h1>
      </div>
      <div style="padding:var(--sp-8);display:flex;flex-direction:column;justify-content:center">
        <h3 class="h-md" style="margin-bottom:var(--sp-5)">三个维度的知识融合</h3>
        <ul style="list-style:none;padding:0">
          <li class="body-sm" style="margin-bottom:var(--sp-4)"><span class="t-cat" style="color:var(--accent)">理论课</span><br>布尔检索 · VSM · 倒排索引 · TF-IDF · P/R/F1/MAP</li>
          <li class="body-sm" style="margin-bottom:var(--sp-4)"><span class="t-cat" style="color:var(--accent)">实践课</span><br>N-gram · 分词评测 (7种) · 关键词抽取 · 文本分类 · 词汇聚类</li>
          <li class="body-sm"><span class="t-cat" style="color:var(--accent)">产品设计</span><br>五层模型: 战略 → 范围 → 结构 → 框架 → 表现</li>
        </ul>
      </div>
    </div>
  </div>
</section>
"""

# ===== P6: Functions Overview (S06 KPI Tower) =====
slides += """
<section class="slide light" data-animate="tower-rise">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">范围层 · 功能规格</div><div class="r">06 / 10</div></div>
    <h1 class="h-xl" style="margin-bottom:1.6vh">范围层：核心功能</h1>
    <p class="lead" style="max-width:56ch;margin-bottom:4vh">10 大功能模块 · 覆盖信息组织、检索、展示全链路</p>
    <div class="kpi-tower-row">
      <div class="bar-tower" style="height:62vh"><span class="t-cat">F1</span><span class="h-md">多字段检索</span><span class="body-sm">VSM + TF-IDF<br>标题 2.0x 权重</span></div>
      <div class="bar-tower" style="height:54vh"><span class="t-cat">F2</span><span class="h-md">结构化筛选</span><span class="body-sm">布尔过滤<br>6 维度</span></div>
      <div class="bar-tower" style="height:46vh"><span class="t-cat">F3</span><span class="h-md">关键词高亮</span><span class="body-sm">查询词<br>黄色标记</span></div>
      <div class="bar-tower" style="height:38vh"><span class="t-cat">F4</span><span class="h-md">多维度排序</span><span class="body-sm">相关度 · 日期<br>月薪 升/降</span></div>
      <div class="bar-tower accent" style="height:66vh"><span class="t-cat" style="color:var(--accent-on)">F5</span><span class="h-md" style="color:var(--accent-on)">行业标签</span><span class="body-sm" style="color:rgba(255,255,255,.8)">9 大类<br>NB 分类器</span></div>
      <div class="bar-tower accent" style="height:58vh"><span class="t-cat" style="color:var(--accent-on)">F6</span><span class="h-md" style="color:var(--accent-on)">技能族群</span><span class="body-sm" style="color:rgba(255,255,255,.8)">10 个族群<br>K-Means</span></div>
    </div>
  </div>
</section>
"""

# ===== P7: Function Details (S16 Brief Grid) =====
tfidf_f1 = kw_eval.get("TF-IDF", {}).get("F1", "N/A")
tr_f1 = kw_eval.get("TextRank", {}).get("F1", "N/A")

slides += f"""
<section class="slide light" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">范围层 · 功能详解</div><div class="r">07 / 10</div></div>
    <h1 class="h-xl" style="margin-bottom:4vh">功能详解 & 实验融合</h1>
    <div class="brief-grid">
      <div class="brief-card"><span class="t-cat">F7 · 相似推荐</span><h3 class="h-md" style="margin:1vh 0">TF-IDF 余弦相似度</h3><p class="body-sm">5,000 条 × Top-5<br>相似度 0.27~0.34<br>底层即 VSM 检索</p></div>
      <div class="brief-card card-accent"><span class="t-cat" style="color:var(--accent-on)">F8 · 关键词抽取</span><h3 class="h-md" style="color:var(--accent-on);margin:1vh 0">TF-IDF F1={tfidf_f1}%</h3><p class="body-sm" style="color:rgba(255,255,255,.8)">TextRank F1={tr_f1}%<br>100 条金标准评测<br>Top-15 标签云</p></div>
      <div class="brief-card"><span class="t-cat">F9 · 数据统计</span><h3 class="h-md" style="margin:1vh 0">多维度可视化</h3><p class="body-sm">行业排行 · 城市热度<br>薪资直方图 · 学历饼图<br>关键词词云</p></div>
      <div class="brief-card card-ink"><span class="t-cat" style="color:var(--paper)">F10 · 分词展示</span><h3 class="h-md" style="color:var(--paper);margin:1vh 0">7 种分词器对比</h3><p class="body-sm" style="color:rgba(255,255,255,.7)">FMM/BMM/BiMM/MinSeg<br>HMM/Ngram/CRF<br>详情页弹窗</p></div>
      <div class="brief-card card-accent"><span class="t-cat" style="color:var(--accent-on)">实验五 · 分类</span><h3 class="h-md" style="color:var(--accent-on);margin:1vh 0">NB vs KNN 9类</h3><p class="body-sm" style="color:rgba(255,255,255,.8)">NB Micro F1=25%<br>KNN Micro F1=16%<br>400条标注</p></div>
      <div class="brief-card card-ink"><span class="t-cat" style="color:var(--paper)">实验六 · 聚类</span><h3 class="h-md" style="color:var(--paper);margin:1vh 0">10个技能族群</h3><p class="body-sm" style="color:rgba(255,255,255,.7)">800词 K-Means<br>簇间分离度 0.78<br>词云矩阵图</p></div>
    </div>
  </div>
</section>
"""

# ===== P8: Clusters (S15 Matrix) =====
slides += """
<section class="slide dark" data-animate="matrix-fill">
  <div class="canvas-card" style="background:var(--ink);color:var(--paper)">
    <div class="chrome-min"><div class="l" style="color:var(--paper)">实验六 · 技能族群</div><div class="r" style="color:rgba(255,255,255,.4)">08 / 10</div></div>
    <h1 class="h-xl" style="color:var(--paper);margin-bottom:4vh">10 个技能族群 · 无监督聚类</h1>
    <div class="matrix-fill" style="grid-template-columns:repeat(5,1fr);gap:1.6vh;margin-bottom:var(--sp-8)">
      <div class="matrix-cell card-accent"><span class="t-cat" style="color:var(--accent-on)">族群 1</span><span class="body-sm" style="color:rgba(255,255,255,.8);margin-top:0.6vh">设备制造自动化<br><span class="mono">85 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 2</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">采购管理项目<br><span class="mono">132 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 3</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">HR 招聘应聘<br><span class="mono">26 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 4</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">生产质量检测<br><span class="mono">330 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 5</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">研发投资研究<br><span class="mono">49 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 6</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">财务税务审计<br><span class="mono">22 词</span></span></div>
      <div class="matrix-cell card-ink" style="background:var(--accent)"><span class="t-cat" style="color:var(--accent-on)">族群 7</span><span class="body-sm" style="color:rgba(255,255,255,.8);margin-top:0.6vh">电商运营营销<br><span class="mono">49 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 8</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">食品安全质量<br><span class="mono">12 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 9</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">数据 AI 系统<br><span class="mono">52 词</span></span></div>
      <div class="matrix-cell card-outlined" style="border-color:rgba(255,255,255,.2)"><span class="t-cat" style="color:var(--accent)">族群 10</span><span class="body-sm" style="color:rgba(255,255,255,.5);margin-top:0.6vh">客户销售市场<br><span class="mono">43 词</span></span></div>
    </div>
    <div style="display:flex;gap:var(--sp-8)">
      <div><span class="mono" style="font-size:6vw;font-weight:200;color:var(--accent)">0.78</span><span class="t-cat" style="display:block;color:rgba(255,255,255,.5)">簇间分离度</span></div>
      <div><span class="mono" style="font-size:6vw;font-weight:200;color:var(--accent)">0.15</span><span class="t-cat" style="display:block;color:rgba(255,255,255,.5)">簇内紧密度</span></div>
    </div>
  </div>
</section>
"""

# ===== P9: Structure Layer (S17 System Diagram) =====
slides += """
<section class="slide light" data-animate="pipeline">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">结构层 · 信息架构</div><div class="r">09 / 10</div></div>
    <h1 class="h-xl" style="margin-bottom:4vh">结构层：页面架构 & 交互设计</h1>
    <div class="system-diagram" style="grid-template-columns:repeat(3,1fr);gap:var(--sp-6)">
      <div class="card-fill"><span class="t-cat" style="color:var(--accent)">首页</span><h3 class="h-md" style="margin:1.2vh 0">搜索门户</h3><p class="body-sm">搜索框 + 快捷筛选<br>9 行业标签 + 最新招聘</p></div>
      <div class="card-accent"><span class="t-cat" style="color:var(--accent-on)">检索结果</span><h3 class="h-md" style="color:var(--accent-on);margin:1.2vh 0">列表 + 筛选</h3><p class="body-sm" style="color:rgba(255,255,255,.8)">左筛选面板 (布尔过滤)<br>右结果列表 (VSM 排序)<br>分页 + 排序切换</p></div>
      <div class="card-fill"><span class="t-cat" style="color:var(--accent)">职位详情</span><h3 class="h-md" style="margin:1.2vh 0">完整信息</h3><p class="body-sm">基本信息 + 描述高亮<br>关键词标签 + 分类预测<br>分词效果 + 相似推荐</p></div>
      <div class="card-fill"><span class="t-cat" style="color:var(--accent)">行业浏览</span><h3 class="h-md" style="margin:1.2vh 0">9 大行业列表</h3><p class="body-sm">点击行业 → 该行业<br>下的职位列表页</p></div>
      <div class="card-fill"><span class="t-cat" style="color:var(--accent)">数据统计</span><h3 class="h-md" style="margin:1.2vh 0">可视化仪表盘</h3><p class="body-sm">行业排行 · 城市热度<br>薪资直方图 · 学历饼图</p></div>
      <div class="card-ink"><span class="t-cat" style="color:var(--paper)">聚类视图</span><h3 class="h-md" style="color:var(--paper);margin:1.2vh 0">技能族群词云</h3><p class="body-sm" style="color:rgba(255,255,255,.7)">词云矩阵 + PCA 散点图<br>各族群关键词列表</p></div>
    </div>
  </div>
</section>
"""

# ===== P10: Framework + Surface (S19 Four Cards) =====
slides += """
<section class="slide light" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min"><div class="l">框架层 & 表现层</div><div class="r">10 / 10</div></div>
    <h1 class="h-xl" style="margin-bottom:3.2vh">框架层 & 表现层</h1>
    <p class="lead" style="max-width:56ch;margin-bottom:4vh">线框设计 · 信息设计 · 导航设计 · 视觉风格</p>
    <div class="four-cards">
      <div class="card-outlined"><span class="t-cat" style="color:var(--accent)">框架层</span><h3 class="h-md" style="margin:1.2vh 0">线框设计</h3><p class="body-sm">首页搜索居中<br>结果页左筛选右列表<br>详情页上信息下描述<br>统计页网格布局</p></div>
      <div class="card-outlined"><span class="t-cat" style="color:var(--accent)">框架层</span><h3 class="h-md" style="margin:1.2vh 0">信息设计</h3><p class="body-sm">职位卡片统一模板<br>搜索结果关键词高亮<br>分页器 · 排序选择器</p></div>
      <div class="card-outlined"><span class="t-cat" style="color:var(--accent)">表现层</span><h3 class="h-md" style="margin:1.2vh 0">视觉风格</h3><p class="body-sm">主色 #1a73e8 蓝<br>辅色 #34a853 绿<br>字体: 微软雅黑<br>14px 正文 · 24px 标题</p></div>
      <div class="card-accent"><span class="t-cat" style="color:var(--accent-on)">表现层</span><h3 class="h-md" style="color:var(--accent-on);margin:1.2vh 0">一致性</h3><p class="body-sm" style="color:rgba(255,255,255,.8)">全局统一配色/字体/间距<br>Flask + Jinja2 模板继承<br>base.html → 所有页面</p></div>
    </div>
  </div>
</section>
"""

# Replace
cut_start = html.find("<!-- SLIDES_HERE")
cut_end = html.find("<script>", cut_start)
html = html[:cut_start] + slides + "\n" + html[cut_end:]

with open(PPT_OUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Generated: {PPT_OUT} ({len(slides):,} chars)")
print(f"Slides: {slides.count('<section class=')}")
