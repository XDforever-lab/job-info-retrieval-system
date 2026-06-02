"""Build PPT HTML - plain strings, no f-string issues"""
import os, base64, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *

OUT = "C:/Users/ZhangQichen/Desktop/recruit/code/ppt/index.html"

def b64(path):
    if not os.path.exists(path): return ""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

zipf = b64(os.path.join(DIR_9_NGRAM, "zipf_loglog.png"))
# seg  = b64(os.path.join(DIR_7_SEG_EVAL, "seg_eval_chart.png"))
kw   = b64(os.path.join(DIR_10_KW, "eval_chart.png"))
clu  = b64(os.path.join(DIR_15_CLUSTER, "word_cluster_pca.png"))

def IMG(b):
    if not b: return ""
    return '<img src="data:image/png;base64,'+b+'" style="max-width:100%;max-height:38vh;object-fit:contain;border-radius:6px">'

A='#d2991d'; P='#e6edf3'; PR='230,237,243'

# Build complete HTML as a list of strings, joined at the end
out = []
out.append('''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>上市公司招聘信息检索系统 · 开题汇报</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden;background:#0d1117;color:'''+P+''';font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif}
#deck{position:fixed;inset:0;width:10000vw;height:100vh;display:flex;transition:transform .5s cubic-bezier(.77,0,.175,1);z-index:10}
.slide{width:100vw;height:100vh;flex-shrink:0;overflow-y:auto;overflow-x:hidden;padding:6vh 6vw;display:flex;flex-direction:column}
.slide.dark{background:#0d1117;color:'''+P+'''}
.accent{color:'''+A+'''}
.mono{font-family:Consolas,"Courier New",monospace;font-size:13px}
h2{font-size:min(3.4vw,5.2vh);font-weight:300;letter-spacing:-.02em;margin-bottom:3.6vh}
h3{font-size:min(1.6vw,22px);font-weight:400;margin-bottom:1.2vh}
p,li{font-size:min(1.15vw,16px);line-height:1.7;color:rgba('''+PR+''',.7)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:3vw;flex:1}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:2vw;flex:1}
.grid4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1.6vw;flex:1}
.card{padding:2.4vh 1.8vw;border-radius:8px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06)}
.card2{padding:2vh 1.6vw;border-radius:6px;background:rgba(210,153,29,.08);border-left:3px solid '''+A+'''}
.table{display:grid;gap:4px 12px;font-family:Consolas,monospace;font-size:13px}
.table .th{color:'''+A+''';font-weight:600}
.table div{color:rgba('''+PR+''',.6)}
.kpi{text-align:center;padding:2vh 1vw}
.kpi .num{font-size:min(4.8vw,7vh);font-weight:100;color:'''+A+''';line-height:1.1}
.kpi .label{font-size:13px;color:rgba('''+PR+''',.45);margin-top:6px}
.n{font-weight:600;color:'''+A+'''}
#nav{position:fixed;bottom:3vh;left:50%;transform:translateX(-50%);z-index:30;display:flex;gap:8px}
#nav .dot{width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.25);border:0;cursor:pointer}
#nav .dot.active{background:'''+A+'''}
#ov{display:none;position:fixed;inset:0;z-index:100;background:rgba(13,17,23,.96);padding:6vh 4vw;overflow:auto}
#ov .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:2vh 1.6vw;max-width:90vw;margin:0 auto}
#ov .item{cursor:pointer;border:2px solid rgba(255,255,255,.1);border-radius:6px;padding:2vh 1.4vw;color:rgba(255,255,255,.5);font-size:14px}
#ov .item.active{border-color:'''+A+''';color:'''+P+'''}
</style></head><body><div id="deck">''')

# ========= SLIDES =========
def add(html):
    out.append(html)

# P1
add('''<section class="slide dark" style="justify-content:center;align-items:center;text-align:center">
<div style="font-size:14px;letter-spacing:.28em;text-transform:uppercase;color:'''+A+''';margin-bottom:3vh">信息检索系统实验 · 期末大作业</div>
<h1 style="font-size:min(8vw,13vh);font-weight:100;line-height:1.1;letter-spacing:-.03em;margin-bottom:4vh">上市公司招聘<br>信息检索系统</h1>
<p style="font-size:min(2vw,20px);color:rgba('''+PR+''',.45);max-width:48ch">中国 A 股上市公司 2026 年招聘发布数据<br>面向求职者与行业研究者的垂直领域检索系统</p>
<div style="margin-top:8vh;font-size:12px;color:rgba('''+PR+''',.2);letter-spacing:.2em">南京农业大学 · 信息检索系统实验课程 · 2026 年 6 月</div>
</section>''')

# P2
add('''<section class="slide dark">
<h2>02 · 语料特征与规模</h2>
<div class="grid4" style="margin-bottom:4vh">
<div class="kpi"><div class="num">5,000</div><div class="label">清洗后记录数</div></div>
<div class="kpi"><div class="num">14</div><div class="label">保留字段</div></div>
<div class="kpi"><div class="num">9</div><div class="label">行业大类</div></div>
<div class="kpi"><div class="num">965</div><div class="label">停用词数量</div></div>
</div>
<div class="grid2">
<div><h3 class="accent">数据来源与预处理</h3>
<p>原始数据：中国 A 股上市公司 2026 年招聘发布数据 (macrodatas.cn)，原始规模 93,602 条 × 81 个行业 × 8,796 家企业。经 GBK→UTF-8 编码转换、文本清洗（去除 URL/公众号广告/空白字符）、字段筛选（23→14 字段）、缺失值处理（关键字段缺失删行，次要字段填默认值）、学历/经验/城市名称标准化、招聘结束日期自动生成（发布日期+60天）后保留 <span class="n">5,000 条</span>。</p>
<p style="margin-top:2vh">核心字段：企业名称 · 股票简称 · 上市公司行业 · 招聘岗位 · 工作城市 · 最低/最高月薪 · 职位描述 · 学历要求 · 要求经验 · 招聘类别 · 发布日期 · 结束日期</p></div>
<div><h3 class="accent">9 大行业分类 & 分词方案</h3>
<p>从原 81 类行业归并为 <span class="n">9 个对求职者有实际意义的行业大类</span>：互联网/软硬件技术 · 销售/市场/客服 · 人力/财务/行政 · 生产制造/汽车 · 金融 · 房地产/建筑 · 物流/贸易/采购 · 医疗/能源/环保/农业 · 文化传媒/教育/生活服务。</p>
<p style="margin-top:2vh">分词方案：采用 <span class="n">CRF 条件随机场</span> 对职位描述和招聘岗位进行分词。在金标准 200 条人工精标语料上，CRF 评测 F1=74.77%，优于 FMM/BMM/HMM 等 6 种对比方法。系统全量数据依此生成分词字段。</p></div>
</div></section>''')

# P3
add('<section class="slide dark"><h2>03 · N-Gram 统计与齐普夫定律验证</h2><div class="grid2"><div>'
'<h3 class="accent">实验一：Unigram 词频统计</h3>'
'<p>对 5,000 条职位描述 CRF 分词结果统计：总词次 <span class="n">1,049,179</span>，唯一词型 <span class="n">32,365</span>。Top-500 高频词中以标点符号（11.4%）、结构虚词（的/及/与/和）、招聘文本套话（工作/负责/岗位/相关）为主——这些词出现于几乎所有文档，对检索无区分价值。</p>'
'<h3 class="accent" style="margin-top:3vh">齐普夫定律验证</h3>'
'<p>绘制词频 vs 排名双对数曲线并做线性回归。拟合优度 <span class="n">R² > 0.85</span>，斜率接近 -1.0。招聘文本词频分布基本符合 Zipf 定律——词频与排名近似成反比。曲线两端存在自然偏离：高频端受虚词影响，低频端形成长尾（仅出现 1 次的词占大量词型）。</p>'
'<h3 class="accent" style="margin-top:3vh">停用词表构建</h3>'
'<p>Top-500 词表人工逐词审查（标记"高频但无检索价值"的词） + 哈工大通用停用词表 + 百度停用词表 → 合并去重得 <span class="n">965 个停用词</span>。检索系统查询时过滤。</p></div>'
'<div style="display:flex;align-items:center;justify-content:center">' + IMG(zipf) + '</div>'
'</div></section>')

# P4 - Personas
add('''<section class="slide dark"><h2>04 · 战略层：用户需求分析</h2><div class="grid3">
<div class="card"><div style="font-size:12px;color:'''+A+''';letter-spacing:.14em;margin-bottom:1.6vh">PERSONA 1</div><h3>应届生小李</h3><p>22 岁 · 计算机科学与技术本科应届</p><p style="margin-top:1.2vh"><span class="n">目标：</span>寻找 IT/互联网行业的校园招聘岗位</p><p><span class="n">核心需求：</span>① 按行业筛选 IT 相关岗位 ② 按城市筛选工作地点 ③ 关键词搜索（Python、机器学习等技能词）④ 按薪资范围排序</p><p style="margin-top:1.2vh"><span class="n">痛点：</span>招聘网站信息分散，想专门了解上市公司（尤其是科技类）的招聘动态</p></div>
<div class="card"><div style="font-size:12px;color:'''+A+''';letter-spacing:.14em;margin-bottom:1.6vh">PERSONA 2</div><h3>求职者小赵</h3><p>28 岁 · 3 年工作经验 · 在职寻找更好的机会</p><p style="margin-top:1.2vh"><span class="n">目标：</span>找到薪资更高、稳定性更强、有成长空间的岗位</p><p><span class="n">核心需求：</span>① 薪资区间精确筛选 ② 学历+经验多维度过滤 ③ 相似岗位推荐（发现"能迁移技能的方向"）④ 偏好上市公司（福利规范、稳定性强）</p><p style="margin-top:1.2vh"><span class="n">痛点：</span>传统网站只能搜岗位名，难以发现跨行业的可迁移机会</p></div>
<div class="card"><div style="font-size:12px;color:'''+A+''';letter-spacing:.14em;margin-bottom:1.6vh">PERSONA 3</div><h3>分析师张工</h3><p>35 岁 · 行业研究分析师</p><p style="margin-top:1.2vh"><span class="n">目标：</span>通过招聘数据洞察行业发展趋势</p><p><span class="n">核心需求：</span>① 按行业浏览各上市公司的招聘规模与岗位结构 ② 对比不同行业的薪资水平 ③ 分析不同行业对学历/经验的要求差异 ④ 关键词趋势分析热门技能需求</p><p style="margin-top:1.2vh"><span class="n">痛点：</span>现有招聘平台侧重岗位匹配，缺乏宏观的行业招聘数据分析视角</p></div>
</div></section>''')

# P5 - Goals
add('''<section class="slide dark"><h2>05 · 战略层：产品目标</h2><div class="grid3">
<div class="card2"><div class="mono" style="font-size:48px;color:'''+A+'''">01</div><h3 style="margin-top:1.6vh">学术目标</h3><p>完成信息检索课程综合实验要求。完整展现 <span class="n">信息组织（倒排索引+TF-IDF加权）、信息检索（VSM余弦相似度+布尔过滤）、信息展示（高亮+排序+可视化）</span> 三大核心模块，以及六次实验任务的知识融合。</p></div>
<div class="card2"><div class="mono" style="font-size:48px;color:'''+A+'''">02</div><h3 style="margin-top:1.6vh">能力目标</h3><p>掌握从 <span class="n">数据预处理 → NLP实验（分词/关键词/分类/聚类）→ 检索引擎构建 → Web前端开发 → 系统评测</span> 的完整信息检索系统构建流程。区分个人实现与 AI 辅助部分。</p></div>
<div class="card2"><div class="mono" style="font-size:48px;color:'''+A+'''">03</div><h3 style="margin-top:1.6vh">展示目标</h3><p>产出一个 <span class="n">可演示、可交互的 Web 检索系统</span>，直观呈现上市公司招聘数据全貌。遵循 <span class="n">以用户为中心的五层产品设计模型</span>：战略层 → 范围层 → 结构层 → 框架层 → 表现层。</p></div>
</div></section>''')

# P6 - Scope 1/3: F1-F5 detailed
add('''<section class="slide dark"><h2>06 · 范围层 (1/3)：核心检索功能</h2><div class="grid2"><div>
<div class="card2" style="margin-bottom:1.6vh"><h3>F1 · 多字段联合检索 — VSM + TF-IDF</h3>
<p><span class="n">做什么：</span>用户输入关键词，系统同时对"招聘岗位""职位描述""企业名称"三字段执行全文检索，返回按相关度排序的结果列表。</p>
<p><span class="n">怎么做：</span>Step 8: CRF 对全量 5k 职位描述和岗位名称分词 → 倒排索引。Step 16: 构建 TF-IDF 文档-词条稀疏矩阵 (5000×32,365)，L2归一化。查询时：用户输入分词后，查 vocab_map.json 定位向量维度 → 与矩阵批量计算余弦相似度 → 降序排列。招聘岗位（标题）字段赋予 2.0×TF-IDF 权重加成。关键词在结果中用黄色高亮标记。</p>
<p><span class="n">关联课程：</span>理论课第2章(向量空间模型) + 第3章(倒排文档检索) + 第6章(TF-IDF标引加权)</p></div>
<div class="card2" style="margin-bottom:1.6vh"><h3>F2 · 结构化筛选 — 布尔逻辑检索</h3>
<p><span class="n">做什么：</span>用户选择行业、城市、学历、经验、薪资范围、日期范围等条件，系统对全文检索结果做进一步过滤。</p>
<p><span class="n">怎么做：</span>Step 3 清洗后数据中各字段已标准化。筛选条件转换为布尔表达式（如"行业=金融 AND 城市=上海 AND 最低月薪>=15000"），在倒排索引候选集上执行布尔过滤。</p>
<p><span class="n">关联课程：</span>理论课第2章(布尔逻辑检索) + 第3章(顺排文档检索)</p></div>
</div><div>
<div class="card2" style="margin-bottom:1.6vh"><h3>F3 · 关键词高亮 + F4 · 多维排序</h3>
<p><span class="n">F3 高亮：</span>结果列表和详情页中，对用户输入关键词用黄色背景+加粗标记。前端 JS 正则匹配替换为 &lt;mark&gt; 标签。</p>
<p><span class="n">F4 排序：</span>支持四种排序方式切换：相关度(TF-IDF得分)、发布日期(降序)、最低月薪(降/升)、最高月薪(降/升)。检索结果返回后前端本地排序。</p>
<p><span class="n">关联课程：</span>信息的展示</p></div>
<div class="card2"><h3>F5 · 9 行业标签浏览 — 文本分类</h3>
<p><span class="n">做什么：</span>首页展示 9 个行业分类标签卡片，用户点击标签进入该行业招聘列表。</p>
<p><span class="n">怎么做：</span>Step 12: 从 5k 数据均匀抽取 400 条，人工标注 9 类标签。Step 13: 8:2 分层抽样(320训练/80测试) → 手写余弦KNN(K=3~40网格搜索) vs 手写多项式 NB(α=0.01~5.0) → <span class="n">NB 微观 F1=25% 较优</span>(KNN 16%) → NB 对全量 5k 分类 → 追加"预测行业"列。Step 14: 保存模型为 joblib。</p>
<p><span class="n">关联课程：</span>实验五(文本分类) + 第7章(KNN、朴素贝叶斯)</p></div>
</div></div></section>''')

# P7 - Scope 2/3: Keywords + Clustering
add('<section class="slide dark"><h2>07 · 范围层 (2/3)：智能服务 — 关键词与聚类</h2><div class="grid2"><div>'
'<div class="card2" style="margin-bottom:2vh"><h3>F6 · 关键词抽取 — TF-IDF + TextRank</h3>'
'<p><span class="n">做什么：</span>为每条职位描述自动提取 Top-15 关键词，在详情页以标签形式展示，用户点击标签触发二次搜索。统计页按行业/城市/类别聚合展示关键词词云。</p>'
'<p><span class="n">怎么做：</span>Step 10: 加载分词数据 → 构建文档-语词矩阵 → 全局 IDF → 逐文档 TF-IDF (标题 2.0x 权重) + TextRank (窗口=5,阻尼=0.85,PageRank迭代)。Step 11: 100条人工金标准 → 宏平均评测：<span class="n">TF-IDF F1=44.61%</span> (P=32.97% R=76.11%)，TextRank F1=41.92%。TF-IDF 为最终方案，Pooling融合后生成全量关键词JSON + 行业/城市/类别聚合词云数据。</p>'
'<p><span class="n">关联课程：</span>实验四(关键词抽取) + 第6章(自动标引)</p></div>'
+ IMG(kw) +
'</div><div>'
'<div class="card2" style="margin-bottom:2vh"><h3>F7 · 技能族群聚类 — K-Means 词汇聚类</h3>'
'<p><span class="n">做什么：</span>发现"哪些技能天然一起出现在招聘要求中"。800 个高频技能词被聚为 10 个技能族群——形成如"财务审计族群""数据AI族群""设备制造族群"等，帮助求职者理解技能间的共生关系。</p>'
'<p><span class="n">怎么做：</span>Step 15: 从 5k 文档中筛选 DF≥10 的词 × TF-IDF方差最高的 800 词 → 构建词-文档 TF-IDF 矩阵 → L2归一化 → K-Means (K=6~12网格搜索) + 等级聚类 (Ward/Single/Complete) → 肘部法则+轮廓系数定 K → <span class="n">10 个技能族群</span>，簇间分离度 0.78，簇内紧密度 0.15 → PCA 2D散点图 + 词云矩阵图。</p>'
'<p><span class="n">关联课程：</span>实验六(文本聚类) + 第7章(K-Means、等级聚类)</p></div>'
+ IMG(clu) +
'</div></div></section>')


# New P8 - F8/F9/F10
add("""<section class="slide dark"><h2>08 · 范围层 (3/4)：F8-F10 智能展示功能</h2><div class="grid3">
<div class="card2"><div class="mono" style="font-size:24px;color:#d2991d">F8 · 相似岗位推荐</div><h3 style="margin-top:1vh">TF-IDF 余弦相似度</h3>
<p style="font-size:14px"><span class="n">做什么：</span>详情页底部展示与该岗位最相似的 Top-5 其他岗位。本质即 VSM 检索——对当前职位 TF-IDF 向量与全量 5,000 文档矩阵做批量余弦计算，返回相似度最高的 5 个。相似度范围 0.27~0.34（短文本正常）。</p>
<p style="font-size:14px;margin-top:1vh"><span class="n">怎么做：</span>Step 16: 构建全量 5,000×32,365 TF-IDF 稀疏矩阵 → models/tfidf_matrix.npz。batch_cosine_sim(job_id, top_k=5) 函数逐文档计算 cos 并排除自身 → Top-5。结果保存为 similar_jobs.json，Web 启动时加载。</p>
<p style="font-size:14px;margin-top:1vh"><span class="n">关联课程：</span>第2章(向量空间模型+余弦相似度) + 第3章(倒排文档检索)</p></div>
<div class="card2"><div class="mono" style="font-size:24px;color:#d2991d">F9 · 数据统计仪表盘</div><h3 style="margin-top:1vh">多维度数据可视化</h3>
<p style="font-size:14px"><span class="n">做什么：</span>提供宏观数据视角：行业招聘数量排行(Top-15)、城市招聘热度(Top-10)、薪资分布直方图(5k区间)、学历要求饼图、经验要求分布图。服务分析师张工(Persona 3)的行业趋势分析需求。</p>
<p style="font-size:14px;margin-top:1vh"><span class="n">怎么做：</span>Step 1: 数据探索已生成 stats_summary.json。Step 11: 关键词按行业/城市/类别聚合为 agg_*.json。前端用 Matplotlib 预渲染 + 前端 ECharts 交互渲染。数据源自 cleaned.csv 的结构化字段。</p>
<p style="font-size:14px;margin-top:1vh"><span class="n">关联课程：</span>信息的展示 · 实验一(宏观统计)</p></div>
<div class="card2"><div class="mono" style="font-size:24px;color:#d2991d">F10 · 分词效果对比</div><h3 style="margin-top:1vh">7种分词器评测弹窗</h3>
<p style="font-size:14px"><span class="n">做什么：</span>职位详情页提供"查看分词效果"按钮，点击弹出模态框，展示同一职位描述的 7 种分词方案（FMM/BMM/BiMM/MinSeg/HMM/Ngram/CRF）切分结果对比，差异位置浅红标注。底部附 P/R/F1 评测表。</p>
<p style="font-size:14px;margin-top:1vh"><span class="n">怎么做：</span>Step 7: 200条人工金标准 + BIES字符级 + 集合元素化双评测。7种分词器结果保存至 seg_result_*.txt，Web 加载后按 / 拆分为词列表，diff 算法标注差异位置渲染。</p>
<p style="font-size:14px;margin-top:1vh"><span class="n">关联课程：</span>实验二(分词规范+人工精标) + 实验三(自动分词算法实现与迭代优化)</p></div>
</div></section>""")
# P8 - Results
add('''<section class="slide dark"><h2>08 · 范围层 (4/4)：核心实验数据汇总</h2><div class="grid2"><div>
<h3 class="accent" style="margin-bottom:1.6vh">分词评测 (200条金标准, 7种模型)</h3>
<div class="table" style="grid-template-columns:1fr 1fr 1fr 1fr">
<div class="th">模型</div><div class="th">P(%)</div><div class="th">R(%)</div><div class="th">F1(%)</div>
<div>FMM</div><div>70.98</div><div>62.49</div><div>66.46</div>
<div>BMM</div><div>71.23</div><div>62.85</div><div>66.78</div>
<div>BiMM</div><div>71.87</div><div>63.20</div><div>67.26</div>
<div>MinSeg</div><div>71.67</div><div>63.03</div><div>67.08</div>
<div>HMM</div><div>70.09</div><div>64.28</div><div>67.06</div>
<div>Ngram</div><div>78.43</div><div>69.04</div><div style="color:'''+A+'''">73.43</div>
<div>CRF</div><div>78.76</div><div>71.16</div><div style="color:'''+A+''';font-weight:600">74.77</div>
</div>
<p style="margin-top:1.2vh;font-size:13px;color:rgba('''+PR+''',.4)">BIES 字符级 + 集合元素化双评测 · 机械分词 F1≈66% · 统计分词 F1≈74% · CRF 选为系统主分词器</p>
<h3 class="accent" style="margin-top:3vh;margin-bottom:1.6vh">关键词评测 (100条金标准)</h3>
<div class="table" style="grid-template-columns:1fr 1fr 1fr 1fr">
<div class="th">算法</div><div class="th">P(%)</div><div class="th">R(%)</div><div class="th">F1(%)</div>
<div>TF-IDF</div><div>32.97</div><div>76.11</div><div style="color:'''+A+''';font-weight:600">44.61</div>
<div>TextRank</div><div>30.87</div><div>72.82</div><div>41.92</div>
</div>
</div><div>
<h3 class="accent" style="margin-bottom:1.6vh">文本分类评测 (400条标注, 9类)</h3>
<div class="table" style="grid-template-columns:1fr 1fr 1fr 1fr;margin-bottom:1.2vh">
<div class="th">模型</div><div class="th">宏 P</div><div class="th">宏 R</div><div class="th">宏 F1</div>
<div>KNN(K=3)</div><div>16.92%</div><div>14.85%</div><div>15.28%</div>
<div>NB(α=0.1)</div><div>20.05%</div><div>18.77%</div><div style="color:'''+A+'''">18.36%</div>
</div>
<div class="table" style="grid-template-columns:1fr 1fr 1fr 1fr">
<div class="th">模型</div><div class="th">微 P</div><div class="th">微 R</div><div class="th">微 F1</div>
<div>KNN(K=3)</div><div>16.25%</div><div>16.25%</div><div>16.25%</div>
<div>NB(α=0.1)</div><div>25.00%</div><div>25.00%</div><div style="color:'''+A+''';font-weight:600">25.00%</div>
</div>
<p style="font-size:13px;color:rgba('''+PR+''',.4);margin-top:1vh">9类×320训练≈36条/类，随机基线≈11%。NB 微平均显著领先，选 NB 对全量分类。</p>
<h3 class="accent" style="margin-top:3vh;margin-bottom:1.2vh">10 个技能族群 (K-Means K=10)</h3>
<p style="font-size:13px;color:rgba('''+PR+''',.55);line-height:1.8">设备制造自动化(85词) · 采购管理项目(132词) · HR招聘(26词) · 生产质量(330词) · 研发投资(49词) · 财务审计(22词) · 电商运营(49词) · 食品安全(12词) · 数据AI(52词) · 客户销售(43词)<br>簇间分离度 0.78 · 簇内紧密度 0.15</p>
</div></div></section>''')

# P9 - Structure
add('''<section class="slide dark"><h2>10 · 结构层：页面架构与信息架构</h2><div class="grid2"><div><h3 class="accent">7 个核心页面</h3>
<div class="card" style="margin-bottom:1vh"><span class="n">首页 · 搜索门户</span> — 搜索框 + 城市/行业/学历/薪资快捷筛选 + 9大行业标签卡片 + 最新招聘滚动列表</div>
<div class="card" style="margin-bottom:1vh"><span class="n">检索结果页</span> — 左侧筛选面板(行业/城市多选 + 学历/经验/薪资滑块/日期) + 右侧结果列表(职位卡片: 岗位名+企业+股票简称+月薪+学历+经验+描述高亮) + 排序选择器 + 分页器</div>
<div class="card" style="margin-bottom:1vh"><span class="n">职位详情页</span> — 基本信息区 + 完整描述(高亮) + Top-15关键词标签(可点击搜索) + NB分类预测(原始vs预测+置信度) + 相似职位推荐(5张卡片) + 分词效果对比按钮(弹出7种分词器模态框)</div>
<div class="card" style="margin-bottom:1vh"><span class="n">行业浏览页</span> — 9大行业列表(含职位数量) · 点击进入行业职位列表(支持二次关键词检索)</div>
<div class="card" style="margin-bottom:1vh"><span class="n">数据统计页</span> — 行业排行柱状图 · 城市热度 · 薪资直方图 · 学历/经验饼图</div>
<div class="card" style="margin-bottom:1vh"><span class="n">聚类视图页</span> — 10个技能族群词云矩阵 + PCA散点图 + 各族群关键词</div>
<div class="card"><span class="n">关于页</span> — 系统简介 · 技术栈 · AI辅助声明 · 组员信息</div>
</div><div><h3 class="accent">用户交互流程</h3>
<div style="background:rgba(210,153,29,.06);padding:2vh 1.6vw;border-radius:6px;margin-bottom:1.6vh"><p style="font-size:15px;line-height:2"><span class="n">搜索流：</span>首页输入关键词 → 检索结果(筛选+排序+分页) → 点击卡片 → 详情页(高亮+标签+分类预测+分词对比+相似推荐) → 返回浏览 / 修改条件</p></div>
<div style="background:rgba(210,153,29,.06);padding:2vh 1.6vw;border-radius:6px;margin-bottom:1.6vh"><p style="font-size:15px;line-height:2"><span class="n">浏览流：</span>首页点击行业标签 → 行业职位列表(可二次关键词检索) → 详情页</p></div>
<div style="background:rgba(210,153,29,.06);padding:2vh 1.6vw;border-radius:6px"><p style="font-size:15px;line-height:2"><span class="n">分析流：</span>数据统计页 / 聚类视图页 → 浏览宏观数据 → 点击下钻</p></div>
<h3 class="accent" style="margin-top:3vh">信息架构</h3>
<p>Flask 路由：<span class="mono">/ /search /job/&lt;id&gt; /industry /stats /cluster /about</span> + API 接口 <span class="mono">/api/search /api/job/&lt;id&gt; /api/stats</span></p>
<p style="margin-top:1.2vh">前端：Jinja2 模板继承 (base.html → 所有页面) · 职位卡片复用组件 · 分页器组件 · 筛选面板组件</p>
</div></div></section>''')

# P10 - Framework + Surface
add('''<section class="slide dark"><h2>11 · 框架层与表现层</h2>
<div class="grid4"><div class="card"><h3 class="accent">线框设计</h3><p style="font-size:14px">首页：搜索框居中，筛选条件横向排列，行业标签卡片网格，最新招聘列表。</p><p style="font-size:14px;margin-top:1vh">结果页：左侧 20% 筛选面板，右侧 80% 结果列表。详情页：上下分区，信息区→描述区→标签区→推荐区。</p></div>
<div class="card"><h3 class="accent">导航与信息设计</h3><p style="font-size:14px">顶部固定导航栏：Logo + [搜索] [行业浏览] [数据统计] [聚类视图] [关于]。</p><p style="font-size:14px;margin-top:1vh">职位卡片统一模板。统一分页器、排序选择器、空状态/加载状态提示组件。</p></div>
<div class="card"><h3 class="accent">视觉风格</h3><p style="font-size:14px">配色：主色 <span style="color:#1a73e8">#1a73e8</span> 蓝 · 辅色 <span style="color:#34a853">#34a853</span> 绿 · 强调 <span style="color:#ea4335">#ea4335</span> 红</p><p style="font-size:14px;margin-top:1vh">字体：微软雅黑 · 标题 24px · 正文 14px · 辅助 12px。背景 #f8f9fa · 卡片 #fff</p></div>
<div class="card"><h3 class="accent">一致性设计</h3><p style="font-size:14px">所有页面继承 base.html 统一布局（导航栏+内容区+页脚）。配色/字体/间距全局 CSS 变量统一。</p><p style="font-size:14px;margin-top:1vh">页脚：版权信息 + 数据来源声明 + AI辅助标注。</p></div>
</div></section>''')

# Close deck, add nav JS
out.append('''</div><div id="nav"></div><div id="ov"><div class="grid"></div></div><script>
const D=document.getElementById('deck'),S=D.querySelectorAll('.slide');let idx=0,total=S.length,lock=false;
const nav=D.parentElement.querySelector('#nav'),ov=D.parentElement.querySelector('#ov'),ovg=ov.querySelector('.grid');
S.forEach((s,i)=>{const b=document.createElement('button');b.className='dot';b.onclick=()=>go(i);nav.appendChild(b)});
function go(n){if(lock||n<0||n>=total)return;idx=n;D.style.transform='translateX('+(-idx*100)+'vw)';nav.querySelectorAll('.dot').forEach((d,i)=>d.classList.toggle('active',i===idx));lock=true;setTimeout(()=>lock=false,450)}
function toggleOV(){let on=ov.style.display==='block';if(on){ov.style.display='none'}else{ovg.innerHTML='';S.forEach((s,i)=>{const d=document.createElement('div');d.className='item'+(i===idx?' active':'');const h=s.querySelector('h2');d.textContent=(i+1+'').padStart(2,'0')+'  '+(h?h.textContent.trim().slice(0,30):'Slide '+(i+1));d.onclick=()=>{ov.style.display='none';go(i)};ovg.appendChild(d)});ov.style.display='block'}}
addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();toggleOV();return}if(ov.style.display==='block')return;if(e.key==='ArrowRight'||e.key==='ArrowDown'||e.key==='PageDown'||e.key===' '){e.preventDefault();go(idx+1)}if(e.key==='ArrowLeft'||e.key==='ArrowUp'||e.key==='PageUp'){e.preventDefault();go(idx-1)}if(e.key==='Home')go(0);if(e.key==='End')go(total-1)});
addEventListener('wheel',e=>{go(idx+(e.deltaY>0?1:-1))},{passive:true});go(0);
</script></body></html>''')

# Write
full = '\n'.join(out)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(full)

imgs = full.count('data:image/png;base64')
print(f'Done! {len(full):,} bytes, {S} slides' if False else f'Done! {len(full):,} bytes, {full.count("<section class=")} slides, {imgs} images')
