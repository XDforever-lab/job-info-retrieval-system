"""生成学术论文风格的统计图表（PNG）"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import re, math, json

# ── 路径设置 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV = os.path.join(BASE_DIR, 'output', '3_cleaner', '上市公司招聘数据_cleaned.csv')
SEGMENTED_CSV = os.path.join(BASE_DIR, 'output', '8_full_seg', '上市公司招聘数据_segmented.csv')
EVAL_REPORT = os.path.join(BASE_DIR, 'output', '7_seg_eval', 'seg_eval_report.json')
KW_EVAL_REPORT = os.path.join(BASE_DIR, 'output', '10_keywords', 'eval_report.json')
CLS_EVAL_REPORT = os.path.join(BASE_DIR, 'output', '13_classify', 'classify_eval.json')
STOPWORDS_FILE = os.path.join(BASE_DIR, 'output', 'stopwords.txt')
SEARCH_EVAL_REPORT = os.path.join(BASE_DIR, 'output', '17_eval', 'search_eval_report.json')
OUT_DIR = os.path.join(BASE_DIR, 'output', '18_stats')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 中文字体 ──
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 学术风格配色 ──
C_BLUE   = '#2B5F8A'
C_ORANGE = '#C44E2C'
C_GREEN  = '#3A7D5A'
C_PURPLE = '#6B4E7D'
C_TEAL   = '#287D7D'
C_RED    = '#A83C3C'
C_GOLD   = '#8B7233'
C_GRAY   = '#555555'

PALETTE_4 = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_TEAL, C_RED, C_GOLD, C_GRAY]
PALETTE_9 = ['#1F4E79','#3A6B9F','#4D8AB5','#6FA3C7','#8FBCD6','#A8CDE0','#C2DEEB','#DCE9F2','#EEF4F9']
PALETTE_10 = ['#1F4E79','#2C5F8A','#3A709B','#4881AC','#5692BD','#6BA3CA','#80B4D7','#95C5E4','#AAD6F1','#BFE7FE']


def _is_nan(v):
    try:
        return pd.isna(v) or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


def load_data(path):
    df = pd.read_csv(path)
    print(f"加载数据: {len(df)} 条记录")
    return df


# ══════════════════════════════════════════════════
# 图 1：行业招聘数量柱状图
# ══════════════════════════════════════════════════
def chart_industry(df):
    counts = df['上市公司行业'].value_counts().head(9)
    names = list(counts.index)[::-1]
    values = list(counts.values)[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = PALETTE_9[::-1]
    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.6, height=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=8.5, color=C_GRAY)

    ax.set_xlabel('招聘数量（条）', fontsize=9.5, color=C_GRAY)
    ax.set_title('行业招聘数量分布', fontsize=12, fontweight='bold', color='#333', pad=12)
    ax.tick_params(axis='both', labelsize=8.5, colors=C_GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='x', color='#E8E8E8', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'industry_bar.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 行业柱状图 → {path}")


# ══════════════════════════════════════════════════
# 图 2：城市招聘热度柱状图
# ══════════════════════════════════════════════════
def chart_city(df):
    counts = df['工作城市'].value_counts().head(10)
    names = list(counts.index)
    values = list(counts.values)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = PALETTE_10
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.6, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.012,
                str(val), ha='center', va='bottom', fontsize=7.5, color=C_GRAY)

    ax.set_ylabel('招聘数量（条）', fontsize=9.5, color=C_GRAY)
    ax.set_title('城市招聘热度分布（Top-10）', fontsize=12, fontweight='bold', color='#333', pad=12)
    ax.tick_params(axis='both', labelsize=8.0, colors=C_GRAY)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', color='#E8E8E8', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'city_bar.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 城市柱状图 → {path}")


# ══════════════════════════════════════════════════
# 图 3：薪资分布直方图
# ══════════════════════════════════════════════════
def chart_salary(df):
    salaries = []
    for _, row in df.iterrows():
        try:
            lo = float(row['最低月薪']) if not _is_nan(row['最低月薪']) else 0
            hi = float(row['最高月薪']) if not _is_nan(row['最高月薪']) else 0
            if lo > 0 and hi > 0:
                salaries.append((lo + hi) / 2)
        except (ValueError, TypeError):
            pass

    bins_labels = ['0-5k', '5k-10k', '10k-15k', '15k-20k',
                   '20k-25k', '25k-30k', '30k-40k', '40k+']
    bin_edges = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 200000]

    counts = [0] * len(bins_labels)
    for s in salaries:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= s < bin_edges[i + 1]:
                counts[i] += 1
                break

    percentages = [c / len(salaries) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    color_map = ['#E8EDF2', '#CAD7E3', '#ADC2D5', '#90ACC7', '#7397B9', '#5682AB',
                 '#3A6D9D', C_BLUE]
    bars = ax.bar(bins_labels, counts, color=color_map, edgecolor='white', linewidth=0.6, width=0.65)

    for bar, val, pct in zip(bars, counts, percentages):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.015,
                    f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7.5, color=C_GRAY)

    ax.set_ylabel('招聘数量（条）', fontsize=9.5, color=C_GRAY)
    ax.set_xlabel('平均月薪区间', fontsize=9.5, color=C_GRAY)
    ax.set_title('薪资分布', fontsize=12, fontweight='bold', color='#333', pad=12)
    ax.tick_params(axis='both', labelsize=8.5, colors=C_GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', color='#E8E8E8', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'salary_hist.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 薪资直方图 → {path}")


# ══════════════════════════════════════════════════
# 图 4：学历要求分布饼图
# ══════════════════════════════════════════════════
def chart_education(df):
    counts = df['学历要求'].value_counts().to_dict()

    # 排序：按学历层次
    order = ['初中及以下', '中专', '高中', '中技', '大专', '本科', '硕士', '博士', 'MBA', 'EMBA']
    labels = []
    values = []
    for k in order:
        if k in counts:
            labels.append(k)
            values.append(counts[k])
    for k, v in counts.items():
        if k not in order:
            labels.append(k)
            values.append(v)

    # 合并占比 < 2% 的项为"其他"
    total = sum(values)
    main_labels, main_values, other_val = [], [], 0
    for lbl, val in zip(labels, values):
        if val / total < 0.02:
            other_val += val
        else:
            main_labels.append(lbl)
            main_values.append(val)
    if other_val > 0:
        main_labels.append('其他')
        main_values.append(other_val)

    # 柔和学术色板 — ColorBrewer 风格的调和色调
    palette = ['#7EA8C4', '#B8C9A8', '#E8B298', '#C4A8C4', '#A8C4C4',
               '#D4A8A8', '#C4B898', '#A8B0C4', '#B8C4B0', '#C4B0A8']
    colors = palette[:len(main_labels)]
    # 最大扇区稍微突出，其余不突出
    explode = [0.04 if v == max(main_values) else 0.0 for v in main_values]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    wedges, texts = ax.pie(
        main_values,
        labels=None,
        autopct=None,               # 不在圆环上显示数字
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops={
            'edgecolor': 'white',
            'linewidth': 1.8,
            'width': 0.4,           # 环形（甜甜圈）样式
        },
    )

    # 使用图例代替直接标签，避免拥挤
    legend_labels = [f'{lbl}  ({val}条, {val/total*100:.1f}%)'
                     for lbl, val in zip(main_labels, main_values)]
    ax.legend(
        wedges, legend_labels,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),
        fontsize=8.5,
        frameon=False,
        labelspacing=0.8,
        handletextpad=0.6,
    )

    ax.set_title('学历要求分布', fontsize=13, fontweight='bold', color='#333', pad=18)

    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, 'education_pie.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 学历饼图 → {path}")


# ══════════════════════════════════════════════════
# 图 5：Zipf 双对数曲线（齐普夫定律验证）
# ══════════════════════════════════════════════════
def chart_zipf():
    print("加载分词数据用于词频统计...")
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig")
    col = "职位描述_分词"
    if col not in df.columns:
        print(f"  [WARN] 找不到 '{col}' 列，跳过 Zipf 图")
        return

    # 词频统计
    counter = Counter()
    for text in df[col].dropna():
        for w in str(text).split("/"):
            w = w.strip()
            if w:
                counter[w] += 1

    total_tokens = sum(counter.values())
    unique_types = len(counter)
    print(f"  总词次: {total_tokens:,}, 唯一词型: {unique_types:,}")

    # 按频次降序排列，赋予排名
    sorted_items = counter.most_common()
    ranks = np.arange(1, len(sorted_items) + 1, dtype=float)
    freqs = np.array([v for _, v in sorted_items], dtype=float)

    log_rank = np.log10(ranks)
    log_freq = np.log10(freqs)

    # 线性回归
    X = log_rank.reshape(-1, 1)
    model = LinearRegression().fit(X, log_freq)
    r2 = model.score(X, log_freq)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X)

    print(f"  拟合: log(F) = {slope:.4f} * log(R) + {intercept:.4f},  R^2 = {r2:.4f},  斜率 = {slope:.4f}")

    # 间隔采样（太多点会重叠，每隔N个取一个用于绘图）
    sample_step = max(1, unique_types // 3000)
    sample_idx = np.arange(0, unique_types, sample_step)
    # 确保首尾点保留
    sample_idx = np.unique(np.concatenate([sample_idx, [0, unique_types - 1]]))
    # 多点在前部（高频区）
    extra_head = np.arange(0, min(50, unique_types))
    sample_idx = np.unique(np.concatenate([sample_idx, extra_head]))

    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    # 散点
    ax.scatter(log_rank[sample_idx], log_freq[sample_idx],
               s=4, alpha=0.5, color='#5D8DB8', edgecolors='none', zorder=2,
               label='实际数据')

    # 回归线
    ax.plot(log_rank, y_pred, color='#C44E2C', linewidth=1.8, zorder=3,
            label=f'线性拟合  $R^2={r2:.4f}$')

    # 理论 Zipf 线（斜率=-1）
    zipf_y = log_freq[0] - 1.0 * log_rank
    ax.plot(log_rank, zipf_y, color='#888888', linewidth=1.0, linestyle='--',
            dashes=(6, 4), zorder=1, alpha=0.7,
            label='理论 Zipf 线 (斜率=-1)')

    # 标注框
    textstr = f'$\\log(f) = {slope:.3f} \\cdot \\log(r) + {intercept:.3f}$\n$R^2 = {r2:.4f}$'
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC', alpha=0.9)
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=props, color='#444')

    ax.set_xlabel('$\\log_{10}$(排名)', fontsize=10.5, color=C_GRAY)
    ax.set_ylabel('$\\log_{10}$(词频)', fontsize=10.5, color=C_GRAY)
    ax.set_title('齐普夫定律验证 — 词频-排名双对数分布', fontsize=12, fontweight='bold', color='#333', pad=12)
    ax.legend(fontsize=8.5, loc='upper right', frameon=True, facecolor='white',
              edgecolor='#E0E0E0', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', labelsize=9, colors=C_GRAY)
    ax.grid(True, color='#E8E8E8', linewidth=0.4, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'zipf_loglog.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] Zipf 双对数图 → {path}")


# ══════════════════════════════════════════════════
# 图 6：分词器评测对比表
# ══════════════════════════════════════════════════
def table_seg_eval():
    import json
    with open(EVAL_REPORT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按 F1 升序排列
    order = sorted(data.keys(), key=lambda k: data[k]['F1_classic'])

    col_labels = ['分词器', 'BIES\n完全匹配率', 'P (%)', 'R (%)', 'F1 (%)']
    cell_text = []
    for seg in order:
        d = data[seg]
        cell_text.append([
            seg,
            f"{d['BIES完全匹配率']:.1f}",
            f"{d['P_macro']:.2f}",
            f"{d['R_macro']:.2f}",
            f"{d['F1_classic']:.2f}",
        ])

    # 高亮最佳行（CRF）
    row_colors = []
    for seg in order:
        if seg == 'CRF':
            row_colors.append('#E6F0E6')  # 浅绿高亮
        else:
            row_colors.append('#FFFFFF')

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.14, 0.18, 0.14, 0.14, 0.14],
    )

    # 样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.6)
        cell.set_text_props(color='#333333')

        if key[0] == 0:  # 表头行
            cell.set_facecolor('#E8EDF2')
            cell.set_fontsize(8.5)
            cell.set_text_props(fontweight='bold', color='#2B5F8A')
        else:
            row_idx = key[0] - 1
            cell.set_facecolor(row_colors[row_idx])

    ax.set_title('七种分词器评测对比', fontsize=12, fontweight='bold', color='#333', pad=18)

    # 底部注释
    fig.text(0.5, 0.02, '注：BIES 完全匹配率为全句一字不差匹配的文档占比；P/R/F1 基于词集合交并比计算，不计词序。',
             ha='center', fontsize=7.5, color='#888888')

    plt.tight_layout(pad=2.0)
    path = os.path.join(OUT_DIR, 'seg_eval_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 分词评测表 → {path}")


# ══════════════════════════════════════════════════
# 图 7：关键词抽取方法对比表
# ══════════════════════════════════════════════════
def table_kw_eval():
    import json
    with open(KW_EVAL_REPORT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    col_labels = ['方法', 'P (%)', 'R (%)', 'F1 (%)']
    cell_text = []
    # TF-IDF 排前面（更优）
    for method in ['TF-IDF', 'TextRank']:
        d = data[method]
        cell_text.append([
            method,
            f"{d['P']:.2f}",
            f"{d['R']:.2f}",
            f"{d['F1']:.2f}",
        ])

    # TF-IDF 行高亮
    row_colors = ['#E6F0E6', '#FFFFFF']

    fig, ax = plt.subplots(figsize=(4.8, 1.6))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.22, 0.18, 0.18, 0.18],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.6)
        cell.set_text_props(color='#333333')
        if key[0] == 0:  # 表头
            cell.set_facecolor('#E8EDF2')
            cell.set_fontsize(9)
            cell.set_text_props(fontweight='bold', color='#2B5F8A')
        else:
            cell.set_facecolor(row_colors[key[0] - 1])

    ax.set_title('关键词抽取方法评测对比（Top-10）', fontsize=12, fontweight='bold', color='#333', pad=14)

    # 底部注释
    fig.text(0.5, 0.04, '注：P/R/F1 以人工标注的黄金标准关键词集合为基准，对 100 篇文档取宏平均。',
             ha='center', fontsize=7.5, color='#888888')

    plt.tight_layout(pad=2.0)
    path = os.path.join(OUT_DIR, 'keyword_eval_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 关键词评测表 → {path}")


# ══════════════════════════════════════════════════
# 图 8：KNN 与朴素贝叶斯分类评测对比表
# ══════════════════════════════════════════════════
def table_classify_eval():
    import json
    with open(CLS_EVAL_REPORT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    knn_best = data['KNN']['K=3']
    nb_best = data['NB']['a=0.1']

    col_labels = ['方法', '评测方式', 'P (%)', 'R (%)', 'F1 (%)', 'BEP (%)']
    cell_text = [
        ['KNN\n(K=3)', '宏观', f"{knn_best['macro']['P']*100:.2f}", f"{knn_best['macro']['R']*100:.2f}",
         f"{knn_best['macro']['F1']*100:.2f}", f"{knn_best['macro']['BEP']*100:.2f}"],
        ['', '微观', f"{knn_best['micro']['P']*100:.2f}", f"{knn_best['micro']['R']*100:.2f}",
         f"{knn_best['micro']['F1']*100:.2f}", f"{knn_best['micro']['BEP']*100:.2f}"],
        ['朴素贝叶斯\n(α=0.1)', '宏观', f"{nb_best['macro']['P']*100:.2f}", f"{nb_best['macro']['R']*100:.2f}",
         f"{nb_best['macro']['F1']*100:.2f}", f"{nb_best['macro']['BEP']*100:.2f}"],
        ['', '微观', f"{nb_best['micro']['P']*100:.2f}", f"{nb_best['micro']['R']*100:.2f}",
         f"{nb_best['micro']['F1']*100:.2f}", f"{nb_best['micro']['BEP']*100:.2f}"],
    ]

    # NB 方法名行高亮
    row_colors = ['#FFFFFF', '#FFFFFF', '#E6F0E6', '#E6F0E6']

    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.16, 0.10, 0.12, 0.12, 0.12, 0.12],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.6)
        cell.set_text_props(color='#333333')
        if key[0] == 0:
            cell.set_facecolor('#E8EDF2')
            cell.set_fontsize(8.5)
            cell.set_text_props(fontweight='bold', color='#2B5F8A')
        else:
            cell.set_facecolor(row_colors[key[0] - 1])

    ax.set_title('KNN与朴素贝叶斯分类评测对比（9类 / 400条样本）',
                 fontsize=12, fontweight='bold', color='#333', pad=16)

    fig.text(0.5, 0.03,
             '注：KNN 取最优 K=3（余弦相似度）；NB 取最优 α=0.1（拉普拉斯平滑）。宏观平均先算各类再求均值，微观平均先汇总所有类再计算。',
             ha='center', fontsize=7, color='#888888')

    plt.tight_layout(pad=2.2)
    path = os.path.join(OUT_DIR, 'classify_eval_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 分类评测表 → {path}")


# ══════════════════════════════════════════════════
# 图 9：词汇聚类 PCA 二维散点图
# ══════════════════════════════════════════════════
def chart_word_cluster_pca():
    print("构建词向量并进行词汇聚类...")
    # ── 加载停用词 ──
    sw = set()
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, encoding="utf-8") as f:
            sw = {l.strip() for l in f if l.strip() and not l.startswith("#")}

    # ── 加载分词数据 ──
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig")
    print(f"  文档数: {len(df):,}")

    # ── 逐文档统计词频（仅中文2字以上实词） ──
    doc_word_counts = []
    all_words = Counter()
    for i in range(len(df)):
        seg = str(df.iloc[i].get("职位描述_分词", ""))
        title = str(df.iloc[i].get("招聘岗位_分词", ""))
        tokens = [w.strip() for w in (title + "/" + seg).split("/")
                  if w.strip() and w != "nan" and w not in sw
                  and len(w.strip()) >= 2
                  and not re.search(r'[，。；：、（）【】\-;:,.!?\d]', w.strip())]
        doc_word_counts.append(Counter(tokens))
        all_words.update(doc_word_counts[-1].keys())

    N = len(doc_word_counts)
    WORD_MIN_DF = 10
    WORD_TOP_N = 800

    words_above_thresh = {w for w, c in all_words.items() if c >= WORD_MIN_DF}
    print(f"  候选词 (DF>={WORD_MIN_DF}): {len(words_above_thresh):,}")

    # ── 按 TF-IDF 方差选 Top-N 区分度最高的词 ──
    word_tfidf_var = {}
    for w in words_above_thresh:
        df_w = sum(1 for dc in doc_word_counts if w in dc)
        idf = math.log(N / (df_w + 1))
        scores = [dc.get(w, 0) * idf for dc in doc_word_counts]
        word_tfidf_var[w] = np.var(scores)
    top_words = sorted(word_tfidf_var, key=word_tfidf_var.get, reverse=True)[:WORD_TOP_N]
    print(f"  选取区分度最高的 {len(top_words)} 个词")

    # ── 构建词-文档 TF-IDF 矩阵 ──
    word_list = list(top_words)
    X = np.zeros((len(word_list), N))
    for wi, w in enumerate(word_list):
        df_w = sum(1 for dc in doc_word_counts if w in dc)
        idf = math.log(N / (df_w + 1))
        for di, dc in enumerate(doc_word_counts):
            if w in dc:
                X[wi, di] = dc[w] * idf
    X_norm = normalize(X, norm='l2')
    print(f"  词向量矩阵: {X_norm.shape}")

    # ── K-Means 聚类 (K=10) ──
    BEST_K = 10
    km = KMeans(n_clusters=BEST_K, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_norm)
    print(f"  K-Means 完成, K={BEST_K}")

    # ── PCA 降维到 2D ──
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_norm)
    ev1, ev2 = pca.explained_variance_ratio_
    print(f"  PCA: PC1={ev1:.1%}, PC2={ev2:.1%}, 合计={ev1+ev2:.1%}")

    # ── 生成簇的简短名称 ──
    cluster_names = {}
    for k in range(BEST_K):
        cluster_words = [word_list[i] for i in np.where(labels == k)[0]]
        cluster_names[k] = cluster_words[:5]

    # ── 绘图 ──
    # 学术风格配色 (10簇)
    palette = ['#4A90B8', '#C47E5A', '#6B9E6E', '#B878A8', '#8B7D5A',
               '#C4565A', '#5A8B8B', '#8B6BAA', '#A09060', '#6090A0']

    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    for k in range(BEST_K):
        mask = labels == k
        size = mask.sum()
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=30, alpha=0.75, color=palette[k],
                   edgecolors='white', linewidth=0.3,
                   label=f'族群 {k+1} ({size}词)')
        # 每簇标注3个代表性词
        cluster_coords = coords[mask]
        cw = [word_list[i] for i in np.where(mask)[0]]
        for j in range(min(3, len(cw))):
            ax.annotate(cw[j], (cluster_coords[j, 0], cluster_coords[j, 1]),
                       fontsize=6.5, alpha=0.7, color='#444444',
                       ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({ev1:.1%})', fontsize=10.5, color=C_GRAY)
    ax.set_ylabel(f'PC2 ({ev2:.1%})', fontsize=10.5, color=C_GRAY)
    ax.set_title(f'词汇聚类 PCA 二维可视化 — {len(word_list)} 个技能词 → {BEST_K} 个技能族群',
                 fontsize=12, fontweight='bold', color='#333', pad=14)

    ax.legend(fontsize=7.5, loc='upper right', ncol=2,
              frameon=True, facecolor='white', edgecolor='#E0E0E0',
              framealpha=0.9, handletextpad=0.5, labelspacing=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', labelsize=8.5, colors=C_GRAY)
    ax.grid(True, color='#E8E8E8', linewidth=0.3, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'word_cluster_pca.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 词汇聚类PCA图 → {path}")


# ══════════════════════════════════════════════════
# 图 10：检索评测指标汇总表
# ══════════════════════════════════════════════════
def table_search_eval():
    import json
    with open(SEARCH_EVAL_REPORT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    per_query = data['per_query']
    N = len(per_query)

    col_labels = ['编号', '查询词', '类型', 'P@5', 'P@10', 'R-Prec', 'AP']
    cell_text = []
    for q in per_query:
        cell_text.append([
            f"#{q['query_id']}",
            q['query'],
            q['desc'],
            f"{q['P@5']:.2f}",
            f"{q['P@10']:.2f}",
            f"{q['R-Precision']:.3f}",
            f"{q['AP']:.3f}",
        ])

    # 汇总行
    cell_text.append([
        '均值', '', f'{N}个查询',
        f"{data['avg_P@5']:.2f}",
        f"{data['avg_P@10']:.2f}",
        f"{data['avg_R_Precision']:.3f}",
        f"{data['MAP']:.3f}",
    ])

    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(7.5, n_rows * 0.32 + 1.2))
    ax.axis('off')

    col_widths = [0.07, 0.18, 0.20, 0.09, 0.09, 0.11, 0.10]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    # 列头
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#E8EDF2')
        cell.set_text_props(fontweight='bold', fontsize=8, color='#2B5F8A')
        cell.set_edgecolor('#CCCCCC')

    # 数据行
    for i in range(1, n_rows + 1):
        bg = '#FFFEF8' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#E0E0E0')
            cell.set_text_props(color='#333333')
            
    # 汇总行特殊样式
    last_row = n_rows
    for j in range(len(col_labels)):
        cell = table[last_row, j]
        cell.set_facecolor('#E6EDF3')
        cell.set_text_props(fontweight='bold', color='#2B5F8A', fontsize=8.5)
        cell.set_edgecolor('#CCCCCC')

    # 高亮 P@5 >= 1.0 的单元格
    for i, q in enumerate(per_query):
        row_idx = i + 1
        if q['P@5'] >= 1.0:
            table[row_idx, 3].set_text_props(fontweight='bold', color='#2B7A3D')
        if q['P@10'] >= 1.0:
            table[row_idx, 4].set_text_props(fontweight='bold', color='#2B7A3D')
        if q['AP'] >= 0.99:
            table[row_idx, 6].set_text_props(fontweight='bold', color='#2B7A3D')

    ax.set_title('离线检索评测 — 各查询 P@5 / P@10 / R-Precision / AP（16个查询，人工标注）',
                 fontsize=11, fontweight='bold', color='#333', pad=16)

    fig.text(0.5, 0.02,
             '注：MAP=0.932，avg P@5=0.75，avg P@10=0.63，avg R-Precision=0.89。粗体标绿表示满分。检索采用 TF-IDF 加权(lrc)+倒排索引AND逻辑。',
             ha='center', fontsize=7, color='#888888')

    plt.tight_layout(pad=3.0)
    path = os.path.join(OUT_DIR, 'search_eval_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 检索评测表 → {path}")


# ══════════════════════════════════════════════════
# 图 11：11点插值 P/R 曲线
# ══════════════════════════════════════════════════
def chart_pr_curve():
    import json
    with open(SEARCH_EVAL_REPORT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    curve = data['pr_curve_11pt']
    recalls = [p['recall'] for p in curve]
    precisions = [p['precision'] for p in curve]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    ax.plot(recalls, precisions, 'o-', color=C_BLUE,
            linewidth=2.0, markersize=8, markerfacecolor='white',
            markeredgewidth=2.0, markeredgecolor=C_BLUE,
            label=f'插值P/R曲线 (MAP={data["MAP"]:.3f})', zorder=3)

    # 在每个点标注精确率
    for r, p in zip(recalls, precisions):
        ax.annotate(f'{p:.3f}', (r, p),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=7.5, color='#666666')

    ax.set_xlabel('Recall', fontsize=11, color=C_GRAY)
    ax.set_ylabel('Precision', fontsize=11, color=C_GRAY)
    ax.set_title('11点插值 Precision-Recall 曲线',
                 fontsize=12, fontweight='bold', color='#333', pad=12)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.85, 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=9, loc='lower left', frameon=True,
              facecolor='white', edgecolor='#E0E0E0', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(axis='both', labelsize=9, colors=C_GRAY)
    ax.grid(True, color='#E8E8E8', linewidth=0.4, alpha=0.7, zorder=0)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'pr_curve_11pt.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] P/R曲线 → {path}")


# ══════════════════════════════════════════════════
# 图 12：六大实验评测结果汇总表
# ══════════════════════════════════════════════════
def table_all_experiments():
    _F1 = '%'
    col_labels = ['实验编号', '实验名称', '数据规模', '最优方法', '核心指标', '指标值', '说明']
    cell_text = [
        ['1', '中文分词评测', '200条金标准\n7种分词器',
         'CRF\n(sklearn-crfsuite)', '集合 F1',
         '74.77', 'BIES字级标注; LBFGS优化\n特征: 当前字+前后字+边界标记'],

        ['2', '关键词抽取', '100条文档\n人工标注 Top-10',
         'TF-IDF\n(逆文档频率加权)', 'Top-10 F1',
         '44.61', 'IDF过滤全局高频通用词\n优于TextRank(41.92)'],

        ['3', '文本分类', '400条训练集\n9个行业大类',
         '朴素贝叶斯\n(α=0.1 拉普拉斯平滑)', '微平均 F1',
         '25.00', '基线~11%; 对数转换防下溢\nKNN K=3 微平均F1=16.25'],

        ['4', '词汇聚类', '5,000条文档\n800个技能词',
         'K-Means K=10\n(词-文档 L2 向量)', '类内相似度',
         '0.149', 'PCA降维可视化\n10个技能族群, 类间距离0.779'],

        ['5', '倒排索引构建', '5,000篇文档\n38,288个词条',
         '全字段倒排\n(Lexicon + Posting List)', '词典规模',
         '38,288', '平均DF=11.1; 构建耗时0.8s\n检索加速比 200~500×'],

        ['6', '检索效果评测', '16个查询\n10条@查询人工标注',
         'TF-IDF(ltc)加权\nAND逻辑倒排查询', 'MAP',
         '0.932', 'P@5=0.75, P@10=0.63\nR-Precision=0.89, NDCG=0.96'],
    ]

    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(9.0, n_rows * 0.72 + 1.0))
    ax.axis('off')

    col_widths = [0.08, 0.12, 0.16, 0.19, 0.11, 0.09, 0.25]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    # 列头
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2B5F8A')
        cell.set_text_props(fontweight='bold', fontsize=8.5, color='#FFFFFF')
        cell.set_edgecolor('#1E4A6E')

    # 数据行
    for i in range(1, n_rows + 1):
        bg = '#F5F8FB' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#D8DFE8')
            cell.set_text_props(color='#333333', fontsize=8)
            # 方法名列加深色
            if j == 3:
                cell.set_text_props(fontweight='bold', color='#2B5F8A', fontsize=8)
            # 指标值加粗
            if j == 5:
                cell.set_text_props(fontweight='bold', fontsize=9, color='#333333')

    ax.set_title('六大实验评测结果汇总',
                 fontsize=13, fontweight='bold', color='#333', pad=16)

    fig.text(0.5, 0.015,
             '注：实验1-3为前期核心实验（分词 → 标引 → 分类），实验4-6为后期检索系统实验（聚类 → 索引 → 评测）。所有实验均基于5,000条清洗后的上市公司招聘数据。',
             ha='center', fontsize=7, color='#888888')

    plt.tight_layout(pad=3.5)
    path = os.path.join(OUT_DIR, 'all_experiments_summary.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] 实验汇总表 → {path}")


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 50)
    print("生成学术论文风格统计图表")
    print("=" * 50)
    df = load_data(CLEANED_CSV)
    print(f"数据字段: {list(df.columns)}")
    print(f"总记录数: {len(df)}")
    print("-" * 50)
    chart_industry(df)
    chart_city(df)
    chart_salary(df)
    chart_education(df)
    chart_zipf()
    table_seg_eval()
    table_kw_eval()
    table_classify_eval()
    chart_word_cluster_pca()
    table_search_eval()
    chart_pr_curve()
    table_all_experiments()
    print("=" * 50)
    print(f"图表已保存至: {OUT_DIR}")
    print("=" * 50)
