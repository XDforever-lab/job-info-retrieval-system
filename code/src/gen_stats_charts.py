"""生成学术论文风格的统计图表（PNG）"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

# ── 路径设置 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV = os.path.join(BASE_DIR, 'output', '3_cleaner', '上市公司招聘数据_cleaned.csv')
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
    print("=" * 50)
    print(f"四张图表已保存至: {OUT_DIR}")
    print("=" * 50)
