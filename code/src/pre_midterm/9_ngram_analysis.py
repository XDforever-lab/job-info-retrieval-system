"""
N-gram 统计分析与齐普夫定律验证 —— 覆盖 To Do List #86 ~ #92

    #86: 加载全量分词数据，提取职位描述_分词字段
    #87: 绘制词级 unigram "词频 vs 排名" 散点图
    #88: 绘制双对数曲线
    #89: 线性回归拟合 + R²
    #90: 齐普夫定律验证结论
    #91: 高频尾 / 低频长尾偏离分析
    #92: 导出 Top-500 高频词表

用法:
    conda activate my_project
    python src/9_ngram_analysis.py

输出:
    output/9_ngram/ngram_report.json      统计结果
    output/9_ngram/zipf_scatter.png       散点图
    output/9_ngram/zipf_loglog.png        双对数图
    output/9_ngram/top500_words.txt       Top-500 词表
    output/9_ngram/ngram_conclusion.txt   分析结论
"""

import json
import os
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEGMENTED_CSV, DIR_9_NGRAM

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── 输出路径 ──
REPORT_JSON = os.path.join(DIR_9_NGRAM, "ngram_report.json")
CHART_SCATTER = os.path.join(DIR_9_NGRAM, "zipf_scatter.png")
CHART_LOGLOG = os.path.join(DIR_9_NGRAM, "zipf_loglog.png")
TOP500_FILE = os.path.join(DIR_9_NGRAM, "top500_words.txt")
CONCLUSION_FILE = os.path.join(DIR_9_NGRAM, "ngram_conclusion.txt")


# ══════════════════════════════════════════════════
# #86: 加载分词数据 + 词频统计
# ══════════════════════════════════════════════════

def load_and_count():
    print("[#86] 加载分词数据...")
    df = pd.read_csv(SEGMENTED_CSV, encoding="utf-8-sig", engine="python")
    col = "职位描述_分词"
    if col not in df.columns:
        raise KeyError(f"缺少列 '{col}'，可用列: {list(df.columns)}")

    print(f"  文档数: {len(df):,}")

    # 收集所有词条
    counter = Counter()
    total_tokens = 0
    for i, text in enumerate(df[col].dropna()):
        words = str(text).split("/")
        for w in words:
            w = w.strip()
            if w:
                counter[w] += 1
                total_tokens += 1

    print(f"  总词次 (tokens): {total_tokens:,}")
    print(f"  唯一词数 (types): {len(counter):,}")

    # 转为 DataFrame（词频降序）
    freq_df = pd.DataFrame(counter.most_common(), columns=["词条", "频次"])
    freq_df["排名"] = range(1, len(freq_df) + 1)
    freq_df["频率"] = freq_df["频次"] / total_tokens

    return freq_df, total_tokens


# ══════════════════════════════════════════════════
# #87: 词频 vs 排名散点图
# ══════════════════════════════════════════════════

def plot_scatter(freq_df):
    print("[#87] 绘制词频 vs 排名散点图...")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(freq_df["排名"], freq_df["频次"], s=1, alpha=0.4, color="#5D9CE8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("排名 (log)")
    ax.set_ylabel("词频 (log)")
    ax.set_title("上市公司招聘数据 — 词频 vs 排名 (Unigram, 双对数坐标)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plt.savefig(CHART_SCATTER, dpi=150)
    plt.close()
    print(f"  已保存: {CHART_SCATTER}")


# ══════════════════════════════════════════════════
# #88: 双对数曲线 + #89: 线性回归
# ══════════════════════════════════════════════════

def plot_loglog_and_fit(freq_df):
    print("[#88-#89] 绘制双对数曲线 + 线性回归拟合...")
    ranks = freq_df["排名"].values.astype(float)
    freqs = freq_df["频次"].values.astype(float)

    log_rank = np.log10(ranks)
    log_freq = np.log10(freqs)

    # 线性回归
    X = log_rank.reshape(-1, 1)
    y = log_freq
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    y_pred = model.predict(X)

    print(f"  拟合结果: log(F) = {slope:.4f} * log(R) + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  斜率 = {slope:.4f} (齐普夫定律预期: -1.0)")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(log_rank, log_freq, s=1, alpha=0.3, color="#5D9CE8", label="实际数据")
    ax.plot(log_rank, y_pred, color="#EA4335", linewidth=2,
            label=f"线性拟合 (R²={r2:.4f}, slope={slope:.3f})")
    ax.set_xlabel("log10(排名)")
    ax.set_ylabel("log10(词频)")
    ax.set_title("上市公司招聘数据 — Zipf 双对数曲线 (Unigram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(CHART_LOGLOG, dpi=150)
    plt.close()
    print(f"  已保存: {CHART_LOGLOG}")

    return {
        "斜率": round(slope, 4),
        "截距": round(intercept, 4),
        "R²": round(r2, 4),
        "预期斜率": -1.0,
        "斜率偏差": round(abs(slope - (-1.0)), 4),
    }


# ══════════════════════════════════════════════════
# #90 + #91: 齐普夫结论 + 首尾偏离分析
# ══════════════════════════════════════════════════

def write_conclusion(freq_df, fit, total_tokens):
    print("[#90-#91] 撰写分析结论...")
    lines = []
    lines.append("上市公司招聘数据 — Zipf 定律验证报告")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"语料规模: {total_tokens:,} 词次, {len(freq_df):,} 唯一词型")
    lines.append(f"线性拟合: log(F) = {fit['斜率']} * log(R) + {fit['截距']}")
    lines.append(f"拟合优度 R² = {fit['R²']}")
    lines.append(f"斜率偏差 = {fit['斜率偏差']} (预期 -1.0)")
    lines.append("")

    # #90 结论
    if fit["R²"] >= 0.85 and abs(fit["斜率"] + 1.0) <= 0.3:
        lines.append("#90: 齐普夫定律验证: 基本成立")
        lines.append(f"  R²={fit['R²']} (>=0.85)，斜率={fit['斜率']} (接近-1.0)")
        lines.append("  招聘文本的词频分布符合 Zipf 定律——词频与排名近似成反比。")
    else:
        lines.append("#90: 齐普夫定律验证: 部分成立")
        lines.append(f"  R²={fit['R²']}，斜率={fit['斜率']}")
        lines.append("  存在一定偏差，详见下方偏离分析。")

    # #91 首尾偏离分析
    lines.append("")
    lines.append("#91: 首尾偏离分析")
    lines.append("-" * 40)

    top_20 = freq_df.head(20)
    tail_start = max(0, int(len(freq_df) * 0.85))
    tail_sample = freq_df.iloc[tail_start:tail_start + 20]

    lines.append(f"  高频头部 (Top-20 词):")
    for _, row in top_20.iterrows():
        w = str(row["词条"])
        # Filter to only show Chinese+meaningful words (not pure punct/numbers)
        if re.match(r"^[一-龥]+$", w):
            lines.append(f"    {w:<8s} {row['频次']:>8,}次  (排名 {int(row['排名'])})")

    lines.append(f"")
    lines.append(f"  低频长尾 (排名 {tail_start:,} 之后):")
    once_count = (freq_df["频次"] == 1).sum()
    lines.append(f"    仅出现 1 次的词 (hapax legomena): {once_count:,} 个")
    lines.append(f"    占全部词型的 {once_count/len(freq_df)*100:.1f}%")

    lines.append(f"")
    lines.append("  偏离分析:")
    lines.append("  - 高频端: 虚词、标点、数字等高频词可能偏离理想 Zipf 曲线")
    lines.append("  - 低频端: 长尾处频次=1的大量稀有词形成'尾巴'，词频不再平滑递减")
    lines.append("  - 这是自然语言的普遍特征，与理论 Zipf 定律的偏离在预期范围内")

    with open(CONCLUSION_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  已保存: {CONCLUSION_FILE}")
    print("\n" + "\n".join(lines))


# ══════════════════════════════════════════════════
# #92: 导出 Top-500 高频词表
# ══════════════════════════════════════════════════

def export_top500(freq_df, total_tokens):
    print(f"\n[#92] 导出 Top-500 高频词表...")
    top500 = freq_df.head(500).copy()
    top500["频率"] = top500["频次"] / total_tokens

    with open(TOP500_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'排名':<8} {'词条':<16} {'频次':>10} {'频率':>12}\n")
        f.write("-" * 50 + "\n")
        for _, row in top500.iterrows():
            f.write(f"{int(row['排名']):<8} {str(row['词条']):<16} "
                    f"{int(row['频次']):>10,} {row['频率']:>11.6f}\n")
    print(f"  已保存: {TOP500_FILE}")
    print(f"  示例 (Top-10):")
    for _, row in top500.head(10).iterrows():
        w = str(row["词条"])
        if re.match(r"^[一-龥]+$", w):
            print(f"    {int(row['排名']):>3}. {w:<10s}  {int(row['频次']):>8,}次")


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main():
    os.makedirs(DIR_9_NGRAM, exist_ok=True)
    print("=" * 60)
    print("  N-gram 分析 & Zipf 验证 #86 ~ #92")
    print("=" * 60)

    # #86
    freq_df, total_tokens = load_and_count()

    # #87
    plot_scatter(freq_df)

    # #88 + #89
    fit = plot_loglog_and_fit(freq_df)

    # #90 + #91
    write_conclusion(freq_df, fit, total_tokens)

    # #92
    export_top500(freq_df, total_tokens)

    # 保存 JSON 报告
    report = {
        "总词次": total_tokens,
        "唯一词数": len(freq_df),
        "齐普夫拟合": fit,
        "Top-10 词": freq_df.head(10)[["词条", "频次"]].to_dict("records"),
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {REPORT_JSON}")
    print(f"{'=' * 60}")
    print(f"  #86 ~ #92 完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
