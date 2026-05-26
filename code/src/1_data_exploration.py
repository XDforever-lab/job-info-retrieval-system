"""
数据探索脚本 —— 覆盖 To Do List 2.1 节 #16 ~ #25
    16. 各字段缺失率统计
    17. 81个行业的记录数量分布
    18. 企业的记录数量分布
    19. 城市地理分布 Top-20
    20. 月薪数值分布
    21. 学历要求类别分布
    22. 要求经验类别分布
    23. 招聘类别分布
    24. 发布日期的年份与月份分布
    25. 输出完整数据探索报告（保存为 JSON + 打印控制台表格）

用法:
    conda activate my_project
    python src/data_exploration.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, STATS_SUMMARY

# ── 0. 数据加载 ──────────────────────────────────

def load_data():
    """加载清洗后的招聘数据 CSV"""
    csv_path = os.path.join(DATA_DIR, "上市公司招聘数据2026.csv")
    encodings = ["gbk", "gb2312", "gb18030", "utf-8"]

    for enc in encodings:
        try:
            with open(csv_path, "rb") as f:
                f.read(50000).decode(enc)
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)

            # 强制薪资列转为数值（字符串/空值 → NaN）
            for col in ["最低月薪", "最高月薪"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            print(f"[加载完成] 编码={enc}, 行数={len(df):,}, 字段数={len(df.columns)}")
            return df
        except (UnicodeDecodeError, LookupError):
            continue

    raise RuntimeError("无法识别 CSV 编码")


# ── 辅助函数 ─────────────────────────────────────

def fmt_pct(num, denom):
    """百分比格式化"""
    if denom == 0:
        return "0.00%"
    return f"{num / denom * 100:.2f}%"


def fmt_num(n):
    """千分位数字格式化"""
    return f"{n:,}"


def count_distribution(series, top_n=20):
    """统计分类列的频次分布，返回 dict"""
    counts = series.value_counts().head(top_n)
    return {str(k): int(v) for k, v in counts.items()}


def numeric_stats(series, label=""):
    """数值列的基本统计量"""
    s = series.dropna()
    if len(s) == 0:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "p25": None, "p75": None}
    return {
        "count": int(len(s)),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "std": round(float(s.std()), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "p75": round(float(s.quantile(0.75)), 2),
    }


# ── 各分析模块 ──────────────────────────────────

def analyze_missing(df):
    """#16: 各字段缺失率统计"""
    total = len(df)
    result = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        result.append({
            "字段": col,
            "缺失数": missing,
            "缺失率": fmt_pct(missing, total),
            "完整率": fmt_pct(total - missing, total),
        })
    return result


def analyze_industry(df):
    """#17: 行业记录数量分布"""
    col = "上市公司行业"
    counts = df[col].value_counts()
    return {
        "行业总数": int(counts.shape[0]),
        "最大值": int(counts.max()),
        "最小值": int(counts.min()),
        "中位数": round(float(counts.median()), 1),
        "均值": round(float(counts.mean()), 1),
        "标准差": round(float(counts.std()), 1),
        "零记录行业数": int((df[col].isna()).sum()),
        "Top10行业": count_distribution(df[col], 10),
        "Bottom10行业": {
            str(k): int(v)
            for k, v in counts.sort_values(ascending=True).head(10).items()
        },
        "全行业分布": {str(k): int(v) for k, v in counts.items()},
    }


def analyze_company(df):
    """#18: 企业记录数量分布"""
    col = "企业名称"
    counts = df[col].value_counts()
    return {
        "企业总数": int(df[col].nunique()),
        "最大值_条/企业": int(counts.max()),
        "最小值_条/企业": int(counts.min()),
        "中位数": round(float(counts.median()), 1),
        "均值": round(float(counts.mean()), 1),
        "标准差": round(float(counts.std()), 1),
        "仅有1条记录的企业数": int((counts == 1).sum()),
        "Top10企业": count_distribution(df[col], 10),
    }


def analyze_city(df):
    """#19: 城市地理分布 Top-20"""
    col = "工作城市"
    total = len(df)
    counts = df[col].value_counts()
    top20 = {}
    for city, cnt in counts.head(20).items():
        top20[str(city)] = {
            "记录数": int(cnt),
            "占比": fmt_pct(cnt, total),
        }
    return {
        "城市总数": int(df[col].nunique()),
        "Top20城市": top20,
    }


def analyze_salary(df):
    """#20: 月薪数值分布（最低月薪 & 最高月薪）"""
    return {
        "最低月薪": numeric_stats(df["最低月薪"]),
        "最高月薪": numeric_stats(df["最高月薪"]),
        "平均薪资_均值": round(
            float(
                (df["最低月薪"].dropna() + df["最高月薪"].dropna()).mean() / 2
            ), 2
        ) if len(df["最低月薪"].dropna()) > 0 else None,
        "月薪区间分布_5k分段": salary_segment_distribution(df),
    }


def salary_segment_distribution(df):
    """将平均月薪按 5000 元分段统计"""
    avg_salary = (df["最低月薪"] + df["最高月薪"]) / 2
    avg_salary = avg_salary.dropna()
    bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 100000, 999999]
    labels = [
        "0-5k", "5k-10k", "10k-15k", "15k-20k", "20k-25k",
        "25k-30k", "30k-40k", "40k-50k", "50k-100k", "100k+"
    ]
    segment_counts = pd.cut(avg_salary, bins=bins, labels=labels, right=False).value_counts()
    return {str(k): int(v) for k, v in segment_counts.items()}


def analyze_education(df):
    """#21: 学历要求类别分布"""
    col = "学历要求"
    counts = df[col].value_counts()
    return {
        "类别总数": int(df[col].nunique()),
        "分布": {str(k): int(v) for k, v in counts.items()},
        "缺失数": int(df[col].isna().sum()),
    }


def analyze_experience(df):
    """#22: 经验要求类别分布"""
    col = "要求经验"
    counts = df[col].value_counts()
    return {
        "类别总数": int(df[col].nunique()),
        "分布": {str(k): int(v) for k, v in counts.items()},
        "缺失数": int(df[col].isna().sum()),
    }


def analyze_category(df):
    """#23: 招聘类别分布"""
    col = "招聘类别"
    counts = df[col].value_counts()
    return {
        "类别总数": int(df[col].nunique()),
        "Top10": count_distribution(df[col], 10),
        "全部分布": {str(k): int(v) for k, v in counts.items()},
    }


def analyze_date(df):
    """#24: 发布日期年份与月份分布"""
    col = "招聘发布日期"
    dates = pd.to_datetime(df[col], errors="coerce")
    year_counts = dates.dt.year.value_counts()
    month_counts = dates.dt.month.value_counts()

    return {
        "时间跨度": f"{dates.min().date()} ~ {dates.max().date()}",
        "年份分布": {str(k): int(v) for k, v in year_counts.items()},
        "月份分布": {f"{int(k)}月": int(v) for k, v in month_counts.sort_index().items()},
        "日期异常数": int(dates.isna().sum()),
    }


# ── 主流程 ──────────────────────────────────────

def print_report(results):
    """#25: 打印格式化的数据探索报告（控制台）"""
    sep = "=" * 70
    sub = "-" * 50

    # 基本概况
    base = results["基本信息"]
    print(sep)
    print("  上市公司招聘数据 —— 探索报告")
    print(sep)
    print(f"  总记录数: {fmt_num(base['总记录数'])}")
    print(f"  字段数量: {base['字段数量']}")
    print(f"  字段列表: {', '.join(base['字段列表'])}")

    # #16 缺失率
    print(f"\n{sep}")
    print("  #16 字段缺失率统计")
    print(sep)
    print(f"{'字段':<16s} {'缺失数':>8s} {'缺失率':>10s} {'完整率':>10s}")
    print(sub)
    for item in results["缺失统计"]:
        print(f"{item['字段']:<16s} {fmt_num(item['缺失数']):>8s} {item['缺失率']:>10s} {item['完整率']:>10s}")

    # #17 行业分布
    ind = results["行业分布"]
    print(f"\n{sep}")
    print("  #17 行业记录分布")
    print(sep)
    print(f"  行业总数: {ind['行业总数']}    "
          f"最大: {fmt_num(ind['最大值'])}   最小: {fmt_num(ind['最小值'])}")
    print(f"  均值: {ind['均值']}   中位数: {ind['中位数']}   标准差: {ind['标准差']}")
    print(f"\n  Top-10 行业:")
    print(f"  {'行业':<30s} {'记录数':>8s}  {'占比':>8s}")
    print(f"  {sub}")
    for name, cnt in ind["Top10行业"].items():
        print(f"  {name:<30s} {fmt_num(cnt):>8s}  {fmt_pct(cnt, base['总记录数']):>8s}")

    # #18 企业分布
    comp = results["企业分布"]
    print(f"\n{sep}")
    print("  #18 企业记录分布")
    print(sep)
    print(f"  企业总数: {fmt_num(comp['企业总数'])}")
    print(f"  最多: {comp['最大值_条/企业']}条/企业   最少: {comp['最小值_条/企业']}条/企业")
    print(f"  均值: {comp['均值']}   中位数: {comp['中位数']}   标准差: {comp['标准差']}")
    print(f"  仅1条记录的企业: {fmt_num(comp['仅有1条记录的企业数'])} " +
          f"({fmt_pct(comp['仅有1条记录的企业数'], comp['企业总数'])})")

    # #19 城市分布
    city = results["城市分布"]
    print(f"\n{sep}")
    print("  #19 城市分布 Top-20")
    print(sep)
    print(f"  城市总数: {city['城市总数']}")
    for city_name, info in city["Top20城市"].items():
        print(f"  {city_name:<16s} {fmt_num(info['记录数']):>8s}  ({info['占比']})")

    # #20 薪资分布
    sal = results["薪资分布"]
    print(f"\n{sep}")
    print("  #20 月薪数值分布")
    print(sep)
    print(f"  {'指标':<12s} {'最低月薪':>12s} {'最高月薪':>12s}")
    print(f"  {sub}")
    for metric in ["count", "mean", "median", "std", "min", "p25", "p75", "max"]:
        metric_cn = {
            "count": "样本数", "mean": "均值", "median": "中位数",
            "std": "标准差", "min": "最小值", "p25": "P25", "p75": "P75", "max": "最大值"
        }
        print(f"  {metric_cn[metric]:<12s} {sal['最低月薪'][metric]:>12,.0f} {sal['最高月薪'][metric]:>12,.0f}")

    print(f"\n  平均月薪均值: {sal['平均薪资_均值']:,.0f}")
    print(f"\n  月薪区间分布 (按平均月薪):")
    for seg, cnt in sal["月薪区间分布_5k分段"].items():
        bar = "█" * int(cnt / max(sal["月薪区间分布_5k分段"].values()) * 40)
        print(f"  {seg:<10s} {fmt_num(cnt):>8s}  {bar}")

    # #21 学历分布
    edu = results["学历分布"]
    print(f"\n{sep}")
    print("  #21 学历要求分布")
    print(sep)
    for edu_name, cnt in edu["分布"].items():
        print(f"  {edu_name:<20s} {fmt_num(cnt):>8s}  ({fmt_pct(cnt, base['总记录数'])})")

    # #22 经验分布
    exp = results["经验分布"]
    print(f"\n{sep}")
    print("  #22 经验要求分布")
    print(sep)
    for exp_name, cnt in exp["分布"].items():
        print(f"  {exp_name:<20s} {fmt_num(cnt):>8s}  ({fmt_pct(cnt, base['总记录数'])})")

    # #23 招聘类别分布
    cat = results["招聘类别分布"]
    print(f"\n{sep}")
    print("  #23 招聘类别分布 Top-10")
    print(sep)
    print(f"  类别总数: {cat['类别总数']}")
    for cat_name, cnt in cat["Top10"].items():
        print(f"  {cat_name:<30s} {fmt_num(cnt):>8s}")

    # #24 日期分布
    date = results["日期分布"]
    print(f"\n{sep}")
    print("  #24 发布日期分布")
    print(sep)
    print(f"  时间跨度: {date['时间跨度']}")
    print(f"\n  年份分布:")
    for yr, cnt in date["年份分布"].items():
        print(f"  {yr} 年:  {fmt_num(cnt)} 条")
    print(f"\n  月份分布:")
    for mo, cnt in date["月份分布"].items():
        bar = "█" * int(cnt / max(date["月份分布"].values()) * 30)
        print(f"  {mo:<8s} {fmt_num(cnt):>8s}  {bar}")

    print(f"\n{sep}")
    print(f"  报告生成完毕")
    print(sep)


def main():
    df = load_data()

    # 收集所有结果
    results = {
        "基本信息": {
            "总记录数": len(df),
            "字段数量": len(df.columns),
            "字段列表": list(df.columns),
        },
        "缺失统计": analyze_missing(df),                          # #16
        "行业分布": analyze_industry(df),                         # #17
        "企业分布": analyze_company(df),                          # #18
        "城市分布": analyze_city(df),                             # #19
        "薪资分布": analyze_salary(df),                           # #20
        "学历分布": analyze_education(df),                        # #21
        "经验分布": analyze_experience(df),                       # #22
        "招聘类别分布": analyze_category(df),                     # #23
        "日期分布": analyze_date(df),                             # #24
    }

    # 序列化：把 NumPy 类型转为 Python 原生类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    report = convert(results)

    # 保存 JSON
    os.makedirs(os.path.dirname(STATS_SUMMARY), exist_ok=True)
    output_path = STATS_SUMMARY
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[JSON 已保存] {output_path}")

    # 打印控制台报告
    print_report(results)


if __name__ == "__main__":
    main()
