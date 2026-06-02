"""
数据预处理脚本 —— 覆盖 To Do List 2.2 节 #26 ~ #30
    26. 字段筛选（23字段 → 保留14字段）
    27. 招聘结束日期生成（发布日期 + 60天）
    28. 日期逻辑验证（结束日期 > 发布日期）
    29. 日期格式统一（YYYY/M/D → YYYY-MM-DD）
    30. 字段重命名

用法:
    conda activate my_project
    python src/preprocess.py

输出:
    output/上市公司招聘数据_processed.csv   清洗后的全量数据
    output/preprocess_report.json          预处理过程记录
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DATA_DIR, PROCESSED_CSV, PREPROCESS_REPORT, DIR_2_PREPROC

# ── 配置 ────────────────────────────────────────

# 保留的目标字段（原始CSV → 标准化名称）
TARGET_FIELDS = [
    "企业名称",
    "股票简称",
    "上市公司行业",
    "工作城市",
    "最低月薪",
    "最高月薪",
    "职位描述",
    "学历要求",
    "要求经验",
    "招聘类别",
    "招聘发布日期",
]

# 原始 CSV 的可能字段名变体（兼容不同版本）
FIELD_ALIASES = {
    "企业名称": ["企业名称", "公司名称"],
    "股票简称": ["股票简称"],
    "上市公司行业": ["上市公司行业", "行业"],
    "工作城市": ["工作城市", "城市"],
    "最低月薪": ["最低月薪", "月薪下限"],
    "最高月薪": ["最高月薪", "月薪上限"],
    "职位描述": ["职位描述", "岗位描述"],
    "学历要求": ["学历要求", "学历"],
    "要求经验": ["要求经验", "工作经验"],
    "招聘类别": ["招聘类别", "岗位类别"],
    "招聘发布日期": ["招聘发布日期", "发布日期"],
}

# 输出字段名标准化
OUTPUT_FIELDS = [
    "企业名称",
    "股票简称",
    "上市公司行业",
    "招聘岗位",
    "工作城市",
    "最低月薪",
    "最高月薪",
    "职位描述",
    "学历要求",
    "要求经验",
    "招聘类别",
    "招聘发布日期",
    "招聘结束日期",
]

DATE_OFFSET_DAYS = 60  # 结束日期 = 发布日期 + 60天


# ── 数据加载 ────────────────────────────────────

def load_raw_data():
    """加载原始 CSV，自动识别编码"""
    csv_path = os.path.join(DATA_DIR, "上市公司招聘数据2026.csv")
    encodings = ["gbk", "gb2312", "gb18030", "utf-8"]

    for enc in encodings:
        try:
            with open(csv_path, "rb") as f:
                f.read(50000).decode(enc)
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            print(f"[加载] 编码={enc}, {len(df):,} 行, {len(df.columns)} 字段")
            return df
        except (UnicodeDecodeError, LookupError):
            continue

    raise RuntimeError("无法识别 CSV 编码")


# ── #26: 字段筛选 ───────────────────────────────

def select_fields(df):
    """从原始字段中筛选出目标字段，自动匹配别名"""
    available = {}
    selected = {}

    for target, aliases in FIELD_ALIASES.items():
        found = False
        for alias in aliases:
            if alias in df.columns:
                available[target] = alias
                selected[target] = df[alias].copy()
                found = True
                break
        if not found:
            print(f"  [警告] 未找到字段: {target} (尝试了: {aliases})")

    # 同时保留 编号 / 招聘岗位 / 招聘结束日期（如果存在）
    for extra in ["编号", "招聘岗位", "招聘结束日期"]:
        if extra in df.columns and extra not in selected:
            selected[extra] = df[extra].copy()

    result = pd.DataFrame(selected)
    print(f"  #26 字段筛选: {len(df.columns)} → {len(result.columns)} 字段")
    print(f"     保留字段: {list(result.columns)}")

    return result


# ── #27: 招聘结束日期生成 ────────────────────────

def generate_end_date(df):
    """若招聘结束日期缺失，则用发布日期 + 60天 自动生成"""
    col_pub = "招聘发布日期"
    col_end = "招聘结束日期"

    if col_end not in df.columns:
        df[col_end] = pd.NaT

    # 先统一解析发布日期
    pub_dates = parse_dates(df[col_pub])

    # 只对结束日期为空的行生成
    end_dates = pd.to_datetime(df[col_end], errors="coerce")
    need_gen = end_dates.isna() & pub_dates.notna()
    n_gen = need_gen.sum()

    if n_gen > 0:
        df.loc[need_gen, col_end] = (
            pub_dates[need_gen] + timedelta(days=DATE_OFFSET_DAYS)
        ).dt.strftime("%Y-%m-%d")
        print(f"  #27 结束日期生成: {n_gen:,} 条 (发布日期 + {DATE_OFFSET_DAYS}天)")
    else:
        print(f"  #27 结束日期: 无需生成（已全部存在）")

    return df


# ── #28: 日期逻辑验证 ───────────────────────────

def validate_dates(df):
    """验证并报告日期异常"""
    pub = pd.to_datetime(df["招聘发布日期"], errors="coerce")
    end = pd.to_datetime(df["招聘结束日期"], errors="coerce")

    issues = {
        "发布日期解析失败": int(pub.isna().sum()),
        "结束日期解析失败": int(end.isna().sum()),
        "结束日期 ≤ 发布日期": int(((end <= pub) & pub.notna() & end.notna()).sum()),
        "结束日期 - 发布日期 < 7天": int(((end - pub).dt.days < 7).sum()),
        "结束日期 - 发布日期 > 365天": int(((end - pub).dt.days > 365).sum()),
    }

    is_ok = (
        issues["发布日期解析失败"] == 0
        and issues["结束日期解析失败"] == 0
        and issues["结束日期 ≤ 发布日期"] == 0
    )

    print(f"  #28 日期验证: {'全部通过 ✓' if is_ok else '存在问题 ⚠️'}")
    for k, v in issues.items():
        if v > 0:
            print(f"     {k}: {v:,} 条")

    return issues


# ── #29: 日期格式统一 ───────────────────────────

def parse_dates(series):
    """尝试多种格式解析日期，统一返回 datetime"""
    # 先尝试 pd.to_datetime 默认解析
    result = pd.to_datetime(series, errors="coerce")
    if result.notna().all():
        return result

    # 如果存在 NaT，尝试其他格式逐一修复
    still_na = result.isna()
    if still_na.any():
        for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                fixed = pd.to_datetime(series[still_na], format=fmt, errors="coerce")
                result.loc[fixed.notna()] = fixed[fixed.notna()]
                still_na = result.isna()
                if not still_na.any():
                    break
            except Exception:
                continue

    return result


def unify_date_format(df):
    """将日期列统一为 YYYY-MM-DD 字符串格式"""
    col_pub = "招聘发布日期"
    col_end = "招聘结束日期"

    before_pub = str(df[col_pub].iloc[0]) if len(df) > 0 else ""
    before_end = str(df[col_end].iloc[0]) if len(df) > 0 else ""

    pub_dates = parse_dates(df[col_pub])
    end_dates = parse_dates(df[col_end])

    df[col_pub] = pub_dates.dt.strftime("%Y-%m-%d")
    df[col_end] = end_dates.dt.strftime("%Y-%m-%d")

    after_pub = str(df[col_pub].iloc[0]) if len(df) > 0 else ""
    after_end = str(df[col_end].iloc[0]) if len(df) > 0 else ""

    print(f"  #29 日期格式统一: '{before_pub}' → '{after_pub}', "
          f"'{before_end}' → '{after_end}'")

    return df


# ── #30: 字段重命名 ─────────────────────────────

def normalize_columns(df):
    """确保输出字段名统一，按 OUTPUT_FIELDS 顺序排列"""
    # 如果存在 编号 字段，放在最前面
    output_order = list(OUTPUT_FIELDS)
    if "编号" in df.columns:
        output_order = ["编号"] + output_order

    # 只保留实际存在的字段
    actual_order = [c for c in output_order if c in df.columns]
    df = df[actual_order]
    print(f"  #30 字段规范化: {list(df.columns)}")
    return df


# ── 辅助：小样本展示 ────────────────────────────

def sample_preview(df, n=3):
    """取前 N 行展示"""
    print(f"\n  预处理后前 {n} 行预览:")
    for _, row in df.head(n).iterrows():
        print(f"  {row.get('企业名称', '?')} | "
              f"{row.get('招聘岗位', '?')} | "
              f"¥{row.get('最低月薪', '?')}-{row.get('最高月薪', '?')} | "
              f"{row.get('招聘发布日期', '?')} → {row.get('招聘结束日期', '?')}")


# ── 主流程 ──────────────────────────────────────

def main():
    print("=" * 60)
    print("  上市公司招聘数据 —— 预处理 (#26 ~ #30)")
    print("=" * 60)

    # 加载
    df = load_raw_data()

    # #26
    df = select_fields(df)

    # #29 先统一日期格式（避免 #27 生成时格式混乱）
    df = unify_date_format(df)

    # #27
    df = generate_end_date(df)

    # #28
    date_issues = validate_dates(df)

    # #30
    df = normalize_columns(df)

    # 保存
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    out_csv = PROCESSED_CSV
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  [保存] {out_csv}")
    print(f"  [编码] UTF-8 with BOM (Excel 直接打开)")

    # 保存预处理报告
    report = {
        "输入文件": "上市公司招聘数据2026.csv",
        "输出文件": "上市公司招聘数据_processed.csv",
        "输入行数": len(df),
        "输出行数": len(df),
        "输出字段": list(df.columns),
        "日期偏移天数": DATE_OFFSET_DAYS,
        "日期验证": date_issues,
    }
    report_path = PREPROCESS_REPORT
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [报告] {report_path}")

    # 预览
    sample_preview(df)

    print(f"\n{'=' * 60}")
    print(f"  预处理完成 ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
