"""
数据清洗脚本 —— 覆盖 To Do List 2.3 节 #31 ~ #51

    #31-#36: 文本清洗（URL、广告、不可见字符、元数据、空白、标点）
    #37-#38: 缺失值处理（关键字段删除 / 次要字段标注）
    #39-#40: 月薪异常值处理（逻辑修正 + 范围审查）
    #41-#47: 字段标准化（学历、经验、城市、行业）
    #48     : 清洗后质量验证
    #49     : 清洗前后对比报告
    #50     : 保存清洗后数据集
    #51     : 保存 HTML 质量报告

用法:
    conda activate my_project
    python src/data_cleaner.py
"""

import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, OUTPUT_DIR

# ── 配置常量 ────────────────────────────────────

# 关键字段：缺失则删除整行
CRITICAL_FIELDS = ["企业名称", "职位描述", "上市公司行业"]

# 次要字段：缺失则填充默认值
SECONDARY_DEFAULTS = {
    "学历要求": "学历不限",
    "要求经验": "经验不限",
    "最低月薪": -1,   # -1 表示"面议"
    "最高月薪": -1,
}

# 月薪合理范围
SALARY_MIN_REASONABLE = 1000
SALARY_MAX_REASONABLE = 200000

# ── 脏数据正则（按来源模式分组）──

# ① 马克数据系品牌词（所有变体）
RE_MARKE = re.compile(
    r'马\s*克\s*数\s*据\s*网|马\s*克\s*数\s*据|马\s*克\s*团\s*队|macrodatas?|macrodata',
    re.IGNORECASE
)

# ② URL（含 CSV 引号泄漏）
RE_URL = re.compile(
    r'https?://[^\s，,。\n）\)]*|'
    r'www\.[^\s，,。\n）\)]*|'
    r'\b[a-zA-Z0-9._%+-]+\.(com|cn|org|net|gov|edu|io)[^\s，,。\n）\)]*',
)

# ③ "该数据由<马克数据网>整理" / "数据由马克数据整理"
RE_DATA_BY = re.compile(
    r'(该)?数据由[^。，,\n]{0,30}?整理\s*[。.]?',
)

# ④ 来源标记: "（来源 马克数据网）" / "来源：百度搜索马克数据网" / "来源：www.xxx"
RE_SOURCE = re.compile(
    r'[（(]\s*来源\s*[：:\s]+\s*[^）)]*[）)]|'
    r'来源\s*[：:\s]+\s*(百度|搜索|www\.)?\s*马\s*克\s*[^，。,\n]{0,20}|'
    r'来源\s*[：:\s]+\s*www\.[^\s，。,\n]{0,30}',
)

# ⑤ "更多数据：搜索马克数据网" / "搜索马克数据网"
RE_CROSS_REF = re.compile(
    r'更多数据\s*[：:]\s*搜索\s*马\s*克\s*[^，。,\n]{0,20}|'
    r'搜索\s*马\s*克\s*数据\s*网[^，。,\n]{0,10}',
)

# ⑥ 微信公众号/关注引流文案
RE_WECHAT = re.compile(
    r'(关注|微信|公众号|扫描|扫码|长按|识别).{0,20}(公众号|二维码|微信|好友|关注)',
    re.IGNORECASE
)

# 学历映射表
EDUCATION_MAP = {
    "博士": ["博士", "博士研究生", "博士学位", "博士及以上", "博士研究生及以上"],
    "硕士": ["硕士", "硕士研究生", "硕士学位", "硕士及以上", "硕士研究生及以上"],
    "本科": ["本科", "学士", "大学本科", "本科学历", "本科及以上", "本科以上", "本科或以上",
             "大学本科及以上", "全日制本科", "统招本科", "本科以上学历"],
    "大专": ["大专", "专科", "大学专科", "大专及以上", "专科及以上", "大专以上",
             "大专学历", "大专或以上"],
    "中技/中专": ["中技", "中专", "中技/中专", "技校", "职高", "中职"],
    "高中": ["高中", "高中学历", "高中及以上"],
    "初中及以下": ["初中", "初中及以下", "初中及以上"],
    "学历不限": ["不限", "学历不限", "无学历要求", "无", ""],
}

# 经验映射表
EXPERIENCE_MAP = {
    "经验不限": ["经验不限", "不限", "无经验", "无需经验", "无工作经验", "无需",
               "应届生", "应届毕业生", "在校生/应届生", "应届生/实习生"],
    "1年以下": ["1年以下", "1年以内"],
    "1年及以上": ["1年", "1年及以上", "1年以上", "一年以上", "一年及以上"],
    "2年及以上": ["2年", "2年及以上", "2年以上", "两年以上", "两年及以上"],
    "3年及以上": ["3年", "3年及以上", "3年以上", "三年以上", "三年及以上",
                 "1-3年", "1-3年经验"],
    "5年及以上": ["5年", "5年及以上", "5年以上", "五年以上", "五年及以上",
                 "3-5年", "3-5年经验", "5-7年"],
    "8年及以上": ["8年", "8年及以上", "8年以上", "八年以上",
                 "5-10年", "5-10年经验"],
    "10年及以上": ["10年", "10年及以上", "10年以上", "十年以上"],
}


# ── #31-#36: 文本清洗 ──────────────────────────

def clean_text(text):
    """对单段文本执行全部清洗规则，返回清洗后的文本"""
    if not isinstance(text, str) or not text.strip():
        return ""

    t = text

    # ── 第一轮：去除完整句式的脏数据（可能跨多个字符）──

    # #34a: "该数据由<马克数据网>整理" / "数据由马克数据整理"
    t = RE_DATA_BY.sub("", t)

    # #34b: 来源标记 "（来源 马克数据网）" / "来源：百度搜索..." / "来源：www..."
    t = RE_SOURCE.sub("", t)

    # #34c: "更多数据：搜索马克数据网" / "搜索马克数据网"
    t = RE_CROSS_REF.sub("", t)

    # ── 第二轮：去除品牌词和 URL ──

    # #31: URL
    t = RE_URL.sub(" ", t)

    # #32a: 马克数据系品牌词
    t = RE_MARKE.sub("", t)

    # #32b: 微信公众号引流文案
    t = RE_WECHAT.sub("", t)

    # ── 第三轮：字符级清理 ──

    # #33: 不可见字符（\x00-\x08, \x0b-\x0c, \x0e-\x1f, \x7f-\x9f）
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', t)
    t = t.replace('\xa0', ' ')
    t = t.replace('　', ' ')
    t = t.replace('\r\n', '\n')
    t = t.replace('\r', '\n')

    # #35: 空白压缩
    t = re.sub(r'\t+', ' ', t)
    t = re.sub(r' {2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r' *\n *', '\n', t)

    # #35b: 清洗产生的残留：孤立的 ".数据由" "来源："片段 再次扫描
    #      （第二轮品牌词去除后可能留下残余标点）
    t = re.sub(r'[。.]\s*数据由\s*[^，,.\n]{0,20}整理', '', t)
    t = re.sub(r'[。.]\s*该数据由\s*[^，,.\n]{0,30}整理', '', t)
    t = re.sub(r'[。.]\s*来源\s*[：:\s]+\s*[^，,.\n]{0,20}', '', t)

    t = t.strip()

    return t


def clean_dataframe_text(df):
    """对 DataFrame 中所有文本列执行 clean_text"""
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        before_empty = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
        df[col] = df[col].apply(clean_text)
        after_empty = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
        if before_empty != after_empty:
            print(f"  [文本清洗] '{col}': 清理出 {after_empty - before_empty} 条空记录")
    return df


# ── #37-#38: 缺失值处理 ────────────────────────

def handle_missing_values(df):
    """处理缺失值"""
    n_before = len(df)

    # #37: 关键字段缺失 → 删除整行
    for col in CRITICAL_FIELDS:
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            n_drop = mask.sum()
            if n_drop > 0:
                df = df[~mask]
                print(f"  #37 删除关键字段='{col}'缺失: {n_drop} 行")

    n_after_crit = len(df)

    # #38: 次要字段缺失 → 填充默认值
    for col, default in SECONDARY_DEFAULTS.items():
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            n_fill = mask.sum()
            if n_fill > 0:
                df.loc[mask, col] = default
                print(f"  #38 填充字段='{col}': {n_fill} 行 → '{default}'")

    print(f"  [缺失值] 输入 {n_before} 行 → 输出 {len(df)} 行 "
          f"(删除 {n_before - n_after_crit} 行, 填充 {n_after_crit - len(df)} 行)")

    return df


# ── #39-#40: 月薪异常值 ────────────────────────

def clean_salary(df):
    """月薪异常值处理"""
    issues = {"最低>最高": 0, "低于合理下限": 0, "高于合理上限": 0}

    for col in ["最低月薪", "最高月薪"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    min_sal = df["最低月薪"]
    max_sal = df["最高月薪"]

    valid_min = min_sal.notna()
    valid_max = max_sal.notna()
    valid_both = valid_min & valid_max

    # #39: 最低 > 最高 → 交换
    swap_mask = valid_both & (min_sal > max_sal)
    issues["最低>最高"] = int(swap_mask.sum())
    if swap_mask.sum() > 0:
        df.loc[swap_mask, ["最低月薪", "最高月薪"]] = (
            df.loc[swap_mask, ["最高月薪", "最低月薪"]].values
        )
        print(f"  #39 最低>最高 → 已交换: {issues['最低>最高']} 行")

    # #40: 合理范围检查（标注，不删除）
    for col, label in [("最低月薪", "最低"), ("最高月薪", "最高")]:
        too_low = (df[col].notna()) & (df[col] > 0) & (df[col] < SALARY_MIN_REASONABLE)
        too_high = (df[col].notna()) & (df[col] > SALARY_MAX_REASONABLE)
        issues[f"{label}低于合理下限"] = int(too_low.sum())
        issues[f"{label}高于合理上限"] = int(too_high.sum())
        if too_low.sum() > 0:
            print(f"  #40 {label}月薪 < {SALARY_MIN_REASONABLE}: {too_low.sum()} 行 (保留但标记)")
        if too_high.sum() > 0:
            print(f"  #40 {label}月薪 > {SALARY_MAX_REASONABLE}: {too_high.sum()} 行 (保留但标记)")

    return df, issues


# ── #41-#42: 学历标准化 ────────────────────────

def normalize_education(df):
    """学历要求标准化"""
    col = "学历要求"
    if col not in df.columns:
        return df

    def _map(val):
        if not isinstance(val, str) or not val.strip():
            return "学历不限"
        v = val.strip()
        # 先检查是否已标准化
        if v in EDUCATION_MAP:
            return v
        # 子串匹配
        for std, aliases in EDUCATION_MAP.items():
            for alias in aliases:
                if alias in v:
                    return std
        # 包含"本科"/"硕士"等关键字的模糊匹配
        for std in ["博士", "硕士", "本科", "大专", "高中", "初中"]:
            if std in v:
                for full_std in EDUCATION_MAP:
                    if std in full_std:
                        return full_std
        # 长文本串列 → 标记为未知
        if len(v) > 10:
            return "学历不限"
        return v

    before = df[col].value_counts().to_dict()
    df[col] = df[col].apply(_map)
    after = df[col].value_counts().to_dict()

    n_changed = sum(1 for k, v in before.items() if str(k) not in after or after.get(str(k), 0) != v)
    print(f"  #41-#42 学历标准化: {df[col].nunique()} 类 (变更约 {n_changed} 类)")

    return df


# ── #43-#44: 经验标准化 ────────────────────────

def normalize_experience(df):
    """经验要求标准化"""
    col = "要求经验"
    if col not in df.columns:
        return df

    def _map(val):
        if not isinstance(val, str) or not val.strip():
            return "经验不限"
        v = val.strip()
        if v in EXPERIENCE_MAP:
            return v
        # 子串匹配
        for std, aliases in EXPERIENCE_MAP.items():
            for alias in aliases:
                if alias in v:
                    return std
        # 模糊：包含"年"
        if "年" in v or "经验" in v:
            # 尝试提取年数
            years_match = re.search(r'(\d+)\s*年', v)
            if years_match:
                years = int(years_match.group(1))
                if years <= 1:
                    return "1年及以上"
                elif years <= 2:
                    return "2年及以上"
                elif years <= 3:
                    return "3年及以上"
                elif years <= 5:
                    return "5年及以上"
                elif years <= 8:
                    return "8年及以上"
                else:
                    return "10年及以上"
            return "经验不限"
        # 长文本串列
        if len(v) > 15:
            return "经验不限"
        return v

    before = df[col].value_counts().to_dict()
    df[col] = df[col].apply(_map)
    after = df[col].value_counts().to_dict()

    print(f"  #43-#44 经验标准化: {df[col].nunique()} 类")

    return df


# ── #45-#46: 城市标准化 ────────────────────────

def normalize_city(df):
    """城市名称标准化"""
    col = "工作城市"
    if col not in df.columns:
        return df

    def _map(val):
        if not isinstance(val, str) or not val.strip():
            return "全国"
        v = val.strip()
        # 去除后缀
        v = re.sub(r'(市|地区|自治州|特别行政区)$', '', v)
        # 处理省级名称
        province_map = {
            "福建省": "福建", "广东省": "广东", "浙江省": "浙江",
            "江苏省": "江苏", "四川省": "四川", "湖北省": "湖北",
            "湖南省": "湖南", "河南省": "河南", "河北省": "河北",
            "山东省": "山东", "山西省": "山西", "陕西省": "陕西",
            "云南省": "云南", "贵州省": "贵州", "甘肃省": "甘肃",
            "辽宁省": "辽宁", "吉林省": "吉林", "黑龙江省": "黑龙江",
            "安徽省": "安徽", "江西省": "江西", "海南省": "海南",
            "青海省": "青海", "台湾省": "台湾",
        }
        v = province_map.get(v, v)
        # #46: 空/其他 → 全国
        if not v or v in ["其他", "其它", "不限", "全国"]:
            return "全国"
        return v

    before_nunique = df[col].nunique()
    df[col] = df[col].apply(_map)
    after_nunique = df[col].nunique()

    print(f"  #45-#46 城市标准化: {before_nunique} → {after_nunique} 类")

    return df


# ── #47: 行业名称检查 ──────────────────────────

def check_industry(df):
    """行业分类名称一致性检查"""
    col = "上市公司行业"
    if col not in df.columns:
        return df

    values = df[col].dropna().unique()
    # 检查多余空格
    for v in values:
        stripped = v.strip()
        if stripped != v:
            df.loc[df[col] == v, col] = stripped

    unique = df[col].dropna().unique()
    print(f"  #47 行业检查: {len(unique)} 类, 无冗余空格")

    return df


# ── #48: 清洗后质量验证 ────────────────────────

def validate_cleaned(df, salary_issues):
    """逐字段检查是否有残留脏数据"""
    total = len(df)
    report = {}
    failures = 0

    for col in df.columns:
        col_report = {}

        # 缺失率（关键字段不应有缺失）
        missing = int(df[col].isna().sum())
        col_report["缺失数"] = missing
        col_report["缺失率"] = f"{missing / total * 100:.2f}%"

        if col in CRITICAL_FIELDS and missing > 0:
            col_report["状态"] = "⚠️ 关键字段仍有缺失"
            failures += 1
        elif missing == 0:
            col_report["状态"] = "✓"
        else:
            col_report["状态"] = "已填充默认值" if col in SECONDARY_DEFAULTS else "—"

        # 脏数据残留检测
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).str.cat(sep=" ")
            n_url = len(RE_URL.findall(sample))
            n_marke = len(RE_MARKE.findall(sample))
            n_data_by = len(RE_DATA_BY.findall(sample))
            n_source = len(RE_SOURCE.findall(sample))
            n_cross = len(RE_CROSS_REF.findall(sample))
            n_wechat = len(RE_WECHAT.findall(sample))
            total_dirty = n_url + n_marke + n_data_by + n_source + n_cross + n_wechat
            col_report["残留URL"] = n_url
            col_report["残留品牌词"] = n_marke
            col_report["残留来源标记"] = n_source + n_data_by + n_cross
            col_report["残留引流文案"] = n_wechat
            if total_dirty > 0:
                col_report["状态"] = f"存在 {total_dirty} 处残留脏数据"
                failures += 1

        report[col] = col_report

    # 薪资异常
    report["_薪资异常_"] = salary_issues

    is_clean = (failures == 0)
    print(f"\n  #48 质量验证: {'全部通过 ✓' if is_clean else f'{failures} 项异常 ⚠️'}")

    return report, is_clean


# ── #49: 对比报告 ──────────────────────────────

def generate_comparison_report(before_df, after_df, before_shape, after_shape):
    """清洗前后对比"""
    comp = {
        "行数变化": f"{before_shape[0]:,} → {after_shape[0]:,} "
                    f"(删除 {before_shape[0] - after_shape[0]:,} 行)",
        "字段数变化": f"{before_shape[1]} → {after_shape[1]}",
        "各字段对比": {},
    }

    common_cols = set(before_df.columns) & set(after_df.columns)
    for col in common_cols:
        before_null = int(before_df[col].isna().sum())
        after_null = int(after_df[col].isna().sum())
        before_unique = int(before_df[col].nunique())
        after_unique = int(after_df[col].nunique())
        comp["各字段对比"][col] = {
            "缺失减少": before_null - after_null,
            "唯一值减少": before_unique - after_unique,
        }

    print(f"\n  #49 对比报告: {comp['行数变化']}")
    return comp


# ── #50: 保存数据 ──────────────────────────────

def save_cleaned_data(df, report):
    """保存清洗后的数据和完整清洗报告"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "上市公司招聘数据_cleaned.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  #50 [保存] {csv_path} ({len(df):,} 行)")

    # JSON 报告
    report_path = os.path.join(OUTPUT_DIR, "cleaner_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [报告] {report_path}")


# ── #51: HTML 质量报告 ────────────────────────

def save_html_report(report, df):
    """生成 HTML 格式的数据质量报告预览页"""
    html_path = os.path.join(OUTPUT_DIR, "数据质量报告.html")

    def row_status_icon(status):
        if "✓" in str(status): return "🟢"
        if "⚠" in str(status): return "🟡"
        return "🔴"

    validation_rows = ""
    for col, info in report["验证报告"].items():
        if col.startswith("_"):
            continue
        icon = row_status_icon(info.get("状态", ""))
        validation_rows += (
            f"<tr><td>{icon}</td><td>{col}</td>"
            f"<td>{info.get('缺失数', '-')}</td>"
            f"<td>{info.get('缺失率', '-')}</td>"
            f"<td>{info.get('残留URL', '-')}</td>"
            f"<td>{info.get('残留品牌词', '-')}</td>"
            f"<td>{info.get('残留来源标记', '-')}</td>"
            f"<td>{info.get('状态', '-')}</td></tr>\n"
        )

    salary_issues = report.get("验证报告", {}).get("_薪资异常_", {})
    salary_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in salary_issues.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>数据质量报告</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }}
  h1 {{ border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }}
  h2 {{ margin-top: 32px; color: #1a73e8; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f5f5f5; }}
  .ok {{ color: #34a853; }}
  .warn {{ color: #ea4335; }}
</style>
</head>
<body>
<h1>上市公司招聘数据 —— 质量报告</h1>
<p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<h2>清洗对比</h2>
<table>
  <tr><th>指标</th><th>值</th></tr>
  <tr><td>行数变化</td><td>{report['对比报告'].get('行数变化', '-')}</td></tr>
  <tr><td>字段数变化</td><td>{report['对比报告'].get('字段数变化', '-')}</td></tr>
</table>

<h2>字段级验证</h2>
<table>
  <tr><th></th><th>字段</th><th>缺失数</th><th>缺失率</th><th>残留URL</th><th>残留品牌词</th><th>残留来源标记</th><th>状态</th></tr>
  {validation_rows}
</table>

<h2>薪资异常统计</h2>
<table>
  <tr><th>异常类型</th><th>数量</th></tr>
  {salary_rows}
</table>

<h2>数据样本（前5条）</h2>
<table>
  <tr>{''.join(f'<th>{c}</th>' for c in list(df.columns)[:8])}</tr>
  {''.join(
      f'<tr>{"".join(f"<td>{str(v)[:60]}</td>" for v in row[:8])}</tr>'
      for _, row in df.head(5).iterrows()
  )}
</table>

<p style="margin-top:24px;color:#888;font-size:12px;">
  清洗流程: URL/广告去除 → 空白字符清理 → 缺失值处理 → 薪资异常修正 →
  学历/经验/城市标准化 → 行业名检查
</p>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  #51 [HTML] {html_path}")


# ── 主流程 ──────────────────────────────────────

def main():
    print("=" * 60)
    print("  上市公司招聘数据 —— 数据清洗 (#31 ~ #51)")
    print("=" * 60)

    # 加载预处理后的数据
    proc_csv = os.path.join(OUTPUT_DIR, "上市公司招聘数据_processed.csv")
    if not os.path.exists(proc_csv):
        print("  [提示] 预处理文件不存在，从原始数据开始清洗")
        proc_csv = os.path.join(DATA_DIR, "上市公司招聘数据2026.csv")

    df = pd.read_csv(proc_csv, encoding="utf-8-sig", low_memory=False)
    before_df = df.copy()
    before_shape = df.shape

    print(f"\n  [加载] {proc_csv}")
    print(f"  {before_shape[0]:,} 行, {before_shape[1]} 字段")

    # ── #31-#36: 文本清洗 ──
    print(f"\n{'─' * 40}")
    print("  #31-#36 文本清洗")
    df = clean_dataframe_text(df)

    # ── #37-#38: 缺失值 ──
    print(f"\n{'─' * 40}")
    print("  #37-#38 缺失值处理")
    df = handle_missing_values(df)
    df = df.reset_index(drop=True)

    # ── #39-#40: 月薪异常 ──
    print(f"\n{'─' * 40}")
    print("  #39-#40 月薪异常值")
    df, salary_issues = clean_salary(df)

    # ── #41-#42: 学历标准化 ──
    print(f"\n{'─' * 40}")
    print("  #41-#42 学历标准化")
    df = normalize_education(df)

    # ── #43-#44: 经验标准化 ──
    print(f"\n{'─' * 40}")
    print("  #43-#44 经验标准化")
    df = normalize_experience(df)

    # ── #45-#46: 城市标准化 ──
    print(f"\n{'─' * 40}")
    print("  #45-#46 城市标准化")
    df = normalize_city(df)

    # ── #47: 行业检查 ──
    print(f"\n{'─' * 40}")
    print("  #47 行业名称检查")
    df = check_industry(df)

    # ── #48: 质量验证 ──
    print(f"\n{'─' * 40}")
    validation_report, is_clean = validate_cleaned(df, salary_issues)

    # ── #49: 对比报告 ──
    comparison = generate_comparison_report(before_df, df, before_shape, df.shape)

    # 汇总报告
    report = {
        "清洗时间": datetime.now().isoformat(),
        "对比报告": comparison,
        "验证报告": validation_report,
    }

    # ── #50: 保存 ──
    print(f"\n{'─' * 40}")
    save_cleaned_data(df, report)

    # ── #51: HTML ──
    save_html_report(report, df)

    print(f"\n{'=' * 60}")
    print(f"  数据清洗完成 ✓")
    print(f"  最终数据: {len(df):,} 行, {len(df.columns)} 字段")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
