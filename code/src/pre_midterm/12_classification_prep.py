"""
分类任务准备 —— 覆盖 To Do List #117 ~ #120

    #117: 26 类行业标签定义
    #118: 各类别定义说明 + 典型关键词
    #119: 从 5k 数据中均匀抽取 400 条
    #120: 生成人工标注模板文件

用法:
    conda activate my_project
    python src/12_classification_prep.py

输出:
    output/12_classify/category_definitions.json   26类定义+关键词
    output/12_classify/sample_400.csv             400条抽样数据
    output/12_classify/annotation_template.txt    人工标注模板
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEANED_CSV, DIR_12_CLASSIFY

CAT_DEF_JSON = os.path.join(DIR_12_CLASSIFY, "category_definitions.json")
SAMPLE_400_CSV = os.path.join(DIR_12_CLASSIFY, "sample_400.csv")
ANNO_TEMPLATE = os.path.join(DIR_12_CLASSIFY, "annotation_template.txt")

# ── #117-#118: 26 类行业定义 + 关键词 ──

CATEGORIES = {
    "互联网/软硬件技术": {
        "定义": "互联网产品研发、AI算法、软件开发、大数据/云计算、电子/半导体/通信硬件、嵌入式、产品经理等技术与产品类岗位",
        "合并自": "互联网/AI + 电子/电气/通信 + 产品",
        "关键词": ["软件", "开发", "Java", "Python", "AI", "算法", "芯片", "嵌入式", "PCB", "硬件",
                  "产品经理", "前端", "后端", "架构", "通信", "电子", "半导体", "C++"]
    },
    "销售/市场/客服": {
        "定义": "渠道销售、大客户经理、品牌营销、市场推广、广告公关、电商/用户运营、客服售后等市场与客户类岗位",
        "合并自": "销售 + 市场/公关/广告 + 客服/运营",
        "关键词": ["销售", "客户", "市场", "营销", "品牌", "广告", "客服", "售后", "电商", "运营",
                  "推广", "渠道", "代理商", "BD"]
    },
    "人力/财务/行政": {
        "定义": "HR招聘/培训/薪酬绩效、财务核算/审计/税务、法务合规、行政管理、秘书等企业职能支撑岗位",
        "合并自": "人力/行政/法务 + 财务/审计/税务",
        "关键词": ["人力资源", "HR", "财务", "会计", "审计", "税务", "行政", "法务", "秘书",
                  "招聘", "薪酬", "绩效", "出纳", "合规"]
    },
    "生产制造/汽车": {
        "定义": "车间生产管理、设备操作维护、工艺技术/质量检测、精益生产、新能源汽车三电、汽车零部件等制造与汽车岗位",
        "合并自": "生产制造 + 汽车",
        "关键词": ["生产", "制造", "车间", "工艺", "设备", "质量", "QC", "质检", "普工",
                  "汽车", "电池", "新能源", "电芯", "零部件", "整车"]
    },
    "金融": {
        "定义": "银行、证券、保险、基金、信托、信贷风控、金融科技等金融行业专业岗位",
        "合并自": "金融",
        "关键词": ["金融", "银行", "信贷", "风控", "投资", "证券", "保险", "基金", "理财", "资管"]
    },
    "房地产/建筑": {
        "定义": "房地产开发、建筑工程施工、工程造价/招投标、室内装修、物业管理等地产建筑类岗位",
        "合并自": "房地产/建筑",
        "关键词": ["房地产", "建筑", "工程", "施工", "造价", "物业", "装修", "监理", "土木"]
    },
    "物流/贸易/采购": {
        "定义": "供应链物流、仓储管理、快递配送、货运司机、外贸进出口、采购跟单等流通贸易类岗位",
        "合并自": "物流/仓储/司机 + 采购/贸易",
        "关键词": ["物流", "仓储", "司机", "配送", "采购", "供应商", "外贸", "进出口", "供应链", "货运"]
    },
    "医疗/能源/环保/农业": {
        "定义": "医药研发、医疗器械、光伏风电储能、电力水务环保、现代农业化工等资源与生命科学岗位",
        "合并自": "医疗健康 + 能源/环保/农业",
        "关键词": ["医药", "制药", "医疗器械", "光伏", "风电", "储能", "环保", "水务", "农业", "化工"]
    },
    "文化传媒/教育/生活服务": {
        "定义": "短视频/直播/影视、教育培训、平面/UI/工业设计、零售门店、酒店旅游、餐饮、项目管理、咨询翻译法律、高级管理等综合类岗位",
        "合并自": "直播/影视/传媒 + 教育培训 + 设计 + 零售/生活服务 + 酒店/旅游 + 餐饮 + 项目管理 + 咨询/翻译/法律 + 高级管理 + 其他",
        "关键词": ["直播", "培训", "教育", "设计", "酒店", "餐饮", "零售", "项目管理",
                  "咨询", "翻译", "总监", "总经理", "旅游", "短视频"]
    },
}

# ── #119: 抽样 400 条 ──

def sample_400(df):
    print("[#119] 从 5k 数据中均匀抽取 400 条...")
    n_total = len(df)

    # 使用原始"上市公司行业"字段作为分层依据（81类 → 均匀分布）
    if "上市公司行业" in df.columns:
        # 按行业分组抽样
        groups = df.groupby("上市公司行业")
        samples = []
        target_per_group = max(400 // len(groups), 1)
        remaining = 400

        for name, group in groups:
            n_take = min(len(group), target_per_group)
            samples.append(group.sample(n=n_take, random_state=42))
            remaining -= n_take

        # 剩余名额从大组中补足
        if remaining > 0:
            extra_pool = pd.concat(samples)
            extra = df[~df.index.isin(extra_pool.index)].sample(n=remaining, random_state=42)
            samples.append(extra)

        result = pd.concat(samples).reset_index(drop=True)
    else:
        # 无行业列则简单均匀抽样
        result = df.sample(n=400, random_state=42).reset_index(drop=True)

    # 如果超过 400 条，随机裁剪到 400
    if len(result) > 400:
        result = result.sample(n=400, random_state=42).reset_index(drop=True)

    print(f"  抽样结果: {len(result)} 条")
    return result


# ── #120: 生成标注模板 ──

def generate_template(df):
    print("[#120] 生成人工标注模板...")

    # 列出关键字段用于辅助判断
    display_fields = ["企业名称", "上市公司行业", "招聘岗位", "招聘类别", "职位描述"]

    lines = []
    lines.append("=" * 70)
    lines.append("  上市公司招聘数据 — 26 类行业标签人工标注模板")
    lines.append("  请在每条的 【标注】 行填入 1 个数字 (1~26)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  编码对照表:")
    for idx, (cat, info) in enumerate(CATEGORIES.items(), 1):
        lines.append(f"  {idx:>2}. {cat:<16s} — {info['定义'][:60]}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    for i in range(len(df)):
        lines.append(f"### 样本 {i+1} / {len(df)}")
        for f in display_fields:
            val = str(df.iloc[i].get(f, ""))
            if len(val) > 300:
                val = val[:300] + "..."
            lines.append(f"  {f}: {val}")
        lines.append(f"  原始行业: {df.iloc[i].get('上市公司行业', '')}")
        lines.append(f"  【标注】:")
        lines.append("-" * 50)

    with open(ANNO_TEMPLATE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  模板已保存 → {ANNO_TEMPLATE}")


# ── 主流程 ──

def main():
    os.makedirs(DIR_12_CLASSIFY, exist_ok=True)
    print("=" * 60)
    print("  分类任务准备 #117 ~ #120")
    print("=" * 60)

    # #117-#118: 保存 26 类定义
    print("[#117-#118] 保存 26 类行业定义...")
    with open(CAT_DEF_JSON, "w", encoding="utf-8") as f:
        json.dump(CATEGORIES, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(CATEGORIES)} 类 → {CAT_DEF_JSON}")
    for idx, (cat, info) in enumerate(CATEGORIES.items(), 1):
        print(f"  {idx:>2}. {cat:<16s} | 关键词: {', '.join(info['关键词'][:5])}")

    # #119: 抽样
    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", engine="python")
    print(f"\n  数据总量: {len(df):,} 条")

    sample_df = sample_400(df)
    sample_df.to_csv(SAMPLE_400_CSV, index=False, encoding="utf-8-sig")
    print(f"  抽样已保存 → {SAMPLE_400_CSV}")

    # #120: 标注模板
    generate_template(sample_df)

    print(f"\n{'=' * 60}")
    print(f"  #117 ~ #120 完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
