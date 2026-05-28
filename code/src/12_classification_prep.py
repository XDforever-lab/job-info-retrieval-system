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
    "互联网/AI": {
        "定义": "互联网产品研发、人工智能算法、软件开发、大数据、云计算、网络安全等技术研发类岗位，以及AI大模型、机器学习、计算机视觉等前沿技术方向",
        "关键词": ["软件", "开发", "Java", "Python", "AI", "算法", "前端", "后端", "架构", "数据",
                  "人工智能", "机器学习", "深度学习", "NLP", "CV", "模型", "工程师", "编程", "C++", "SaaS"]
    },
    "餐饮": {
        "定义": "连锁餐饮、酒店餐饮、团餐等餐饮服务行业的厨房烹饪、前厅服务、食品安全管理岗位",
        "关键词": ["厨师", "厨工", "餐饮", "食品", "烹饪", "配菜", "食品安全", "切菜", "面点", "食堂"]
    },
    "直播/影视/传媒": {
        "定义": "短视频、直播带货、影视制作、广告传媒、内容策划、新媒体运营等文化传媒类岗位",
        "关键词": ["直播", "视频", "影视", "剪辑", "拍摄", "抖音", "新媒体", "传媒", "编导", "内容"]
    },
    "电子/电气/通信": {
        "定义": "半导体、芯片、PCB、电子元器件、通信设备、光电、自动化控制、嵌入式系统等硬件技术岗位",
        "关键词": ["电子", "电气", "芯片", "半导体", "嵌入式", "PCB", "射频", "硬件", "通信", "电路",
                  "PLC", "自动化", "传感器", "PCB", "IC"]
    },
    "产品": {
        "定义": "产品规划设计、需求分析、用户体验（UX/UI）、产品运营、版本迭代管理等产品类岗位",
        "关键词": ["产品经理", "产品设计", "需求", "用户体验", "原型", "PRD", "交互", "UI", "UX", "B端", "C端"]
    },
    "客服/运营": {
        "定义": "客户服务、售后支持、电商运营、用户运营、社群运营等面向终端用户的服务与运营类岗位",
        "关键词": ["客服", "售后", "电商", "运营", "社群", "用户", "投诉", "热线", "呼叫", "服务", "店铺"]
    },
    "销售": {
        "定义": "渠道开拓、客户开发维护、商务谈判、销售管理、区域代理等以业绩达成为核心的销售类岗位",
        "关键词": ["销售", "客户经理", "渠道", "代理商", "业务拓展", "签单", "业绩", "提成", "回款", "大客户"]
    },
    "人力/行政/法务": {
        "定义": "招聘、培训、薪酬绩效、员工关系、法务合规、行政管理、秘书助理等企业内部支撑职能岗位",
        "关键词": ["HR", "人力资源", "招聘", "薪酬", "绩效", "培训", "员工关系", "行政", "法务", "合规", "秘书"]
    },
    "财务/审计/税务": {
        "定义": "会计核算、财务报表编制、成本管控、内部审计、税务筹划、出纳等财税审计专业岗位",
        "关键词": ["财务", "会计", "审计", "税务", "出纳", "报表", "成本", "核算", "CPA", "账务", "预算", "结算"]
    },
    "生产制造": {
        "定义": "车间生产管理、设备操作维护、工艺技术、质量检测、精益生产、EHS安全等制造现场岗位",
        "关键词": ["生产", "制造", "车间", "工艺", "设备", "质量", "品质", "QC", "质检", "流水线",
                  "操作工", "普工", "精益", "6S", "IE"]
    },
    "零售/生活服务": {
        "定义": "超市卖场、连锁门店、便利店、美容美发、家政、宠物服务等面向个人消费者的零售及生活服务岗位",
        "关键词": ["门店", "超市", "零售", "店员", "收银", "理货", "美容", "美发", "家政", "宠物"]
    },
    "酒店/旅游": {
        "定义": "酒店运营、前台接待、客房管理、旅游服务、景区管理等酒店旅游行业的服务类岗位",
        "关键词": ["酒店", "前台", "客房", "旅游", "景区", "预订", "礼宾", "大堂", "民宿", "OTA"]
    },
    "教育培训": {
        "定义": "K12教育、职业教育、语言培训、企业内训等教育培训行业的教师、教研、课程设计岗位",
        "关键词": ["教师", "讲师", "培训", "教育", "课程", "教研", "班主任", "助教", "教学", "授课"]
    },
    "设计": {
        "定义": "平面视觉设计、UI/UX设计、工业设计、室内设计、服装设计等专业设计类岗位",
        "关键词": ["设计", "平面", "UI", "UX", "视觉", "工业设计", "室内设计", "CAD", "PS", "渲染", "建模"]
    },
    "房地产/建筑": {
        "定义": "房地产开发、建筑工程施工、工程造价、招投标、物业管理、城市规划等相关岗位",
        "关键词": ["房地产", "建筑", "工程", "施工", "造价", "招投标", "物业", "装修", "监理", "土木", "项目经理"]
    },
    "市场/公关/广告": {
        "定义": "品牌营销、市场推广、活动策划、品牌公关、媒介传播、SEO/SEM等市场与公关类岗位",
        "关键词": ["市场", "营销", "品牌", "公关", "广告", "推广", "活动策划", "SEO", "SEM", "媒介", "传播"]
    },
    "物流/仓储/司机": {
        "定义": "供应链物流、仓储管理、快递配送、货运司机、跨境物流等流通运输类岗位",
        "关键词": ["物流", "仓储", "司机", "配送", "快递", "运输", "供应链", "货运", "跨境", "干线"]
    },
    "采购/贸易": {
        "定义": "原材料采购、供应商管理、外贸进出口、贸易跟单、成本控制等采购与贸易岗位",
        "关键词": ["采购", "供应商", "贸易", "外贸", "进出口", "跟单", "报价", "谈判", "合同", "BOM"]
    },
    "汽车": {
        "定义": "汽车整车制造、零部件生产、新能源汽车（三电系统）、汽车销售、售后维修保养等汽车产业链岗位",
        "关键词": ["汽车", "新能源", "电池", "电芯", "电机", "电控", "整车", "零部件", "4S", "充电", "三电"]
    },
    "医疗健康": {
        "定义": "医药研发、医疗器械、临床医疗、药品生产、健康管理、生物技术等医疗健康领域岗位",
        "关键词": ["医药", "制药", "医疗器械", "临床", "药品", "生物", "医院", "护理", "试剂", "GMP", "GLP"]
    },
    "金融": {
        "定义": "银行、证券、保险、基金、信托、信贷、风控等传统金融及金融科技领域的岗位",
        "关键词": ["金融", "银行", "信贷", "风控", "投资", "证券", "保险", "基金", "理财", "交易", "资管"]
    },
    "项目管理": {
        "定义": "跨行业的项目管理岗位（不含建筑/IT特定），包括项目计划、进度管控、资源协调、PMO等",
        "关键词": ["项目管理", "PMO", "项目经理", "进度", "交付", "里程碑", "甘特图", "PMP", "敏捷", "Scrum"]
    },
    "咨询/翻译/法律": {
        "定义": "管理咨询、行业研究、翻译口译、法律服务（律所/企业法务）、知识产权等专业服务岗位",
        "关键词": ["咨询", "翻译", "律师", "法律", "IP", "知识产权", "尽职调查", "行业研究", "英语翻译", "日语"]
    },
    "能源/环保/农业": {
        "定义": "光伏、风电、储能、电力、水务环保、现代农业、化工等资源能源与环保农业岗位",
        "关键词": ["能源", "光伏", "风电", "储能", "环保", "水务", "农业", "化工", "电力", "碳中和"]
    },
    "高级管理": {
        "定义": "总监及以上级别的高级管理岗位，包括CEO/VP/总监/总经办等战略决策层面职位",
        "关键词": ["总监", "副总裁", "VP", "总经理", "CEO", "CTO", "CFO", "总裁", "事业部负责人", "区域总", "部长"]
    },
    "其他": {
        "定义": "不属于以上 25 类的其他所有岗位，作为兜底分类",
        "关键词": []
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
