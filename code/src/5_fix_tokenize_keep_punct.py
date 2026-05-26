"""纠正分词采样结果：从原始语料中匹配编号，重新用结巴分词，保留标点符号。"""
import csv
import jieba

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEANED_CSV, SAMPLED_CSV

INPUT_SAMPLED = SAMPLED_CSV
INPUT_CORPUS = CLEANED_CSV
OUTPUT_CSV = SAMPLED_CSV

# 自定义词典（与原脚本保持一致）
custom_words = [
    '会计', '财务管理', '大数据', '数据分析', '风险管控', '产品经理', '客户经理',
    '供应链', '票据业务', '流动性管理', '资产负债', '净息差', '资本充足率',
    '贸易融资', '会计核算', '预算分析', '税务管理', '薪酬管理', '数据治理',
    '风险管理', '市场研究', '交易系统', '营销推动', '培训宣导', '客户关系',
    '产品研发', '系统建设', '需求分析', '方案设计', '运营管理', '数据建模',
    '用户体验', '交互设计', '品牌设计', '资本规划', '人才开发', '机器学习',
    '深度学习', '自然语言处理', '计算机视觉', '人工智能', '算法模型',
    'Java', 'Python', 'C++', 'JavaScript', 'SQL', 'HTML', 'CSS', 'Linux',
]
for word in custom_words:
    jieba.add_word(word)

print("STEP 1: 读取采样文件，提取编号列表")
with open(INPUT_SAMPLED, 'r', encoding='utf-8-sig') as f:
    sampled_reader = list(csv.DictReader(f))

sampled_ids = set()
for row in sampled_reader:
    try:
        sampled_ids.add(int(row['编号']))
    except ValueError:
        pass

print(f"  采样文件中的编号数: {len(sampled_ids)}")

print("\nSTEP 2: 读取原始语料，按编号匹配")
with open(INPUT_CORPUS, 'r', encoding='utf-8-sig') as f:
    corpus_reader = list(csv.DictReader(f))

id_to_desc = {}
for row in corpus_reader:
    try:
        rid = int(row['编号'])
        if rid in sampled_ids:
            id_to_desc[rid] = row['职位描述']
    except (ValueError, KeyError):
        pass

print(f"  成功匹配: {len(id_to_desc)}条")
missing = sampled_ids - set(id_to_desc.keys())
if missing:
    print(f"  未匹配到的编号: {sorted(missing)}")

print("\nSTEP 3: 重新分词（保留标点符号）")
tokenized_count = 0
id_to_tokens = {}
for rid in sorted(id_to_desc.keys()):
    desc = id_to_desc[rid]
    if desc:
        desc = desc.replace('\n', ' ').replace('\r', ' ')
        words = jieba.lcut(desc)
        # 只去除空白字符，保留所有标点符号
        words = [w.strip() for w in words if w.strip()]
        id_to_tokens[rid] = '/'.join(words)
        tokenized_count += 1
    else:
        id_to_tokens[rid] = ''

print(f"  成功分词: {tokenized_count}条")

print("\nSTEP 4: 写入结果（按原顺序）")
with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['编号', '职位描述'])
    writer.writeheader()
    for row in sampled_reader:
        rid = int(row['编号'])
        if rid in id_to_tokens:
            writer.writerow({'编号': rid, '职位描述': id_to_tokens[rid]})
        else:
            writer.writerow({'编号': rid, '职位描述': row['职位描述']})

print(f"  结果已保存到: {OUTPUT_CSV}")
print("\n完成！标点符号已保留。")
