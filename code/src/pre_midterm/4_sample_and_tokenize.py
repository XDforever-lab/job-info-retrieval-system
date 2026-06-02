import csv
import random
import jieba
from collections import defaultdict
from datetime import datetime

random.seed(42)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEANED_CSV, SAMPLED_CSV, DIR_4_SAMPLE

INPUT_FILE = CLEANED_CSV
OUTPUT_CSV = SAMPLED_CSV
OUTPUT_DOC = os.path.join(DIR_4_SAMPLE, "分词采样过程记录.txt")

TOTAL_SAMPLES = 200

print("=" * 60)
print("STEP 1: 读取原始CSV文件")
print("=" * 60)

with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
    reader = list(csv.DictReader(f))

total_rows = len(reader)
print(f"总记录数: {total_rows}")

print()
print("=" * 60)
print("STEP 2: 统计行业和企业分布")
print("=" * 60)

industry_data = defaultdict(list)
company_data = defaultdict(list)

for row in reader:
    industry = row['上市公司行业']
    company = row['企业名称']
    industry_data[industry].append(row)
    company_data[company].append(row)

num_industries = len(industry_data)
num_companies = len(company_data)
print(f"不同行业数: {num_industries}")
print(f"不同企业数: {num_companies}")

print()
print("各行业记录数排名(前20):")
industry_counts = sorted(industry_data.items(), key=lambda x: len(x[1]), reverse=True)
for i, (ind, rows) in enumerate(industry_counts[:20], 1):
    print(f"  {i:>2}. {ind}: {len(rows)}条")

print()
print("=" * 60)
print("STEP 3: 按行业比例分配200个采样名额（每行业至少1条）")
print("=" * 60)

# 步骤1: 给每个行业至少分配1条（如果该行业至少有1条记录）
allocation = {}
remaining = TOTAL_SAMPLES

eligible_industries = [ind for ind in industry_data if len(industry_data[ind]) > 0]
eligible_industries.sort()

for ind in eligible_industries:
    allocation[ind] = 1
    remaining -= 1

print(f"先给{len(eligible_industries)}个行业各分配1条，剩余{remaining}条按比例分配")

# 步骤2: 剩余名额按各行业记录数比例分配
total_eligible = sum(len(industry_data[ind]) for ind in eligible_industries)
allocated_extra = 0
industry_items_sorted = sorted(eligible_industries, key=lambda x: len(industry_data[x]), reverse=True)
for ind in industry_items_sorted:
    if remaining <= 0:
        break
    proportion = len(industry_data[ind]) / total_eligible
    extra = max(1, int(proportion * remaining))
    if extra > len(industry_data[ind]) - 1:
        extra = len(industry_data[ind]) - 1
    allocation[ind] += extra
    allocated_extra += extra

# 调整确保正好200
total_allocated = sum(allocation.values())
print(f"当前分配总数: {total_allocated}")

if total_allocated < TOTAL_SAMPLES:
    diff = TOTAL_SAMPLES - total_allocated
    print(f"不足{total_allocated}条，需补齐{diff}条，优先补给记录最多的行业")
    industry_items_sorted = sorted(eligible_industries, key=lambda x: len(industry_data[x]), reverse=True)
    for ind in industry_items_sorted:
        if diff <= 0:
            break
        current = allocation[ind]
        max_possible = len(industry_data[ind])
        add = min(diff, max_possible - current)
        allocation[ind] += add
        diff -= add
elif total_allocated > TOTAL_SAMPLES:
    diff = total_allocated - TOTAL_SAMPLES
    print(f"超出{total_allocated}条，需减少{diff}条，从记录最少的行业减少")
    industry_items_sorted = sorted(eligible_industries, key=lambda x: len(industry_data[x]))
    for ind in industry_items_sorted:
        if diff <= 0:
            break
        if allocation[ind] > 1:
            allocation[ind] -= 1
            diff -= 1

print(f"最终分配总数: {sum(allocation.values())}")
print()
print("各行业分配名额(按数量排序前20):")
allocation_sorted = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
for i, (ind, cnt) in enumerate(allocation_sorted[:20], 1):
    total = len(industry_data[ind])
    print(f"  {i:>2}. {ind}: {cnt}条 / 共{total}条")

print()
print("=" * 60)
print("STEP 4: 在每个行业内均匀从不同企业抽取数据")
print("=" * 60)

sampled_rows = []
industry_sample_detail = []

for industry, target_count in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
    rows_in_industry = industry_data[industry]

    # 按企业分组
    company_groups = defaultdict(list)
    for row in rows_in_industry:
        company_groups[row['企业名称']].append(row)

    companies_list = list(company_groups.keys())
    random.shuffle(companies_list)

    industry_sampled = []
    company_sampled_count = {}

    # 均匀分配给不同企业：轮流从每个企业取1条
    company_idx = 0
    while len(industry_sampled) < target_count and company_idx < len(companies_list) * len(rows_in_industry):
        company = companies_list[company_idx % len(companies_list)]
        if company not in company_sampled_count:
            company_sampled_count[company] = 0

        available = [r for r in company_groups[company] if r not in industry_sampled]
        if available:
            chosen = random.choice(available)
            industry_sampled.append(chosen)
            company_sampled_count[company] += 1

        company_idx += 1
        if len(industry_sampled) >= target_count:
            break

    sampled_rows.extend(industry_sampled)
    industry_sample_detail.append((industry, target_count, len(companies_list), company_sampled_count))

print(f"实际采样总数: {len(sampled_rows)}")

unique_industries_in_sample = set(r['上市公司行业'] for r in sampled_rows)
unique_companies_in_sample = set(r['企业名称'] for r in sampled_rows)
print(f"采样覆盖行业数: {len(unique_industries_in_sample)}")
print(f"采样覆盖企业数: {len(unique_companies_in_sample)}")

print()
print("=" * 60)
print("STEP 5: Jieba分词处理职位描述")
print("=" * 60)

# 添加自定义词典（金融行业常用词）
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

tokenized_count = 0
for row in sampled_rows:
    desc = row.get('职位描述', '')
    if desc:
        desc = desc.replace('\n', ' ').replace('\r', ' ')
        words = jieba.lcut(desc)
        words = [w.strip() for w in words if w.strip() and w.strip() not in ('', ' ', '、', '：', '；', '。', '，', '．', '（', '）', '(', ')', '【', '】', '[', ']')]
        row['职位描述_分词'] = '/'.join(words)
        tokenized_count += 1
    else:
        row['职位描述_分词'] = ''

print(f"成功分词: {tokenized_count}条")

print()
print("=" * 60)
print("STEP 6: 输出结果CSV")
print("=" * 60)

with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['编号', '职位描述'])
    writer.writeheader()
    for row in sampled_rows:
        writer.writerow({
            '编号': row['编号'],
            '职位描述': row['职位描述_分词']
        })

print(f"结果已保存到: {OUTPUT_CSV}")
print(f"输出行数: {len(sampled_rows)}")

print()
print("=" * 60)
print("STEP 7: 生成过程记录文档")
print("=" * 60)

doc_lines = []
doc_lines.append("=" * 70)
doc_lines.append("上市公司招聘数据 - 分词采样处理过程记录")
doc_lines.append("=" * 70)
doc_lines.append(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
doc_lines.append("")

doc_lines.append("一、数据来源")
doc_lines.append("-" * 40)
doc_lines.append(f"  源文件: 上市公司招聘数据_cleaned.csv")
doc_lines.append(f"  原始总记录数: {total_rows}")
doc_lines.append(f"  原始字段: 编号、企业名称、股票简称、上市公司行业、招聘岗位、")
doc_lines.append(f"           工作城市、最低月薪、最高月薪、职位描述、学历要求、")
doc_lines.append(f"           要求经验、招聘类别、招聘发布日期、招聘结束日期")
doc_lines.append("")

doc_lines.append("二、采样策略")
doc_lines.append("-" * 40)
doc_lines.append(f"  目标采样数: {TOTAL_SAMPLES}条")
doc_lines.append(f"  采样原则: 均匀地从不同行业和企业中抽取，兼顾行业覆盖与企业多样性")
doc_lines.append("")
doc_lines.append("  具体方法:")
doc_lines.append("  1. 按行业比例分配200个名额，每个行业至少分配1条")
doc_lines.append(f"  2. {num_industries}个行业中原有{len(eligible_industries)}个行业被分配到名额")
doc_lines.append("  3. 记录较多的行业（计算机通信、电气机械、汽车制造等）获较多名额")
doc_lines.append("  4. 在每个行业内，以轮询(round-robin)方式均匀从不同企业抽取数据")
doc_lines.append("  5. 同一企业在该行业内最多被抽取的条数受其总数据量限制")
doc_lines.append("")

doc_lines.append("三、各行业采样分布")
doc_lines.append("-" * 40)
doc_lines.append(f"  {'行业名称':<35s} {'分配数':>5s} {'行业总数':>8s} {'占比':>7s}")
doc_lines.append(f"  {'-'*35} {'-'*5} {'-'*8} {'-'*7}")
for ind, cnt, _, _ in industry_sample_detail:
    total_in_ind = len(industry_data[ind])
    percentage = f"{cnt/total_in_ind*100:.1f}%"
    doc_lines.append(f"  {ind:<35s} {cnt:>5} {total_in_ind:>8} {percentage:>7s}")

doc_lines.append("")

doc_lines.append("四、分词处理")
doc_lines.append("-" * 40)
doc_lines.append("  分词工具: jieba (结巴分词)")
doc_lines.append("  分隔符: '/'")
doc_lines.append(f"  成功处理: {tokenized_count}条职位描述")
doc_lines.append("  自定义词典: 添加了金融行业常用词（会计、财务管理、大数据等）")
doc_lines.append("  过滤规则: 去除标点符号、空白字符等噪音")
doc_lines.append("")

doc_lines.append("五、输出文件")
doc_lines.append("-" * 40)
doc_lines.append(f"  输出文件: 分词采样结果_200条.csv")
doc_lines.append(f"  输出路径: {OUTPUT_CSV}")
doc_lines.append(f"  输出字段: 编号、职位描述（分词后，以'/'分隔）")
doc_lines.append(f"  输出行数: {len(sampled_rows)}")
doc_lines.append(f"  覆盖行业数: {len(unique_industries_in_sample)}")
doc_lines.append(f"  覆盖企业数: {len(unique_companies_in_sample)}")
doc_lines.append("")

doc_lines.append("六、采样结果统计")
doc_lines.append("-" * 40)

# 按行业统计
sample_industry_count = defaultdict(int)
sample_company_count = defaultdict(int)
for row in sampled_rows:
    sample_industry_count[row['上市公司行业']] += 1
    sample_company_count[row['企业名称']] += 1

for ind, cnt, num_comp, comp_detail in industry_sample_detail:
    comp_list = sorted(comp_detail.items(), key=lambda x: x[1], reverse=True)
    doc_lines.append(f"  行业: {ind} (采样{cnt}条，来自{num_comp}家企业)")
    for comp, c in comp_list:
        doc_lines.append(f"    - {comp}: {c}条")

doc_lines.append("")
doc_lines.append("七、分词示例（前5条）")
doc_lines.append("-" * 40)
for i, row in enumerate(sampled_rows[:5]):
    doc_lines.append(f"  编号{row['编号']}: {row['职位描述_分词'][:200]}...")
    doc_lines.append("")

doc_lines.append("=" * 70)
doc_lines.append("记录完毕")

with open(OUTPUT_DOC, 'w', encoding='utf-8') as f:
    f.write('\n'.join(doc_lines))

print(f"过程记录已保存到: {OUTPUT_DOC}")

print()
print("=" * 60)
print("全部处理完成！")
print("=" * 60)
