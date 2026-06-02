#!/usr/bin/env python3
"""
系统评测 —— To Do List 第十一阶段 #253 ~ #272

对应理论课: 第5章 信息检索性能评价 (2x2 表格、P/R/F1、mAP、NDCG)

功能:
  1. 设计 20 条测试查询 (覆盖: 单关键词/多关键词/行业/城市/薪资/复合条件)
  2. 对每条查询取 Top-10, 生成人工标注模板
  3. 从标注结果计算: P@5, P@10, R-Precision, MAP, NDCG@10
  4. 绘制 11 点插值 P/R 曲线
  5. 离线实验评测数据归档汇总

用法:
    python src/post_midterm/evaluator.py    # 生成标注模板
    # 人工标注后
    python src/post_midterm/evaluator.py --eval  # 从标注文件计算指标
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OUTPUT_DIR, MODEL_DIR

# ── 输出路径 ─────────────────────────────────────
EVAL_DIR = os.path.join(OUTPUT_DIR, '17_eval')
ANNO_FILE = os.path.join(EVAL_DIR, 'query_annotations.json')
EVAL_REPORT_FILE = os.path.join(EVAL_DIR, 'search_eval_report.json')
EVAL_SUMMARY_FILE = os.path.join(EVAL_DIR, 'search_eval_summary.txt')

# ── 20 条测试查询 ────────────────────────────────
# 覆盖场景: 单关键词x4 / 多关键词x4 / 行业限定x3 / 城市限定x3 / 薪资限定x2 / 复合条件x4

TEST_QUERIES = [
    # === 单关键词 (4条) ===
    {'id': 1,  'q': 'Python', 'filters': {}, 'desc': '单关键词-技能'},
    {'id': 2,  'q': '财务', 'filters': {}, 'desc': '单关键词-职能'},
    {'id': 3,  'q': '数据分析', 'filters': {}, 'desc': '单关键词-复合词'},
    {'id': 4,  'q': '电气', 'filters': {}, 'desc': '单关键词-行业特征词'},

    # === 多关键词 (4条) ===
    {'id': 5,  'q': 'Python 开发', 'filters': {}, 'desc': '多关键词-技能组合'},
    {'id': 6,  'q': '财务 审计', 'filters': {}, 'desc': '多关键词-职能组合'},
    {'id': 7,  'q': '机器学习 算法', 'filters': {}, 'desc': '多关键词-AI领域'},
    {'id': 8,  'q': '销售 经理 上海', 'filters': {}, 'desc': '多关键词-岗位+城市'},

    # === 行业限定 (3条) ===
    {'id': 9,  'q': '开发', 'filters': {'industry': '计算机、通信和其他电子设备制造业'}, 'desc': '行业限定-IT硬件'},
    {'id': 10, 'q': '工艺', 'filters': {'industry': '电气机械和器材制造业'}, 'desc': '行业限定-电气制造'},
    {'id': 11, 'q': '投资', 'filters': {'industry': '金融'}, 'desc': '行业限定-金融'},

    # === 城市限定 (3条) ===
    {'id': 12, 'q': 'Python', 'filters': {'city': '上海'}, 'desc': '城市限定-上海'},
    {'id': 13, 'q': '财务', 'filters': {'city': '北京'}, 'desc': '城市限定-北京'},
    {'id': 14, 'q': '工程师', 'filters': {'city': '深圳'}, 'desc': '城市限定-深圳'},

    # === 薪资限定 (2条) ===
    {'id': 15, 'q': 'Java', 'filters': {'salary_min': 20000}, 'desc': '薪资限定-月薪20k+'},
    {'id': 16, 'q': '开发', 'filters': {'salary_min': 30000}, 'desc': '薪资限定-月薪30k+'},

    # === 复合条件 (4条) ===
    {'id': 17, 'q': 'Python', 'filters': {'city': '上海', 'education': '本科'}, 'desc': '复合-上海+本科'},
    {'id': 18, 'q': '财务', 'filters': {'industry': '金融', 'city': '上海'}, 'desc': '复合-金融+上海'},
    {'id': 19, 'q': '开发', 'filters': {'city': '北京', 'education': '硕士', 'salary_min': 20000},
     'desc': '复合-北京+硕士+20k+'},
    {'id': 20, 'q': '数据分析', 'filters': {'industry': '互联网和相关服务', 'city': '深圳'},
     'desc': '复合-互联网+深圳'},
]


def load_engine():
    """加载检索引擎"""
    from src.post_midterm.search_engine import SearchEngine
    engine = SearchEngine()
    engine.load()
    return engine


def generate_annotation_template(engine):
    """生成人工标注模板: 对每条查询输出 Top-10 结果 + 标注字段"""
    print("=" * 60)
    print("生成人工标注模板 (#253)")
    print("=" * 60)

    os.makedirs(EVAL_DIR, exist_ok=True)

    annotations = []

    for query_spec in TEST_QUERIES:
        qid = query_spec['id']
        q_text = query_spec['q']
        filters = query_spec['filters']
        desc = query_spec['desc']

        result = engine.search(q_text, filters=filters, sort_by='relevance', page=1, page_size=10)
        items = []
        for doc_id, score in result['results']:
            job = engine.get_job_by_id(doc_id)
            if not job:
                continue
            items.append({
                'doc_id': doc_id,
                'score': round(score, 4),
                'title': job['title'],
                'company': job['company'],
                'city': job['city'],
                'salary_min': job['salary_min'],
                'salary_max': job['salary_max'],
                'education': job['education'],
                'desc_preview': job['description'][:200],
                'relevant': None,   # 待人工标注: 1=相关, 0=不相关
            })

        annotations.append({
            'query_id': qid,
            'query': q_text,
            'filters': filters,
            'desc': desc,
            'total_hits': result['total'],
            'time_ms': round(result['time'] * 1000, 1),
            'results': items,
        })

        print(f"  查询 #{qid}: '{q_text}' {filters} → {len(items)} 条 ({result['total']} 总命中, {result['time']*1000:.1f}ms)")

    # 保存模板
    with open(ANNO_FILE, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"\n标注模板已保存: {ANNO_FILE}")
    print(f"请在 JSON 中为每条的 'relevant' 字段标注 1 (相关) 或 0 (不相关)")
    print(f"标注完成后运行: python src/post_midterm/evaluator.py --eval")

    return annotations


# ══════════════════════════════════════════════════
# 评价指标计算 (对应第5章)
# ══════════════════════════════════════════════════

def precision_at_k(relevant, k):
    """P@K: 前 K 个结果中相关文档的比例"""
    if k <= 0:
        return 0.0
    return sum(relevant[:k]) / k


def average_precision(relevant):
    """Average Precision: 每个相关文档位置的 precision 平均值"""
    if not any(relevant):
        return 0.0
    ap = 0.0
    num_rel = 0
    for i, rel in enumerate(relevant):
        if rel:
            num_rel += 1
            ap += precision_at_k(relevant, i + 1)
    return ap / num_rel if num_rel > 0 else 0.0


def mean_average_precision(all_relevant):
    """MAP: 所有查询 AP 的均值"""
    aps = [average_precision(rel) for rel in all_relevant]
    return sum(aps) / len(aps) if aps else 0.0


def ndcg_at_k(relevant, k):
    """NDCG@K: 归一化折损累计增益 (假设相关度只有 0/1)"""
    if k <= 0:
        return 0.0
    # DCG
    dcg = 0.0
    for i in range(min(k, len(relevant))):
        if relevant[i]:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 i 从 0 开始, log_2(rank+1)
    # IDCG (理想排序: 所有相关文档排在前面)
    total_rel = sum(relevant)
    idcg = 0.0
    for i in range(min(k, total_rel)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def r_precision(relevant):
    """R-Precision: 第 R 个位置的 Precision, R = 总相关文档数"""
    r = sum(relevant)
    if r == 0:
        return 0.0
    return precision_at_k(relevant, r)


def interpolate_pr_curve(relevant, n_points=11):
    """11 点插值 P/R 曲线"""
    points = []
    ap = average_precision(relevant)
    total_rel = sum(relevant)

    if total_rel == 0:
        return [(i / 10, 0.0) for i in range(n_points)]

    for recall_level in [i / 10 for i in range(n_points)]:
        # 在 recall >= recall_level 的区间取最大 precision
        max_prec = 0.0
        for k in range(1, len(relevant) + 1):
            recall_at_k = sum(relevant[:k]) / total_rel if total_rel > 0 else 0
            if recall_at_k >= recall_level:
                prec_at_k = precision_at_k(relevant, k)
                max_prec = max(max_prec, prec_at_k)
        points.append((recall_level, max_prec))

    return points


def compute_confusion_matrix(relevant, k=10):
    """构建 2x2 评价表格 (对应第5章)"""
    retrieved_rel = sum(relevant[:k])
    retrieved_nonrel = k - retrieved_rel
    # 注意: 这里 total_rel 是 Top-K 中标注为相关的数量
    # 真正的 total_rel 是整个文档集中相关的数量 (我们只知道 Top-10 内的)
    # 这里仅基于 Top-10 标注结果构建表格
    total_annotated = len(relevant)
    total_rel_in_annotated = sum(relevant)
    total_nonrel_in_annotated = total_annotated - total_rel_in_annotated

    return {
        'retrieved_relevant': retrieved_rel,
        'retrieved_nonrelevant': retrieved_nonrel,
        'not_retrieved_relevant': total_rel_in_annotated - retrieved_rel,
        'not_retrieved_nonrelevant': total_nonrel_in_annotated - retrieved_nonrel,
        'total_relevant_in_annotated': total_rel_in_annotated,
    }


def evaluate_from_annotations():
    """从标注文件读取并计算所有指标 (#255-#258)"""
    print("=" * 60)
    print("检索效果评测 (#255-#258)")
    print("=" * 60)

    if not os.path.exists(ANNO_FILE):
        print(f"错误: 标注文件不存在 ({ANNO_FILE})")
        print("请先运行: python src/post_midterm/evaluator.py  生成标注模板")
        return

    with open(ANNO_FILE, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    results = []
    all_relevant = []

    for q in annotations:
        relevant = [r.get('relevant', 0) or 0 for r in q['results']]
        relevant = [int(r) if r else 0 for r in relevant]  # 确保是整数

        if not relevant or sum(relevant) == 0:
            # 未标注: 跳过
            continue

        all_relevant.append(relevant)

        p5 = precision_at_k(relevant, 5)
        p10 = precision_at_k(relevant, 10)
        ap = average_precision(relevant)
        rp = r_precision(relevant)
        ndcg = ndcg_at_k(relevant, 10)
        cm = compute_confusion_matrix(relevant)

        results.append({
            'query_id': q['query_id'],
            'query': q['query'],
            'desc': q['desc'],
            'total_hits': q['total_hits'],
            'time_ms': q['time_ms'],
            'num_annotated': len(relevant),
            'num_relevant': sum(relevant),
            'P@5': round(p5, 4),
            'P@10': round(p10, 4),
            'R-Precision': round(rp, 4),
            'AP': round(ap, 4),
            'NDCG@10': round(ndcg, 4),
            'confusion_matrix': cm,
        })

        print(f"   #{q['query_id']:2d}: '{q['query']}' [{q['desc']}]")
        print(f"         P@5={p5:.3f}  P@10={p10:.3f}  R-Prec={rp:.3f}  AP={ap:.3f}  NDCG@10={ndcg:.3f}")

    # 汇总
    if not results:
        print("\n(未找到已标注的查询)")
        return

    map_val = mean_average_precision(all_relevant)
    avg_p5 = sum(r['P@5'] for r in results) / len(results)
    avg_p10 = sum(r['P@10'] for r in results) / len(results)
    avg_ndcg = sum(r['NDCG@10'] for r in results) / len(results)
    avg_rp = sum(r['R-Precision'] for r in results) / len(results)

    # 11 点 P/R 曲线 (取所有查询的平均)
    all_points = []
    for rel in all_relevant:
        pts = interpolate_pr_curve(rel)
        all_points.append(pts)
    avg_curve = []
    for i in range(11):
        avg_prec = sum(p[i][1] for p in all_points) / len(all_points)
        avg_curve.append({'recall': i / 10, 'precision': round(avg_prec, 4)})

    report = {
        'num_queries': len(results),
        'avg_P@5': round(avg_p5, 4),
        'avg_P@10': round(avg_p10, 4),
        'avg_R_Precision': round(avg_rp, 4),
        'MAP': round(map_val, 4),
        'avg_NDCG@10': round(avg_ndcg, 4),
        'pr_curve_11pt': avg_curve,
        'per_query': results,
    }

    # 保存
    with open(EVAL_REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 文本摘要
    lines = [
        "检索效果评测报告",
        "=" * 60,
        f"评测查询数: {len(results)}",
        f"平均 P@5:   {avg_p5:.4f}",
        f"平均 P@10:  {avg_p10:.4f}",
        f"平均 R-Precision: {avg_rp:.4f}",
        f"MAP:        {map_val:.4f}",
        f"平均 NDCG@10: {avg_ndcg:.4f}",
        "",
        "11点插值 P/R 曲线:",
    ]
    for pt in avg_curve:
        lines.append(f"  Recall {pt['recall']:.1f}: Precision {pt['precision']:.4f}")

    lines.extend([
        "",
        "各查询详情:",
    ])
    for r in results:
        lines.append(
            f"  #{r['query_id']:2d} '{r['query']}' [{r['desc']}]: "
            f"P@10={r['P@10']:.3f} AP={r['AP']:.3f} NDCG@10={r['NDCG@10']:.3f}"
        )

    summary = '\n'.join(lines)
    with open(EVAL_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print(f"评测完成")
    print(f"  平均 P@5:    {avg_p5:.4f}")
    print(f"  平均 P@10:   {avg_p10:.4f}")
    print(f"  MAP:         {map_val:.4f}")
    print(f"  平均 NDCG@10: {avg_ndcg:.4f}")
    print(f"  评测报告:    {EVAL_REPORT_FILE}")
    print(f"  文本摘要:    {EVAL_SUMMARY_FILE}")

    return report


def archive_offline_results():
    """离线实验评测数据归档汇总 (#260-#263)"""
    print("\n" + "=" * 60)
    print("离线实验评测归档 (#260-#263)")
    print("=" * 60)

    archive = {}

    # 分词评测 (7_seg_eval)
    seg_eval_file = os.path.join(OUTPUT_DIR, '7_seg_eval', 'seg_eval_report.json')
    if os.path.exists(seg_eval_file):
        with open(seg_eval_file, 'r', encoding='utf-8') as f:
            archive['segmentation'] = json.load(f)
        print("  [OK] 分词评测数据已归档")

    # 关键词评测 (10_keywords)
    kw_eval_file = os.path.join(OUTPUT_DIR, '10_keywords', 'eval_report.json')
    if os.path.exists(kw_eval_file):
        with open(kw_eval_file, 'r', encoding='utf-8') as f:
            archive['keyword'] = json.load(f)
        print("  [OK] 关键词评测数据已归档")

    # 分类评测 (13_classify)
    cls_eval_file = os.path.join(OUTPUT_DIR, '13_classify', 'classify_eval.json')
    if os.path.exists(cls_eval_file):
        with open(cls_eval_file, 'r', encoding='utf-8') as f:
            archive['classification'] = json.load(f)
        print("  [OK] 分类评测数据已归档")

    # 聚类评测 (15_cluster)
    cluster_report_file = os.path.join(OUTPUT_DIR, '15_cluster', 'word_cluster_report.txt')
    cluster_info_file = os.path.join(OUTPUT_DIR, '15_cluster', 'word_cluster_info.json')
    if os.path.exists(cluster_info_file):
        with open(cluster_info_file, 'r', encoding='utf-8') as f:
            archive['clustering'] = json.load(f)
        print("  [OK] 聚类评测数据已归档")

    archive_file = os.path.join(EVAL_DIR, 'offline_experiment_archive.json')
    with open(archive_file, 'w', encoding='utf-8') as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)

    print(f"  归档文件: {archive_file}")
    return archive


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='系统评测工具')
    parser.add_argument('--eval', action='store_true', help='从标注文件计算指标')
    parser.add_argument('--archive', action='store_true', help='归档离线实验结果')
    args = parser.parse_args()

    if args.eval:
        evaluate_from_annotations()
    elif args.archive:
        archive_offline_results()
    else:
        # 默认: 生成标注模板
        engine = load_engine()
        generate_annotation_template(engine)
        print()
        print("=" * 60)
        print("下一步: 请手动标注 query_annotations.json 中的 relevant 字段")
        print("完成后运行: python src/post_midterm/evaluator.py --eval")
        print("=" * 60)


if __name__ == '__main__':
    main()
