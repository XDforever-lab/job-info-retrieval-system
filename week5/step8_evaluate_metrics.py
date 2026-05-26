import os
import matplotlib.pyplot as plt
import numpy as np

# 设置 Matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

def read_keywords(filename):
    """读取提取结果文件，返回字典 {doc_idx: set(words)}"""
    results = {}
    if not os.path.exists(filename):
        print(f"⚠️ 找不到文件: {filename}")
        return results
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ":" in line:
                doc_str, words_str = line.strip().split(':', 1)
                doc_idx = int(doc_str.replace("Doc_", "").strip()) - 1
                words = {w.strip() for w in words_str.split(',') if w.strip()}
                results[doc_idx] = words
    return results

def calculate_metrics(predicted_dict, standard_dict):
    """
    计算宏平均 (Macro-average) 的 P, R, F1。
    即先算每一篇的指标，再对所有文章求平均。
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    # 仅遍历有标准答案的文档
    for doc_idx, std_words in standard_dict.items():
        if not std_words:  # 如果这篇文档没有标准答案，跳过
            continue
            
        pred_words = predicted_dict.get(doc_idx, set())
        
        # 核心：计算命中数量（交集）
        hit_words = pred_words.intersection(std_words)
        hit_count = len(hit_words)
        
        # 查准率 P = 命中数 / 预测提交数
        p = hit_count / len(pred_words) if len(pred_words) > 0 else 0.0
        # 查全率 R = 命中数 / 标准答案总数
        r = hit_count / len(std_words) if len(std_words) > 0 else 0.0
        # F1 值 = 2*P*R / (P+R)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        
    # 计算均值
    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return avg_p, avg_r, avg_f1

if __name__ == "__main__":
    print("="*40)
    print("正在加载评价数据集...")
    
    # 1. 加载金标准
    std_answers = read_keywords('final_standard_answers.txt')
    evaluated_docs_count = len([v for v in std_answers.values() if v])
    print(f"成功加载标准答案，共包含 {evaluated_docs_count} 篇有效测评文档。")
    
    # 2. 加载三个算法的 K=6 预测结果
    tfidf_res = read_keywords('final_tfidf_keywords_K6.txt')
    textrank_res = read_keywords('final_textrank_keywords_K6.txt')
    lda_res = read_keywords('final_lda_keywords_K6.txt')
    
    # 3. 分别计算指标
    metrics = {}
    metrics['TF-IDF'] = calculate_metrics(tfidf_res, std_answers)
    metrics['TextRank'] = calculate_metrics(textrank_res, std_answers)
    metrics['LDA'] = calculate_metrics(lda_res, std_answers)
    
    # ================= 终端表格输出 =================
    print("\n" + "="*50)
    print("       关键词抽取算法性能对比评测 (K=6)       ")
    print("="*50)
    print(f"{'算法名称':<12} | {'查准率(P)':<10} | {'查全率(R)':<10} | {'F1值':<10}")
    print("-" * 50)
    
    for algo, (p, r, f1) in metrics.items():
        print(f"{algo:<12} | {p*100:>8.2f}% | {r*100:>8.2f}% | {f1*100:>7.2f}%")
    print("="*50 + "\n")
    
    # ================= 生成直观对比柱状图 =================
    labels = ['TF-IDF', 'TextRank', 'LDA']
    p_vals = [metrics[algo][0] * 100 for algo in labels]
    r_vals = [metrics[algo][1] * 100 for algo in labels]
    f1_vals = [metrics[algo][2] * 100 for algo in labels]
    
    x = np.arange(len(labels))
    width = 0.25  # 柱子宽度
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, p_vals, width, label='查准率 (Precision)', color='#4c72b0')
    rects2 = ax.bar(x, r_vals, width, label='查全率 (Recall)', color='#dd8452')
    rects3 = ax.bar(x + width, f1_vals, width, label='F1 值 (F1-Score)', color='#55a868')
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 向上偏移 3 像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    ax.set_ylabel('百分比 (%)', fontsize=12)
    ax.set_title('三种无监督关键词抽取算法性能对比 (Top-K=6)', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='upper right')
    
    ax.set_ylim(0, max(max(p_vals), max(r_vals), max(f1_vals)) + 15) # 留出顶部空间显示数字
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存并展示
    img_filename = "Algorithm_Metrics_Comparison.png"
    plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 评测柱状图已生成并保存至：{os.path.abspath(img_filename)}")