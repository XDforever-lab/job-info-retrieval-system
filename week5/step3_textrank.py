import json
import networkx as nx
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import os

# 设置 Matplotlib 支持中文显示（防止生成的图片里中文变成方块）
# 如果你是 Mac 用户，如果报错请将 'SimHei' 改为 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 第1步：TF-IDF 获取候选词 (精简版) ---
def calculate_df(documents):
    df_dict = defaultdict(int)
    for doc in documents:
        for word in set(doc["title"] + doc["body"]):
            df_dict[word] += 1
    return df_dict

def get_tfidf_candidates(doc, df_dict, N, candidate_k=25, title_weight=2.0):
    word_tf_scores = defaultdict(float)
    for word in doc["body"]:
        word_tf_scores[word] += 1.0
    for word in doc["title"]:
        word_tf_scores[word] += title_weight
        
    scored_words = []
    for word, tf in word_tf_scores.items():
        df = df_dict.get(word, 0)
        idf = math.log(N / (df + 1))
        scored_words.append((word, tf * idf))
        
    scored_words.sort(key=lambda x: x[1], reverse=True)
    return set([word for word, score in scored_words[:candidate_k]])

# --- 第2、3步：TextRank 核心计算与可视化输出 ---
def extract_keywords_textrank_detailed(doc, candidate_words, window_size=4, final_k=6, doc_id=1):
    full_text_sequence = doc["title"] + doc["body"]
    
    # 构建无向图
    graph = nx.Graph()
    for word in candidate_words:
        graph.add_node(word)
        
    # 滑动窗口建立共现边
    seq_length = len(full_text_sequence)
    for i in range(seq_length):
        word_A = full_text_sequence[i]
        if word_A not in candidate_words:
            continue
        for j in range(i + 1, min(i + window_size, seq_length)):
            word_B = full_text_sequence[j]
            if word_B in candidate_words and word_A != word_B:
                if graph.has_edge(word_A, word_B):
                    graph[word_A][word_B]['weight'] += 1.0
                else:
                    graph.add_edge(word_A, word_B, weight=1.0)
                    
    # 计算 PageRank
    try:
        pr_scores = nx.pagerank(graph, alpha=0.85, weight='weight')
    except nx.PowerIterationFailedConvergence:
        return []
        
    sorted_words = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    
    # ================= 专属第一篇文档的深度剖析 =================
    if doc_id == 1:
        print(f"\n{'='*20} 【深度剖析】文档 1 的 TextRank 内部运作 {'='*20}")
        print("【1. 词共现网络边与权重 (Edges & Weights)】")
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            print(f"  [{u}] <---> [{v}]  (共现次数: {data['weight']})")
            edge_count += 1
            if edge_count >= 15:
                print("  ... (省略部分连线展示)")
                break
                
        print("\n【2. 节点 PageRank 最终迭代得分 (Node Scores)】")
        for word, score in sorted_words[:10]:
            print(f"  节点: {word:<10} | 得分: {score:.6f}")
        print("  ...")

        # --- 自动绘制并保存网络图 ---
        plt.figure(figsize=(10, 8))
        # 使用弹簧布局，让连接紧密的节点靠在一起
        pos = nx.spring_layout(graph, k=0.5, seed=42) 
        
        # 提取节点大小 (基于 PR 得分放大)
        node_sizes = [pr_scores[node] * 30000 for node in graph.nodes()]
        # 提取边的粗细 (基于共现次数)
        edge_widths = [graph[u][v]['weight'] * 1.5 for u, v in graph.edges()]
        
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(graph, pos, font_size=11, font_family='SimHei')
        
        plt.title(f"文档 1 关键词 TextRank 共现网络 (窗口大小={window_size})", fontsize=15)
        plt.axis('off') # 隐藏坐标轴
        img_filename = "TextRank_Network_Doc1.png"
        plt.savefig(img_filename, dpi=300, bbox_inches='tight')
        print(f"\n✅ 已自动生成高清网络结构图：{os.path.abspath(img_filename)}")
        print("="*66 + "\n")

    return [word for word, score in sorted_words[:final_k]]

# ================= 运行区 =================
if __name__ == "__main__":
    input_json = 'processed_corpus.json'
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"找不到 {input_json}，请先运行预处理。")
        exit()

    N = len(docs)
    df_dict = calculate_df(docs)
    
    CANDIDATE_K = 25  
    WINDOW_SIZE = 5   
    FINAL_K = 6       
    output_filename = f'final_textrank_keywords_K{FINAL_K}.txt'
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, doc in enumerate(docs):
            candidates = get_tfidf_candidates(doc, df_dict, N, candidate_k=CANDIDATE_K)
            # 传入 idx+1 作为 doc_id，以便仅对第一篇文章进行可视化剖析
            final_keywords = extract_keywords_textrank_detailed(doc, candidates, window_size=WINDOW_SIZE, final_k=FINAL_K, doc_id=idx+1)
            f.write(f"Doc_{idx+1}: {', '.join(final_keywords)}\n")
            
    print(f"✅ 全量抽取完成，最终结果已保存至：{output_filename}")