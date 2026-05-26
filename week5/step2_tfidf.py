import json
import math
from collections import defaultdict

def calculate_df(documents):
    """计算全局的文档频率 (DF)"""
    df_dict = defaultdict(int)
    for doc in documents:
        unique_words_in_doc = set(doc["title"] + doc["body"])
        for word in unique_words_in_doc:
            df_dict[word] += 1
    return df_dict

def extract_single_doc_keywords(doc, df_dict, N, title_weight=2.0, top_k=6):
    """
    精简版核心计算逻辑：使用绝对词频 (TF) 进行抽取，返回指定文档的 Top-K 词汇列表
    """
    word_tf_scores = defaultdict(float)
    
    for word in doc["body"]:
        word_tf_scores[word] += 1.0
    for word in doc["title"]:
        word_tf_scores[word] += title_weight
        
    scored_words = []
    for word, tf in word_tf_scores.items():
        df = df_dict.get(word, 0)
        idf = math.log(N / (df + 1))
        score = tf * idf
        scored_words.append((word, score))
        
    # 按 TF-IDF 得分降序排列
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    # 仅提取词汇本身，丢弃分数
    return [word for word, score in scored_words[:top_k]]

# ================= 运行区 =================
if __name__ == "__main__":
    input_json = 'processed_corpus.json'
    
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"找不到 {input_json}，请确保第一步生成了该文件。")
        exit()

    N = len(docs)
    print(f"成功加载语料，共计 {N} 篇文档。正在构建全局 DF 字典...\n")
    df_dict = calculate_df(docs)
    
    # 直接指定最优 K 值为 6
    best_k = 6
    output_filename = f'final_tfidf_keywords_K{best_k}.txt'
    
    print(f"正在使用 K = {best_k} 为全库 {N} 篇文章生成最终关键词文件...")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, doc in enumerate(docs):
            final_keywords = extract_single_doc_keywords(doc, df_dict, N, title_weight=2.0, top_k=best_k)
            # 将结果写入文本，格式为：文档ID: 关键词1, 关键词2...
            f.write(f"Doc_{idx+1}: {', '.join(final_keywords)}\n")
            
    print(f"\n✅ 大功告成！完美包含所有文章的关键词文件已保存至：{output_filename}")