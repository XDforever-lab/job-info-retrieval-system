import json
import matplotlib.pyplot as plt
import math

# 设置 Matplotlib 支持中文显示（防止生成的图片里中文变成方块）
# Mac 系统如果报错，请将 'SimHei' 改为 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def extract_keywords_lda(doc, bow, lda_model, dictionary, top_k=6):
    """
    根据文档的主题分布及主题下词语的概率，选择权重最大的主题下的关键词。
    """
    # 1. 获取当前文档的“文档-主题”概率分布
    # 返回格式例如: [(0, 0.1), (1, 0.8), (2, 0.1)]
    doc_topics = lda_model.get_document_topics(bow)
    
    # 2. 找出当前文档概率最高（权重最大）的那个主题
    dominant_topic = max(doc_topics, key=lambda x: x[1])
    dominant_topic_id = dominant_topic[0]
    dominant_topic_prob = dominant_topic[1]
    
    # 3. 获取该“最强主题”下的高频词分布 (取前 50 个备选)
    topic_terms = lda_model.show_topic(dominant_topic_id, topn=50)
    topic_words = [word for word, prob in topic_terms]
    
    # 4. 核心逻辑：从主题词中挑选关键词
    # 老师 PPT 指出：要根据主题分布选择关键词。
    # 必须要确保抽出来的词，在当前这篇文章里确实存在！
    doc_all_words = set(doc["title"] + doc["body"])
    
    final_keywords = []
    for word in topic_words:
        # 如果这个主题代表词在本文档中出现了，并且不是那种单字的噪音词
        if word in doc_all_words and len(word) > 1:
            final_keywords.append(word)
        # 凑够 K 个就停止
        if len(final_keywords) >= top_k:
            break
            
    return dominant_topic_id, dominant_topic_prob, final_keywords


def extract_keywords_lda(doc, bow, lda_model, dictionary, top_k=6):
    """根据文档的主题分布及主题下词语的概率，选择权重最大的主题下的关键词。"""
    doc_topics = lda_model.get_document_topics(bow)
    dominant_topic = max(doc_topics, key=lambda x: x[1])
    dominant_topic_id = dominant_topic[0]
    dominant_topic_prob = dominant_topic[1]
    
    topic_terms = lda_model.show_topic(dominant_topic_id, topn=50)
    topic_words = [word for word, prob in topic_terms]
    
    doc_all_words = set(doc["title"] + doc["body"])
    
    final_keywords = []
    for word in topic_words:
        if word in doc_all_words and len(word) > 1:
            final_keywords.append(word)
        if len(final_keywords) >= top_k:
            break
            
    return dominant_topic_id, dominant_topic_prob, final_keywords

def generate_static_lda_plots(lda_model, num_topics, output_image="lda_static_topics.png"):
    """
    生成适用于实验报告的高清静态 LDA 主题条形图
    """
    print(f"\n正在生成 LDA 静态高清可视化图片...")
    
    # 动态计算子图的行数和列数（每排画 2 个主题）
    cols = 2
    rows = math.ceil(num_topics / cols)
    
    # 创建画布，动态调整高度
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=False)
    axes = axes.flatten()
    
    for i in range(num_topics):
        # 获取该主题下权重最高的前 10 个词
        topic_words = lda_model.show_topic(i, topn=10)
        words = [word for word, prob in topic_words]
        probs = [prob for word, prob in topic_words]
        
        ax = axes[i]
        # 画水平条形图
        bars = ax.barh(words, probs, color='#5D9CEB', edgecolor='black', linewidth=0.5)
        ax.invert_yaxis()  # 让概率最高的词排在最上面
        
        # 优化图表外观
        ax.set_title(f"主题 {i} (Topic {i})", fontsize=14, pad=10)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel("词语生成概率 (Probability)", fontsize=10)
        
        # 在柱子末尾显示具体的概率数值
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center', ha='left', fontsize=9, color='gray')

    # 如果设定的主题数是奇数（比如 5），会多出一个空的子图，把它隐藏掉
    for j in range(num_topics, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    # 导出高清免抠图背景
    plt.savefig(output_image, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"✅ 静态可视化图片已生成！请查看本地文件：{output_image}")
    print("提示：你可以直接将这张 PNG 图片拖入 Word 实验报告中，完美适配排版。")

# ================= 运行区 =================
if __name__ == "__main__":
    from step4_lda_training import prepare_corpus_for_lda, train_and_display_lda
    
    input_json = 'processed_corpus.json'
    with open(input_json, 'r', encoding='utf-8') as f:
        docs = json.load(f)
        
    texts = prepare_corpus_for_lda(docs)
    
    NUM_TOPICS = 10  # 你可以改成你最终确定的主题数
    lda_model, dictionary, corpus = train_and_display_lda(texts, num_topics=NUM_TOPICS, num_words=15)
    
    FINAL_K = 6
    output_filename = f'final_lda_keywords_K{FINAL_K}.txt'
    
    print(f"\n开始为每篇文档提取 LDA 关键词...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, doc in enumerate(docs):
            bow = corpus[idx]
            dom_topic_id, dom_prob, keywords = extract_keywords_lda(doc, bow, lda_model, dictionary, top_k=FINAL_K)
            
            if idx < 3:
                print(f"【文档 {idx+1}】(归属主题 {dom_topic_id}, 概率 {dom_prob*100:.1f}%): {keywords}")
                
            f.write(f"Doc_{idx+1}: {', '.join(keywords)}\n")
            
    print(f"\n✅ LDA 关键词已全部生成并保存至：{output_filename}")
    
    # 核心更改：调用新的静态绘图函数
    generate_static_lda_plots(lda_model, num_topics=NUM_TOPICS, output_image="LDA_Topics_BarCharts.png")


