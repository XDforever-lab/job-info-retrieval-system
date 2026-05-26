import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 设置 Matplotlib 中文显示（Windows默认SimHei，Mac用户若报错可换成 Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 绘制极度浓缩的主题词云矩阵 =================
def generate_topic_wordclouds(lda_model, num_topics, font_path=None):
    print("\n正在生成紧凑版【主题词云矩阵图】...")
    
    # 动态计算行列（例如10个主题就是 2行5列）
    cols = 5
    rows = int(np.ceil(num_topics / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    
    # 注意：词云必须要指定中文字体路径，否则全是方块。
    # 这里提供一个 Windows 系统常用的黑体路径，如果你是 Mac，请改为 '/System/Library/Fonts/PingFang.ttc'
    if not font_path:
        import platform
        if platform.system() == 'Windows':
            font_path = 'C:/Windows/Fonts/simhei.ttf'
        else:
            font_path = '/System/Library/Fonts/PingFang.ttc'
            
    for i in range(num_topics):
        # 提取主题词和概率，转化为字典喂给词云
        topic_words = dict(lda_model.show_topic(i, topn=30))
        
        try:
            wc = WordCloud(font_path=font_path, width=400, height=300, 
                        background_color='white', colormap='tab20',
                        max_words=30).generate_from_frequencies(topic_words)
            
            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f"主题 {i}", fontsize=14, pad=10)
            axes[i].axis('off') # 隐藏坐标轴
        except OSError:
            print(f"⚠️ 警告：找不到字体文件 {font_path}，词云生成失败。")
            return

    # 隐藏多余的空白子图
    for j in range(num_topics, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    output_image = "LDA_Compact_WordClouds.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ 主题词云矩阵已生成：{output_image}")


# ================= 2. 绘制文档-主题概率分布热力图 =================
def generate_doc_topic_heatmap(corpus, lda_model, num_topics, num_docs_to_show=25):
    """
    为了防止文章太多导致图太长，我们默认展示前 25 篇文章的概率分布。
    """
    print("\n正在生成【文档-主题概率分布热力图】...")
    
    # 初始化一个全 0 矩阵 (行=文档数, 列=主题数)
    actual_docs = min(len(corpus), num_docs_to_show)
    doc_topic_matrix = np.zeros((actual_docs, num_topics))
    
    # 填充概率矩阵
    for i in range(actual_docs):
        # minimum_probability=0.0 保证返回所有主题的概率，防止矩阵对不齐
        topic_probs = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        for topic_id, prob in topic_probs:
            doc_topic_matrix[i, topic_id] = prob
            
    # 使用 Seaborn 绘制高颜值热力图
    plt.figure(figsize=(10, 8))
    
    # 设置坐标轴标签
    doc_labels = [f"Doc_{i+1}" for i in range(actual_docs)]
    topic_labels = [f"Topic {i}" for i in range(num_topics)]
    
    # 画图：cmap='YlGnBu' 是一种从黄到蓝渐变的颜色，非常学术
    ax = sns.heatmap(doc_topic_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=topic_labels, yticklabels=doc_labels,
                    cbar_kws={'label': '概率大小 (Probability)'})
    
    plt.title(f"各文档的主题概率分布热力图 (共{actual_docs}篇)", fontsize=16, pad=15)
    plt.xlabel("潜在主题分类", fontsize=12)
    plt.ylabel("文档编号", fontsize=12)
    
    # 把 X 轴的刻度放到上面去，看起来更清爽
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    
    output_image = "LDA_Doc_Topic_Heatmap.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ 概率分布热力图已生成：{output_image}")


# ================= 运行区 =================
if __name__ == "__main__":
    from step4_lda_training import prepare_corpus_for_lda, train_and_display_lda
    
    input_json = 'processed_corpus.json'
    with open(input_json, 'r', encoding='utf-8') as f:
        docs = json.load(f)
        
    texts = prepare_corpus_for_lda(docs)
    
    # 假设你最终确定的是 10 个主题
    NUM_TOPICS = 10 
    lda_model, dictionary, corpus = train_and_display_lda(texts, num_topics=NUM_TOPICS, num_words=15)
    
    # 调用我们的两个终极绘图函数
    generate_topic_wordclouds(lda_model, num_topics=NUM_TOPICS)
    generate_doc_topic_heatmap(corpus, lda_model, num_topics=NUM_TOPICS, num_docs_to_show=len(docs))
    
    print("\n🎉 所有的可视化图表已准备就绪，可以向报告里粘贴了！")