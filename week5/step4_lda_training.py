import json
from gensim import corpora
from gensim.models import LdaModel

def prepare_corpus_for_lda(documents, title_weight=2):
    """
    为 LDA 模型准备语料。
    这里依然保留了我们在 TF-IDF 中使用的“标题加权”思想：
    将标题中的词在输入列表中重复出现 title_weight 次，以增加其在生成主题时的比重。
    """
    texts = []
    for doc in documents:
        # 正文词汇直接加入
        doc_words = list(doc["body"])
        # 标题词汇按权重重复加入
        for _ in range(title_weight):
            doc_words.extend(doc["title"])
        texts.append(doc_words)
    return texts

def train_and_display_lda(texts, num_topics=5, num_words=10):
    """
    训练 LDA 模型并打印“主题-语词”分布供人工命名。
    num_topics: 尝试的主题数量（建议从 3 到 5 开始尝试）
    num_words: 每个主题展示的代表性词汇数量
    """
    # 1. 构建词典（将每一个不重复的词映射为一个唯一的 ID）
    dictionary = corpora.Dictionary(texts)
    
    # 过滤掉极低频和极高频的词（可选优化，使得主题更加聚焦）
    # no_below: 至少在 2 篇文档中出现
    # no_above: 在超过 80% 的文档中出现的词会被丢弃（类似停用词）
    dictionary.filter_extremes(no_below=2, no_above=0.8)

    # 2. 构建语料库（将每篇文档转化为词频向量，即 Bag-of-Words）
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    print(f"正在训练 LDA 模型（设置主题数：{num_topics}）...\n")
    # 3. 训练 LDA 模型
    # passes=15 表示算法在整个语料库上迭代的次数，迭代次数越多越稳定
    lda_model = LdaModel(corpus=corpus, 
                        id2word=dictionary, 
                        num_topics=num_topics, 
                        random_state=42, # 固定随机种子，保证每次运行结果一致，方便对比
                        passes=15)
    
    # 4. 提取并展示“主题-语词”分布
    print(f"{'='*20} LDA 模型提取的潜在主题 {'='*20}")
    print("请观察以下每个主题的代表词汇，并记录下你为它们起的名字。\n")
    
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    
    for topic_id, word_probs in topics:
        print(f"【主题 {topic_id}】")
        # 将词和概率分开，为了更直观，我们将概率转换为百分比显示
        word_list = []
        for word, prob in word_probs:
            word_list.append(f"{word}({prob*100:.1f}%)")
        print("  包含词汇: " + ", ".join(word_list))
        print("-" * 60)
        
    return lda_model, dictionary, corpus

# ================= 运行区 =================
if __name__ == "__main__":
    input_json = 'processed_corpus.json'
    
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"找不到 {input_json}，请先运行第一步的数据预处理。")
        exit()

    # 准备文本数据（带标题加权）
    texts = prepare_corpus_for_lda(docs, title_weight=2)
    
    # 老师建议可以多做几次实验，这里我们先默认设置为 5 个主题
    # 你可以修改 num_topics 的值（例如改成 3 或 4）来观察哪种聚类结果在人工看来更合理
    NUM_TOPICS = 10
    NUM_WORDS_TO_SHOW = 15
    
    lda_model, dictionary, corpus = train_and_display_lda(texts, num_topics=NUM_TOPICS, num_words=NUM_WORDS_TO_SHOW)
    
    print("\n💡 下一步操作指引：")
    print("1. 仔细阅读上面输出的各个主题及其包含的词汇。")
    print("2. 尝试用一到两个词来概括每个主题（例如：主题 0 是'文化建设'，主题 1 是'卫生防疫'）。")
    print("3. 如果觉得分得太散或者太杂，可以修改代码中的 NUM_TOPICS 重新运行。")