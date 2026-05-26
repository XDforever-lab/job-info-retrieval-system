import json

def read_keyword_file(filename):
    """读取关键词txt文件，返回二维列表"""
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析 "Doc_1: 词1, 词2..." 格式
            if ":" in line:
                words_str = line.strip().split(': ', 1)[1]
                words = words_str.split(', ')
                results.append(words)
    return results

# ================= 运行区 =================
if __name__ == "__main__":
    # 1. 加载所有数据
    print("正在加载语料与算法提取结果...")
    try:
        with open('processed_corpus.json', 'r', encoding='utf-8') as f:
            original_docs = json.load(f)
    except FileNotFoundError:
        print("找不到 processed_corpus.json，请检查路径。")
        exit()

    tfidf_res = read_keyword_file('final_tfidf_keywords_K15.txt')
    textrank_res = read_keyword_file('final_textrank_keywords_K15.txt')
    lda_res = read_keyword_file('final_lda_keywords_K15.txt')

    # 老师说：选20个就行，不需要全部做
    SAMPLE_SIZE = 37 
    output_filename = 'llm_judge_tasks.txt'

    with open(output_filename, 'w', encoding='utf-8') as out_f:
        # 添加给大模型的 System Prompt（设定它的身份）
        out_f.write("【全局任务指令】\n")
        out_f.write("你现在是一个专业的信息检索与自然语言处理专家。我将为你提供多篇新闻报道的原文，以及一个通过算法提取出的【候选关键词池】。\n")
        out_f.write("请你阅读原文，从【候选关键词池】中筛选出真正能概括文章核心主旨的词汇，剔除那些停用词、通用词或边缘词。\n")
        out_f.write("请直接返回你确定的关键词（词数不限，宁缺毋滥），多个词之间用逗号隔开，不要输出解释性废话。\n\n")
        out_f.write("=" * 60 + "\n\n")

        for i in range(min(SAMPLE_SIZE, len(original_docs))):
            # 获取当前篇的原始正文和标题
            doc = original_docs[i]
            # 为了让大模型阅读通顺，我们将列表拼接成字符串
            full_text = "".join(doc["title"]) + "。" + "".join(doc["body"])
            
            # 核心步骤：将三种算法的结果进行 并集去重 操作
            pool_set = set()
            pool_set.update(tfidf_res[i])
            pool_set.update(textrank_res[i])
            pool_set.update(lda_res[i])
            
            # 过滤掉空字符串
            candidate_pool = [word for word in pool_set if word]

            # 格式化写入文本
            out_f.write(f"### 【文档 {i+1}】\n")
            out_f.write(f"【原文内容】: {full_text}\n")
            out_f.write(f"【候选关键词池】: {', '.join(candidate_pool)}\n")
            out_f.write(f"【专家筛选输出】: \n\n")
            out_f.write("-" * 40 + "\n\n")

    print(f"\n✅ 成功聚合了 {SAMPLE_SIZE} 篇文档的测试集！")
    print(f"数据已格式化保存至：{output_filename}")