import os

def read_keywords(filename):
    """通用的文件读取函数，返回一个字典 {文档索引: 词汇集合}"""
    results = {}
    if not os.path.exists(filename):
        print(f"⚠️ 警告: 找不到文件 {filename}")
        return results
        
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ":" in line:
                doc_str, words_str = line.strip().split(':', 1)
                # 提取 Doc_1 中的数字 1，转化为索引 0
                doc_idx = int(doc_str.replace("Doc_", "").strip()) - 1
                # 清洗并存储词汇
                words = {w.strip() for w in words_str.split(',') if w.strip()}
                results[doc_idx] = words
    return results

if __name__ == "__main__":
    # 1. 加载三个算法的原始候选池 (K=15)
    tfidf_res = read_keywords('final_tfidf_keywords_K15.txt')
    textrank_res = read_keywords('final_textrank_keywords_K15.txt')
    lda_res = read_keywords('final_lda_keywords_K15.txt')
    
    # 2. 加载大模型生成的粗糙答案
    llm_judgement = read_keywords('llm_raw_judgement.txt')
    
    output_filename = 'final_standard_answers.txt'
    total_hallucinations = 0
    
    print("开始进行大模型防幻觉交叉校验...\n")
    
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        for doc_idx, llm_words in sorted(llm_judgement.items()):
            # 拿到三个算法在这个文档的输出，构建合法候选池并集
            pool_set = set()
            pool_set.update(tfidf_res.get(doc_idx, set()))
            pool_set.update(textrank_res.get(doc_idx, set()))
            pool_set.update(lda_res.get(doc_idx, set()))
            
            # 如果某篇文档三个算法都没出结果（几乎不可能），跳过
            if not pool_set:
                out_f.write(f"Doc_{doc_idx+1}: \n")
                continue
                
            valid_words = []
            hallucinated_words = []
            
            # 校验大模型给出的每一个词
            for word in llm_words:
                if word in pool_set:
                    valid_words.append(word)
                else:
                    hallucinated_words.append(word)
            
            # 如果发现了幻觉，在控制台打印日志方便你写进报告
            if hallucinated_words:
                print(f"【Doc_{doc_idx+1}】发现并剔除幻觉词: {hallucinated_words}")
                total_hallucinations += len(hallucinated_words)
                
            # 将清洗后的纯净合法词汇写入最终标准答案文件
            # 保持大模型原本的顺序逻辑，这里不重新打乱顺序
            out_f.write(f"Doc_{doc_idx+1}: {', '.join(valid_words)}\n")

    print(f"\n✅ 校验完成！共拦截并剔除 {total_hallucinations} 个幻觉词。")
    print(f"✅ 纯净的最终版标准答案池已生成：{output_filename}")