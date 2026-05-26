def discover_new_words_and_update_dict(crf_result_file, dict_file, output_dict_file, top_k=100, min_freq=2):
    """通过对比 CRF 的分词结果和现有词典，提取候选 OOV 并更新词典"""
    # ...（读取原词典与 CRF 结果省略）...
    
    oov_counter = Counter()
    for word in crf_words:
        # 过滤条件：不在原词典中、长度大于1、全是汉字
        if word not in existing_dict and len(word) > 1 and re.match(r'^[\u4e00-\u9fa5]+$', word):
            oov_counter[word] += 1
            
    # 筛选高频新词，过滤掉词频太低的噪音切分
    new_words = [word for word, count in oov_counter.most_common(top_k) if count >= min_freq]
    
    updated_dict = existing_dict.copy()
    updated_dict.update(new_words)
    
    with open(output_dict_file, 'w', encoding='utf-8') as f:
        for w in sorted(updated_dict):
            f.write(w + '\n')