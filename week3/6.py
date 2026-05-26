import re
from collections import Counter

def discover_new_words_and_update_dict(crf_result_file, dict_file, output_dict_file, top_k=100, min_freq=2):
    """
    通过对比 CRF 的分词结果和现有词典，发现新词并更新词典
    """
    print("开始进行迭代一：新词发现...")
    
    # 1. 加载现有词典
    existing_dict = set()
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                existing_dict.add(word)
                
    # 2. 从 CRF 的结果中提取所有词汇
    crf_words = []
    try:
        with open(crf_result_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 按斜杠分割，去除空白
            crf_words = [w.strip() for w in content.split('/') if w.strip()]
    except FileNotFoundError:
        print("找不到 CRF 结果文件，请先运行前面的 CRF 分词。")
        return

    # 3. 统计未登录词 (OOV)
    oov_counter = Counter()
    for word in crf_words:
        # 过滤条件：不在原词典中、长度大于1（单字一般都在词典里了）、全是汉字
        if word not in existing_dict and len(word) > 1 and re.match(r'^[\u4e00-\u9fa5]+$', word):
            oov_counter[word] += 1
            
    # 4. 筛选高频新词
    # 过滤掉词频太低的（可能是 CRF 切错的噪音），取前 top_k 个
    new_words = [word for word, count in oov_counter.most_common(top_k) if count >= min_freq]
    
    print(f"发现了 {len(new_words)} 个高频新词！例如：{new_words[:10]}...")
    
    # 5. 生成新词典
    updated_dict = existing_dict.copy()
    updated_dict.update(new_words)
    
    with open(output_dict_file, 'w', encoding='utf-8') as f:
        for w in sorted(updated_dict):
            f.write(w + '\n')
            
    print(f"迭代完成！新词典已保存至：{output_dict_file}")
    print("👉 下一步：你可以用这个新词典，重新跑一遍 BiMM，看看 F1 值是不是提升了！")

if __name__ == '__main__':
    # 运行此代码前，请确保上一阶段的 result_CRF.txt 已经生成
    discover_new_words_and_update_dict(
        crf_result_file='result_CRF.txt', 
        dict_file='dic_cleaned.txt', 
        output_dict_file='dic_iterated.txt',
        top_k=200, # 可以根据语料大小调整，提取前200个新词
        min_freq=2
    )