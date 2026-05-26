import re
import os

# ==========================================
# 1. 词典加载与文本分块工具
# ==========================================

def load_dictionary(dict_path):
    """加载上一步清洗好的词典，并获取词典中的最大词长"""
    word_dict = set()
    max_word_len = 0
    print(f"正在加载词典: {dict_path} ...")
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    word_dict.add(word)
                    if len(word) > max_word_len:
                        max_word_len = len(word)
    except FileNotFoundError:
        print(f"错误：找不到词典文件 {dict_path}")
        return None, 0
    print(f"词典加载成功，共 {len(word_dict)} 个词，最大词长为 {max_word_len}。")
    return word_dict, max_word_len

def split_text_blocks(text):
    """将文本切分为中文字块和非中文字块（数字、英文、标点等）"""
    # 使用正则按连续的汉字进行分割，保留分隔符
    blocks = re.split(r'([\u4e00-\u9fa5]+)', text)
    # 过滤掉空字符串
    return [b for b in blocks if b]

# ==========================================
# 2. 四种核心分词算法实现
# ==========================================

def forward_maximum_matching(text, word_dict, max_len):
    """正向最大匹配算法 (FMM)"""
    result = []
    text_len = len(text)
    start = 0
    while start < text_len:
        # 取当前位置到最大词长的子串
        window_size = min(max_len, text_len - start)
        matched = False
        
        while window_size > 0:
            current_word = text[start:start + window_size]
            # 如果在词典中，或者是单字，则切分
            if current_word in word_dict or window_size == 1:
                result.append(current_word)
                start += window_size
                matched = True
                break
            window_size -= 1
            
        if not matched: # 理论上不会走到这里，因为 window_size==1 时必定 match
            result.append(text[start])
            start += 1
    return result

def backward_maximum_matching(text, word_dict, max_len):
    """逆向最大匹配算法 (BMM)"""
    result = []
    text_len = len(text)
    end = text_len
    
    while end > 0:
        window_size = min(max_len, end)
        matched = False
        
        while window_size > 0:
            start = end - window_size
            current_word = text[start:end]
            if current_word in word_dict or window_size == 1:
                result.insert(0, current_word) # 逆向匹配，结果插到头部
                end -= window_size
                matched = True
                break
            window_size -= 1
            
    return result

def bidirectional_matching(text, word_dict, max_len):
    """双向匹配法 (BiMM)"""
    fmm_result = forward_maximum_matching(text, word_dict, max_len)
    bmm_result = backward_maximum_matching(text, word_dict, max_len)
    
    # 规则1：词数不同，取词数较少的
    if len(fmm_result) != len(bmm_result):
        return fmm_result if len(fmm_result) < len(bmm_result) else bmm_result
    
    # 规则2：词数相同，取单字较少的
    fmm_single_chars = sum(1 for w in fmm_result if len(w) == 1)
    bmm_single_chars = sum(1 for w in bmm_result if len(w) == 1)
    
    if fmm_single_chars != bmm_single_chars:
        return fmm_result if fmm_single_chars < bmm_single_chars else bmm_result
        
    # 规则3：完全相同或单字数相同，根据统计逆向匹配通常精度略高，返回逆向结果
    return bmm_result

def minimum_segmentation(text, word_dict, max_len):
    """最少切分算法（基于动态规划）"""
    n = len(text)
    if n == 0: return []
    
    # dp[i] 表示 text[0:i] 的最少切分词数
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    # path[i] 记录 dp[i] 取得最小值时，最后一个词的起始位置
    path = [-1] * (n + 1)
    
    for i in range(1, n + 1):
        # 考察所有可能的最后一个词 text[j:i]
        start_j = max(0, i - max_len)
        for j in range(start_j, i):
            word = text[j:i]
            # 如果是单字，或者是一个合法的词
            if (i - j == 1) or (word in word_dict):
                if dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    path[i] = j
                    
    # 回溯寻找路径
    result = []
    curr = n
    while curr > 0:
        prev = path[curr]
        result.insert(0, text[prev:curr])
        curr = prev
        
    return result

# ==========================================
# 3. 语料文件处理主流程
# ==========================================

def process_corpus(raw_filepath, dict_filepath):
    word_dict, max_len = load_dictionary(dict_filepath)
    if not word_dict:
        return

    print(f"开始处理生语料: {raw_filepath} ...")
    
    # 定义四种方法的输出文件
    output_files = {
        "FMM": "result_FMM1.txt",
        "BMM": "result_BMM1.txt",
        "BiMM": "result_BiMM1.txt",
        "MinSeg": "result_MinSeg1.txt"
    }
    
    # 初始化四个列表存放结果
    results = {k: [] for k in output_files.keys()}
    
    try:
        with open(raw_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到生语料文件 {raw_filepath}")
        return

    for line in lines:
        line = line.strip()
        if not line:
            for k in results: results[k].append("\n")
            continue
            
        # 对每行分别初始化四种方法的临时结果
        line_results = {k: [] for k in results.keys()}
        
        # 分块处理：区分中文字块和非中文字块
        blocks = split_text_blocks(line)
        
        for block in blocks:
            # 如果是中文字块，进行机械分词
            if re.match(r'^[\u4e00-\u9fa5]+$', block):
                line_results["FMM"].extend(forward_maximum_matching(block, word_dict, max_len))
                line_results["BMM"].extend(backward_maximum_matching(block, word_dict, max_len))
                line_results["BiMM"].extend(bidirectional_matching(block, word_dict, max_len))
                line_results["MinSeg"].extend(minimum_segmentation(block, word_dict, max_len))
            else:
                # 非中文字块（如标点、日期、英文等），直接作为一个整体或按需处理
                # 这里为了格式清晰，统一视为一个 token
                for k in line_results: line_results[k].append(block)
                
        # 将切分后的词用 '/' 连接
        for k in results:
            results[k].append("/".join(line_results[k]) + "/")

    # 将结果写入对应的文件
    for k, filename in output_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(results[k]))
        print(f"已生成文件: {os.path.abspath(filename)}")
        
    print("四种分词方法处理完成！")

if __name__ == '__main__':
    # 请确保这里的文件名与你实际的文件名一致
    # 第一步生成的纯净词典
    DICTIONARY_FILE = 'dic_iterated.txt' 
    # 本周的生语料
    RAW_CORPUS_FILE = '19123222_张琪琛.txt'  
    
    process_corpus(RAW_CORPUS_FILE, DICTIONARY_FILE)