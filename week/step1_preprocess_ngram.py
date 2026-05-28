import os
import re
import csv
import math
import urllib.request
from collections import defaultdict

# ----------------- 配置区 -----------------
INPUT_FILE = "cnl_202312_check.txt"      # 原始语料路径
OUTPUT_FILE = "ngram_seg_result.csv"    # N-gram版本结果
EXPERT_FILE = "C:/Users/ZhangQichen/Desktop/信息检索系统SJ/week5/2/raw_expert.txt" 
STOPWORDS_FILE = "hit_stopwords.txt"
STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt"
MIN_BODY_WORDS = 15                                # 过滤噪音文章的阈值
# ------------------------------------------

class Ngram_Segmenter:
    def __init__(self):
        self.unigram = defaultdict(int)
        self.bigram = defaultdict(int)
        self.total_words = 0
        
    def train(self, expert_filepath):
        if not os.path.exists(expert_filepath):
            print(f"错误：找不到精标语料 {expert_filepath}")
            return False
        
        print("正在训练 N-gram (Bigram) 模型...")
        with open(expert_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取中文词汇
                words = [w for w in line.strip().split('/') if w.strip() and re.match(r'^[\u4e00-\u9fa5]+$', w)]
                for i, word in enumerate(words):
                    self.unigram[word] += 1
                    self.total_words += 1
                    if i > 0:
                        self.bigram[(words[i-1], word)] += 1
        print("N-gram 模型训练完毕！")
        return True

    def get_prob(self, w1, w2):
        """加一平滑的条件概率 P(w2|w1)"""
        count_w1_w2 = self.bigram.get((w1, w2), 0)
        count_w1 = self.unigram.get(w1, 0)
        V = len(self.unigram)  # 词表大小
        return math.log((count_w1_w2 + 1) / (count_w1 + V))

    def segment(self, text):
        """基于最大概率路径的分词（Viterbi算法简化版）"""
        # 预处理文本：只保留汉字
        text = "".join(re.findall(r'[\u4e00-\u9fa5]+', text))
        n = len(text)
        if n == 0: return []
        
        # dp[i] 表示到第i个字符的最大对数概率
        dp = [float('-inf')] * (n + 1)
        dp[0] = 0
        path = [-1] * (n + 1)
        
        # 记录每个位置的前一个词，用于计算 Bigram
        last_word_at_pos = [""] * (n + 1)
        last_word_at_pos[0] = "<S>" 
        
        for i in range(1, n + 1):
            for j in range(max(0, i - 10), i):  # 限制词长为10
                word = text[j:i]
                prev_word = last_word_at_pos[j]
                
                # 如果是单字或者在训练词典中
                if (i - j == 1) or (word in self.unigram):
                    prob = dp[j] + self.get_prob(prev_word, word)
                    if prob > dp[i]:
                        dp[i] = prob
                        path[i] = j
                        last_word_at_pos[i] = word
                        
        result = []
        curr = n
        while curr > 0:
            prev = path[curr]
            result.insert(0, text[prev:curr])
            curr = prev
        return result

def load_stopwords():
    """加载停用词表"""
    if not os.path.exists(STOPWORDS_FILE):
        print("正在自动下载停用词表...")
        urllib.request.urlretrieve(STOPWORDS_URL, STOPWORDS_FILE)

    stopwords = set()
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f: stopwords.add(line.strip())
    
    extra_stops = {'：', '【', '】', '　', ' ', '“', '”', '、', '。', '，', '（', '）', '《', '》', '…', '\n', '\t', '■', '「', '」'}
    stopwords.update(extra_stops)
    return stopwords

def main():
    # 1. 模型初始化与训练
    ngram = Ngram_Segmenter()
    if not ngram.train(EXPERT_FILE):
        return
    stopwords = load_stopwords()

    # 2. 解析原始语料（合并标题+正文，去除噪音）
    print(f"正在读取并清洗原始语料: {INPUT_FILE}")
    processed_raw_texts = []
    
    current_title_raw = ""
    current_body_raw = ""
    in_body = False
    
    with open(INPUT_FILE, 'r', encoding='gb2312', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('标题/：/'):
                # 结算前一篇文章
                if current_title_raw or current_body_raw:
                    clean_body = current_body_raw.replace('/', '')
                    # 噪音过滤逻辑：正文汉字数需达到一定长度
                    if len(re.findall(r'[\u4e00-\u9fa5]', clean_body)) > MIN_BODY_WORDS * 2:
                        combined = current_title_raw.replace('/', '') + clean_body
                        processed_raw_texts.append(combined)

                current_title_raw = line[len('标题/：/'):]
                current_body_raw = ""
                in_body = False
                
            elif line.startswith('/副标题/：/') or line.startswith('/作者/：/') or line.startswith('/日期/：/'):
                in_body = False
            elif line.startswith('/正文/：/'):
                in_body = True
                current_body_raw += line[len('/正文/：/'):]
            elif in_body:
                current_body_raw += line

    # 结算最后一篇
    if current_title_raw or current_body_raw:
        clean_body = current_body_raw.replace('/', '')
        if len(re.findall(r'[\u4e00-\u9fa5]', clean_body)) > MIN_BODY_WORDS * 2:
            processed_raw_texts.append(current_title_raw.replace('/', '') + clean_body)

    # 3. 执行 N-gram 分词并导出
    print(f"正在进行 N-gram (Bigram) 分词处理...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['序号', '分词后内容'])
        
        for idx, text in enumerate(processed_raw_texts, 1):
            words = ngram.segment(text)
            filtered = [w for w in words if w not in stopwords]
            writer.writerow([idx, "/".join(filtered)])

    print(f"✅ N-gram 处理完成！共保留 {len(processed_raw_texts)} 篇。")
    print(f"结果保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()