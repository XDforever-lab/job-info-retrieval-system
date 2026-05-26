import math
import pycrfsuite
import sklearn_crfsuite
from collections import defaultdict
import re

# ==========================================
# 辅助函数：文本预处理
# ==========================================
def split_text_blocks(text):
    """提取汉字块进行分词，非汉字块保留"""
    blocks = re.split(r'([\u4e00-\u9fa5]+)', text)
    return [b for b in blocks if b]

def get_word_tags(word):
    """将词语转化为 BIES 标签序列"""
    if len(word) == 1:
        return ['S']
    elif len(word) == 2:
        return ['B', 'E']
    else:
        return ['B'] + ['I'] * (len(word) - 2) + ['E']

# ==========================================
# 1. 隐马尔可夫模型 (HMM) - 纯手写 Viterbi
# ==========================================
class HMM_Segmenter:
    def __init__(self):
        self.states = ['B', 'I', 'E', 'S']
        # 初始化概率矩阵（取对数防止下溢，初始值为负无穷）
        self.start_p = {s: float('-inf') for s in self.states}
        self.trans_p = {s: {s2: float('-inf') for s2 in self.states} for s in self.states}
        self.emit_p = {s: defaultdict(lambda: float('-inf')) for s in self.states}

    def train(self, expert_filepath):
        print("正在训练 HMM 模型...")
        state_count = {s: 0 for s in self.states}
        start_count = {s: 0 for s in self.states}
        trans_count = {s: {s2: 0 for s2 in self.states} for s in self.states}
        
        with open(expert_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                words = [w for w in line.split('/') if w.strip() and re.match(r'^[\u4e00-\u9fa5]+$', w)]
                if not words: continue
                
                tags = []
                for word in words:
                    tags.extend(get_word_tags(word))
                chars = "".join(words)
                
                if not chars: continue
                
                # 统计初始概率
                start_count[tags[0]] += 1
                
                # 统计转移和发射概率
                for i in range(len(tags)):
                    state_count[tags[i]] += 1
                    self.emit_p[tags[i]][chars[i]] = self.emit_p[tags[i]].get(chars[i], 0) + 1
                    if i > 0:
                        trans_count[tags[i-1]][tags[i]] += 1

        # 转换为对数概率
        total_start = sum(start_count.values())
        if total_start > 0:
            for s in self.states:
                if start_count[s] > 0: self.start_p[s] = math.log(start_count[s] / total_start)
                
        for s1 in self.states:
            for s2 in self.states:
                if trans_count[s1][s2] > 0:
                    self.trans_p[s1][s2] = math.log(trans_count[s1][s2] / state_count[s1])
            
            for char, count in self.emit_p[s1].items():
                self.emit_p[s1][char] = math.log(count / state_count[s1])
                
        # 设置一个极小值作为平滑（处理未登录字）
        self.min_prob = -3.14e+100

    def viterbi(self, obs):
        V = [{}]
        path = {}

        # 初始化 t=0
        for y in self.states:
            V[0][y] = self.start_p.get(y, self.min_prob) + self.emit_p[y].get(obs[0], self.min_prob)
            path[y] = [y]

        # 动态规划 t>0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max(
                    (V[t - 1][y0] + self.trans_p[y0].get(y, self.min_prob) + self.emit_p[y].get(obs[t], self.min_prob), y0)
                    for y0 in self.states
                )
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath

        (prob, state) = max((V[len(obs) - 1][y], y) for y in ['E', 'S'])
        return path[state]

    def segment(self, text):
        if not text: return []
        tags = self.viterbi(text)
        result = []
        word = ""
        for char, tag in zip(text, tags):
            word += char
            if tag in ('E', 'S'):
                result.append(word)
                word = ""
        return result

# ==========================================
# 2. 简单的 N-gram (Bigram) 模型
# ==========================================
class Ngram_Segmenter:
    def __init__(self):
        self.unigram = defaultdict(int)
        self.bigram = defaultdict(int)
        self.total_words = 0
        
    def train(self, expert_filepath):
        print("正在训练 N-gram (Bigram) 模型...")
        with open(expert_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = [w for w in line.strip().split('/') if w.strip() and re.match(r'^[\u4e00-\u9fa5]+$', w)]
                for i, word in enumerate(words):
                    self.unigram[word] += 1
                    self.total_words += 1
                    if i > 0:
                        self.bigram[(words[i-1], word)] += 1

    def get_prob(self, w1, w2):
        """加一平滑的条件概率 P(w2|w1)"""
        count_w1_w2 = self.bigram.get((w1, w2), 0)
        count_w1 = self.unigram.get(w1, 0)
        V = len(self.unigram) # 词表大小
        return math.log((count_w1_w2 + 1) / (count_w1 + V))

    def get_unigram_prob(self, w):
        return math.log((self.unigram.get(w, 0) + 1) / (self.total_words + len(self.unigram)))

    def segment(self, text):
        """基于最大概率路径的分词（简化版 DAG 寻找最短路）"""
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
            for j in range(max(0, i - 10), i): # 假设词长不超过10
                word = text[j:i]
                prev_word = last_word_at_pos[j]
                
                # 如果是单字或者在词典中
                if (i - j == 1) or (word in self.unigram):
                    # Bigram 概率
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

# ==========================================
# 3. 条件随机场 (CRF) - 基于 sklearn-crfsuite
# ==========================================
class CRF_Segmenter:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
    def char2features(self, text, i):
        """提取字级别的特征"""
        char = text[i]
        features = {
            'bias': 1.0,
            'char': char,
        }
        if i > 0:
            features.update({'char-1': text[i-1], 'char-1:0': text[i-1] + char})
        else:
            features['BOS'] = True
            
        if i < len(text) - 1:
            features.update({'char+1': text[i+1], 'char0:+1': char + text[i+1]})
        else:
            features['EOS'] = True
        return features

    def train(self, expert_filepath):
        print("正在训练 CRF 模型 (这可能需要几十秒时间)...")
        X_train, y_train = [], []
        with open(expert_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = [w for w in line.strip().split('/') if w.strip() and re.match(r'^[\u4e00-\u9fa5]+$', w)]
                if not words: continue
                tags = []
                for w in words: tags.extend(get_word_tags(w))
                chars = "".join(words)
                
                if chars:
                    X_train.append([self.char2features(chars, i) for i in range(len(chars))])
                    y_train.append(tags)
                    
        if X_train:
            self.model.fit(X_train, y_train)

    def segment(self, text):
        if not text: return []
        features = [self.char2features(text, i) for i in range(len(text))]
        tags = self.model.predict_single(features)
        
        result = []
        word = ""
        for char, tag in zip(text, tags):
            word += char
            if tag in ('E', 'S'):
                result.append(word)
                word = ""
        return result

# ==========================================
# 主流程控制
# ==========================================
def process_statistical(expert_filepath, raw_filepath):
    # 初始化并训练三个模型
    hmm = HMM_Segmenter()
    hmm.train(expert_filepath)
    
    ngram = Ngram_Segmenter()
    ngram.train(expert_filepath)
    
    crf = CRF_Segmenter()
    crf.train(expert_filepath)
    
    print("\n模型训练完毕，开始预测生语料...")
    
    # 预测并写入文件
    output_files = {
        "HMM": ("result_HMM.txt", hmm),
        "NGRAM": ("result_Ngram.txt", ngram),
        "CRF": ("result_CRF.txt", crf)
    }
    
    try:
        with open(raw_filepath, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        print("找不到生语料文件。")
        return

    # 为每个模型生成结果
    for model_name, (filename, model) in output_files.items():
        results = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                results.append("\n")
                continue
                
            line_result = []
            blocks = split_text_blocks(line)
            
            for block in blocks:
                if re.match(r'^[\u4e00-\u9fa5]+$', block):
                    line_result.extend(model.segment(block))
                else:
                    line_result.append(block)
                    
            results.append("/".join(line_result) + "/")
            
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))
        print(f"[{model_name}] 分词完成，已生成文件: {filename}")

if __name__ == '__main__':
    EXPERT_FILE = 'raw_expert.txt' # 替换为你的上周精标文件路径
    RAW_FILE = '19123222_张琪琛.txt'          # 替换为生语料路径
    
    process_statistical(EXPERT_FILE, RAW_FILE)