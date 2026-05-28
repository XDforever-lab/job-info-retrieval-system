import os
import re
import csv
import urllib.request
import sklearn_crfsuite

# ----------------- 配置区 -----------------
INPUT_FILE = "cnl_202312_check.txt"      # 原始语料路径
OUTPUT_FILE = "crf_seg_result.csv"      # CRF版本结果
EXPERT_FILE = "C:/Users/ZhangQichen/Desktop/信息检索系统SJ/week5/2/raw_expert.txt"                  # 你的精标训练语料
STOPWORDS_FILE = "hit_stopwords.txt"
STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt"
MIN_BODY_WORDS = 15                                # 过滤噪音文章的阈值
# ------------------------------------------

def get_word_tags(word):
    """为词语生成 B/M/E/S 标签"""
    if len(word) == 1:
        return ['S']
    elif len(word) == 2:
        return ['B', 'E']
    else:
        return ['B'] + ['M'] * (len(word) - 2) + ['E']

class CRF_Segmenter:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs', c1=0.1, c2=0.1,
            max_iterations=100, all_possible_transitions=True
        )
        
    def char2features(self, text, i):
        char = text[i]
        features = {'bias': 1.0, 'char': char}
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
        if not os.path.exists(expert_filepath):
            print(f"错误：找不到精标语料 {expert_filepath}，无法训练CRF模型。")
            return False
        
        print("正在根据精标语料训练 CRF 模型...")
        X_train, y_train = [], []
        with open(expert_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取纯中文词汇
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
            print("CRF 模型训练完毕！")
            return True
        return False

    def segment(self, text):
        if not text: return []
        # 只处理中文，过滤掉其他干扰字符
        clean_text = "".join(re.findall(r'[\u4e00-\u9fa5]+', text))
        if not clean_text: return []
        
        features = [self.char2features(clean_text, i) for i in range(len(clean_text))]
        tags = self.model.predict_single(features)
        
        result = []
        word = ""
        for char, tag in zip(clean_text, tags):
            word += char
            if tag in ('E', 'S'):
                result.append(word)
                word = ""
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
    # 1. 准备分词器和停用词
    crf = CRF_Segmenter()
    if not crf.train(EXPERT_FILE):
        return
    stopwords = load_stopwords()

    # 2. 解析原始语料 (合并标题+正文，去除噪音)
    print(f"正在读取原始语料: {INPUT_FILE}")
    processed_texts = []
    
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
                    # 去掉原始语料中的斜杠，还原成文本进行分词
                    clean_body = current_body_raw.replace('/', '')
                    # 检查有效汉字长度是否达标 (近似词数过滤)
                    if len(re.findall(r'[\u4e00-\u9fa5]', clean_body)) > MIN_BODY_WORDS * 2:
                        combined_raw = current_title_raw.replace('/', '') + clean_body
                        processed_texts.append(combined_raw)

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
            processed_texts.append(current_title_raw.replace('/', '') + clean_body)

    # 3. 使用 CRF 批量分词并写入 CSV
    print(f"正在使用 CRF 模型分词并导出 CSV...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['序号', '分词后内容'])
        
        for idx, raw_text in enumerate(processed_texts, 1):
            # 执行分词
            words = crf.segment(raw_text)
            # 去停用词
            filtered = [w for w in words if w not in stopwords]
            writer.writerow([idx, "/".join(filtered)])

    print(f"✅ CRF 处理完成！共保留 {len(processed_texts)} 篇文章。")
    print(f"文件保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()