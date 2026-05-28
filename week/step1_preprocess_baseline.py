import os
import urllib.request
import csv

# --- 配置区 ---
INPUT_FILE = "cnl_202312_check.txt"
OUTPUT_FILE = "baseline_seg_result.csv"  
STOPWORDS_FILE = "hit_stopwords.txt"
STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt"
MIN_BODY_WORDS = 15  # 【新增】正文有效词数低于该值的文章将被丢弃

def load_stopwords():
    """加载停用词表"""
    if not os.path.exists(STOPWORDS_FILE):
        print("未检测到本地哈工大停用词表，正在自动下载...")
        try:
            urllib.request.urlretrieve(STOPWORDS_URL, STOPWORDS_FILE)
            print("下载完成！")
        except Exception as e:
            print(f"下载失败: {e}。请手动下载并放置在同级目录下。")
            return set()

    stopwords = set()
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    
    extra_stops = {'：', '【', '】', '　', ' ', '“', '”', '、', '。', '，', '（', '）', '《', '》', '…', '\n', '\t', '■', '「', '」', '@', '.'}
    stopwords.update(extra_stops)
    return stopwords

def process_baseline_text(input_path, output_path, stopwords):
    if not os.path.exists(input_path):
        print(f"找不到输入文件: {input_path}，请检查路径。")
        return

    print(f"开始按行精细解析文档并过滤短文本 (编码: GB2312) ...")
    
    processed_articles = []
    
    current_title_words = []
    current_body_words = []
    in_body = False
    filtered_count = 0  # 记录被过滤掉的垃圾文本数量

    with open(input_path, 'r', encoding='gb2312', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('标题/：/'):
                # 结算上一篇文章
                if current_title_words or current_body_words:
                    # 【核心修改】判断正文词数是否达到阈值
                    if len(current_body_words) >= MIN_BODY_WORDS:
                        combined = current_title_words + current_body_words
                        processed_articles.append("/".join(combined))
                    else:
                        filtered_count += 1
                
                # 初始化新文章
                current_title_words = []
                current_body_words = []
                in_body = False
                
                title_content = line[len('标题/：/'):]
                words = [w.strip() for w in title_content.split('/') if w.strip() and w.strip() not in stopwords]
                current_title_words.extend(words)
                
            elif line.startswith('/副标题/：/') or line.startswith('/作者/：/') or line.startswith('/日期/：/'):
                in_body = False
                continue
                
            elif line.startswith('/正文/：/'):
                in_body = True
                body_content = line[len('/正文/：/'):]
                if body_content.strip():
                    words = [w.strip() for w in body_content.split('/') if w.strip() and w.strip() not in stopwords]
                    current_body_words.extend(words)
                    
            elif in_body:
                words = [w.strip() for w in line.split('/') if w.strip() and w.strip() not in stopwords]
                current_body_words.extend(words)

    # 结算最后一篇文章
    if current_title_words or current_body_words:
        if len(current_body_words) >= MIN_BODY_WORDS:
            combined = current_title_words + current_body_words
            processed_articles.append("/".join(combined))
        else:
            filtered_count += 1

    print(f"处理完毕！")
    print(f"✅ 成功保留高质量文章: {len(processed_articles)} 篇")
    print(f"🗑️ 过滤掉无正文或过短的噪音文章: {filtered_count} 篇")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['序号', '分词后内容'])
        for idx, text in enumerate(processed_articles, 1):
            writer.writerow([idx, text])
            
    print("清洗与过滤完成，纯净版 CSV 已生成。")

if __name__ == "__main__":
    stops = load_stopwords()
    process_baseline_text(INPUT_FILE, OUTPUT_FILE, stops)