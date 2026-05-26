import os
import json
import re

def load_custom_stopwords():
    """
    构建全面且定制化的停用词表。
    """
    base_stopwords = {
        '：', '【', '】', '（', '）', '，', '。', '、', '；', '“', '”', '《', '》', '！', '？', '—', '-', '.', ',', '?', '!', '——',
        '我', '我们', '你', '你们', '他', '她', '它', '他们', '它们', '这', '这个', '这些', '那', '那个', '那些', '其', '其中', '某',
        '的', '和', '了', '在', '是', '等', '也', '就', '与', '对', '及', '关于', '向', '指出', '被', '从', '由', '到', '为', '以', '把', '将', '给', '比', '或', '或者', '并且', '而且', '虽然', '但是', '如果', '那么', '因为', '所以', '得', '着', '过',
        '很', '非常', '十分', '极其', '最', '更', '更加', '已经', '曾经', '刚', '刚刚', '正', '正在', '将要', '才', '却', '倒', '几乎', '大约', '大概', '也许', '可能', '一定', '必须', '都', '全', '只', '仅仅', '还', '再', '不', '没', '别'
    }
    
    news_metadata_words = {
        '标题', '副标题', '副', '作者', '日期', '正文', '本报记者', '本报', '电', '记者', 
        '人民日报', '版', '要闻', '社会', '综合', '年', '月', '日', '新华社', '北京'
    }
    
    quantifiers_and_numbers = {
        '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿', '余', '多',
        '个', '名', '项', '届', '批', '次', '家', '例', '条', '部', '场', '位', '篇', '类', '元', '吨', '米'
    }
    
    adverbs_and_directions = {
        '新', '高', '大', '中', '下', '上', '内', '外', '各', '各类', '各级', '各地', 
        '少', '好', '进一步', '充分', '不断', '显著', '日益', '逐渐', '切实', '相', 
        '本', '该', '以此', '通过', '根据', '由于', '为了', '有关', '相关', '分别', '即', 
        '让', '当', '比如', '此外', '牵头', '用于', '以及', '不仅'
    }

    # 【修复 Bug】：补齐了你代码中缺失的这两个变量定义，增加政务新闻常见高频无意义动词
    official_verbs = {
        '有', '没有', '存在', '出现', '发生', '进行', '表示', '认为', '目前', '当前', '下一步', 
        '要求', '强调', '提出', '发现', '看到', '问题', '情况', '方面', '工作', '会议', '报告', 
        '通知', '规定', '办法', '条例', '总的看', '指出', '发展', '建设', '实现'
    }
    
    news_extra_words = set() 

    extra_punctuations = {
        '……', '‘', '’', '■', '+', '·', '—'
    }
    
    return base_stopwords.union(
        news_metadata_words, 
        quantifiers_and_numbers, 
        adverbs_and_directions,
        official_verbs,
        news_extra_words,
        extra_punctuations
    )

def is_number_or_percentage(s):
    """判断字符串是否为纯数字或百分比"""
    return bool(re.match(r'^[\d\.]+%?$', s))

def preprocess_and_save(input_file_path, output_file_path):
    stopwords = load_custom_stopwords()
    documents = []          
    
    # 使用字典结构来分别存储一篇文章的标题词和正文词
    current_doc = {"title": [], "body": []}
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 遇到新文章的标题行
            if line.startswith('标题/：/'):
                # 如果 current_doc 的 body 里有数据，说明上一篇文章读完了，先存起来
                if current_doc["body"]:
                    documents.append(current_doc)
                    # 重置当前文档结构，准备装新文章
                    current_doc = {"title": [], "body": []}
                
                # 提取并清洗标题词汇
                words = line.split('/')
                cleaned_words = [w for w in words if w and w not in stopwords and not is_number_or_percentage(w)]
                current_doc["title"].extend(cleaned_words)
                
            # 过滤不需要的元数据行
            elif line.startswith('/副/标题/：/') or line.startswith('/作者/：/') or line.startswith('/日期/：/'):
                continue
                
            # 处理正文（将其装入字典的 "body" 列表下）
            elif line.startswith('/正文/：/') or len(current_doc["title"]) > 0:
                words = line.split('/')
                cleaned_words = [w for w in words if w and w not in stopwords and not is_number_or_percentage(w)]
                current_doc["body"].extend(cleaned_words)
                
        # 循环结束，保存最后一篇文章
        if current_doc["body"] or current_doc["title"]:
            documents.append(current_doc)
            
    # ===== 将结构化数据保存到 JSON 文件 =====
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
        
    print(f"清洗完成！共处理了 {len(documents)} 篇文档。")
    print(f"结果已成功保存到：{os.path.abspath(output_file_path)}")
    
    return documents

if __name__ == "__main__":
    input_txt = 'raw_expert.txt'   # 你的原始数据文件
    output_json = 'processed_corpus.json' 
    
    if os.path.exists(input_txt):
        docs = preprocess_and_save(input_txt, output_json)
        # 打印第一篇看看结构是否正确
        if docs:
            print("\n第一篇文档的结构预览：")
            print("【标题词】:", docs[0]["title"][:10])
            print("【正文词】:", docs[0]["body"][:10])
    else:
        print(f"找不到文件 {input_txt}，请检查路径。")