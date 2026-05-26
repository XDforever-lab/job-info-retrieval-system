import re
import os

def build_cleaned_dictionary(input_filepath, output_filepath='dic.txt'):
    print(f"正在读取并清洗分词文件：{input_filepath} ...")
    
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        print("找不到输入文件，请检查路径。")
        return

    # 按斜杠分割
    raw_words = text_content.split('/')
    word_set = set()
    
    punctuation_pattern = re.compile(r'^\W+$')
    # 规则2：匹配纯数字、小数、百分比（例如：123, 8.61, 152%）
    number_pattern = re.compile(r'^\d+(\.\d+)?%?$')
    # 规则3：匹配包含英文和数字的混合无意义串（可选，如果含有CPU这种词汇，这条规则要慎用，这里我们暂时只滤纯数字）
    
    for word in raw_words:
        clean_word = word.strip()
        
        # 如果是空字符串，跳过
        if not clean_word:
            continue
            
        # 如果是纯数字/小数/百分数，跳过
        if number_pattern.match(clean_word):
            continue
            
        # (可选) 过滤长度为1，且很少见的生僻单字？
        # 这里建议保留单字，因为“的、了、是”等单字在逆向最大匹配中是必要的兜底词。
            
        # 通过所有考验，加入词典
        word_set.add(clean_word)
            
    # 写入文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for w in sorted(word_set):
            f.write(w + '\n')
            
    print(f"✅ 清洗版词典构建完成！")
    print(f"共提取了 {len(word_set)} 个有效词汇（已剔除标点和纯数字）。")
    print(f"词典已保存至：{os.path.abspath(output_filepath)}")

if __name__ == '__main__':
    # 替换为你实际的文件路径
    INPUT_FILE = 'raw_expert.txt' 
    OUTPUT_FILE = 'dic_cleaned.txt'
    
    build_cleaned_dictionary(INPUT_FILE, OUTPUT_FILE)