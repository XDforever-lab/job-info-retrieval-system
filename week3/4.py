import os
import re

# ==========================================
# 1. 核心算法：严谨修复版的集合元素化评测法
# ==========================================
def eval_by_set_elements(machine_seg, expert_seg):
    """
    计算 精准率P, 召回率R, F测度值
    """
    def get_intervals(seg_text):
        intervals = set()
        # 切分文本
        words = seg_text.split('/')
        current_idx = 1
        for w in words:
            # 🚨 核心修复：不能只用 strip()！
            # 必须使用与 clean_text_for_alignment 完全一致的清洗规则
            # 剔除词块内部可能因为漏分而包裹进去的任何空白符和换行符
            clean_w = re.sub(r'\s+', '', w) 
            
            if not clean_w:
                continue
                
            length = len(clean_w)
            intervals.add((current_idx, current_idx + length - 1))
            current_idx += length
            
        return intervals

    machine_set = get_intervals(machine_seg)
    expert_set = get_intervals(expert_seg)

    overlap = machine_set.intersection(expert_set)

    P = len(overlap) / len(machine_set) if machine_set else 0
    R = len(overlap) / len(expert_set) if expert_set else 0
    F = (2 * P * R) / (P + R) if (P + R) > 0 else 0

    return P, R, F

# ==========================================
# 2. 对齐检查与自动化评测工作台
# ==========================================
def clean_text_for_alignment(text):
    """
    为了严格比对底层字符，剔除所有分词符、空格、换行符、制表符等
    """
    # 替换分词符
    text = text.replace('/', '')
    # 替换所有空白字符（包括空格、\n, \t, \r 以及全角空格）
    text = re.sub(r'\s+', '', text) 
    return text

def check_and_evaluate(expert_file, test_files):
    print(f"正在加载标准答案文件: {expert_file}")
    try:
        with open(expert_file, 'r', encoding='utf-8') as f:
            expert_content = f.read()
    except FileNotFoundError:
        print("❌ 找不到标准答案文件，请检查路径。")
        return

    expert_raw = clean_text_for_alignment(expert_content)
    
    print("-" * 60)
    print(f"【阶段一：底层纯文本对齐校验】 (标准答案纯文本长度: {len(expert_raw)})")
    print("-" * 60)
    
    passed_files = {}

    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_content = f.read()
        except FileNotFoundError:
            print(f"⚠️ 找不到文件 {test_file}，已跳过。")
            continue

        test_raw = clean_text_for_alignment(test_content)
        
        if test_raw == expert_raw:
            print(f"✅ [{test_file}] 底层字符验证通过！")
            passed_files[test_file] = test_content
        else:
            print(f"❌ [{test_file}] 对齐失败！")
            print(f"   测试文件长度: {len(test_raw)} vs 标准长度: {len(expert_raw)}")
            
            min_len = min(len(test_raw), len(expert_raw))
            for i in range(min_len):
                if test_raw[i] != expert_raw[i]:
                    start = max(0, i - 15)
                    end_t = min(len(test_raw), i + 15)
                    end_e = min(len(expert_raw), i + 15)
                    
                    print(f"   【冲突点】第 {i} 个字符：")
                    print(f"   -> 测试文件上下文: ...{test_raw[start:end_t]}...")
                    print(f"   -> 标准答案上下文: ...{expert_raw[start:end_e]}...")
                    print(f"   -> 冲突字符对比 : 测试为 '{test_raw[i]}' vs 标准为 '{expert_raw[i]}'\n")
                    break

    print("\n" + "=" * 60)
    print("【阶段二：量化指标评测 (修复索引偏移版)】")
    print("=" * 60)
    
    if not passed_files:
        print("🚨 没有文件通过对齐校验，请修改原 txt 后再来执行！")
        return

    # 表头
    print(f"{'模型/方法':<15} | {'精准率 (P)':<12} | {'召回率 (R)':<12} | {'F1 测度 (F1)':<12}")
    print("-" * 60)
    
    for filename, test_content in passed_files.items():
        p, r, f1 = eval_by_set_elements(test_content, expert_content)
        
        model_name = filename.replace('result_', '').replace('.txt', '')
        print(f"{model_name:<15} | {p*100:>8.2f} %  | {r*100:>8.2f} %  | {f1*100:>8.2f} %")

if __name__ == '__main__':
    EXPERT_FILE = 'expert_standard_answer.txt' 
    
    TEST_FILES = [
        "result_FMM.txt",
        "result_BMM.txt",
        "result_BiMM.txt",
        "result_BiMM1.txt",
        "result_MinSeg.txt",
        "result_HMM.txt",
        "result_Ngram.txt",
        "result_CRF.txt",
        "result_CRF_Pro.txt"
    ]
    
    check_and_evaluate(EXPERT_FILE, TEST_FILES)