import os
import re

def get_intervals_and_boundaries(text):
    """
    还原纯净字符流，记录每个词的物理区间和边界坐标
    """
    # 彻底移除所有空白符，防止拉链错位
    text = re.sub(r'\s+', '', text)
    words = [w for w in text.split('/') if w]
    
    intervals = []
    boundaries = set()
    curr = 0
    for w in words:
        start = curr
        end = curr + len(w)
        intervals.append((start, end, w))
        boundaries.add(end) # 记录斜杠所在的物理坐标
        curr = end
        
    return intervals, boundaries, curr

def reconstruct_snippet(intervals, start_char, end_char):
    """
    根据物理坐标，反向拼接出带有斜杠的文本片段
    """
    words = []
    for s, e, w in intervals:
        # 只要词的区间与我们要提取的窗口有交集，就提取出来
        if e > start_char and s < end_char:
            words.append(w)
    return "/".join(words)

def extract_absolute_bad_cases(expert_file, test_files, output_file="bad_cases_for_llm.md"):
    print(f"正在加载标准答案: {expert_file}")
    try:
        with open(expert_file, 'r', encoding='utf-8') as f:
            expert_lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("❌ 找不到标准答案文件，请检查路径。")
        return

    report = ["# 自动分词实验 精华 Bad Case 错题本 (绝对物理坐标对齐版)\n"]
    report.append("（说明：已彻底解决错位问题，提取出的必定是纯粹的算法切分歧义）\n\n")

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️ 找不到文件 {test_file}，已跳过。")
            continue

        model_name = test_file.replace('result_', '').replace('.txt', '')
        print(f"正在使用物理坐标系扫描 {model_name} ...")
        report.append(f"## 模型：{model_name}\n")

        with open(test_file, 'r', encoding='utf-8') as f:
            test_lines = [line.strip() for line in f.readlines()]

        diff_count = 0
        for i in range(min(len(expert_lines), len(test_lines))):
            e_line = expert_lines[i]
            t_line = test_lines[i]

            # 获取物理坐标和边界
            e_intervals, e_bound, e_len = get_intervals_and_boundaries(e_line)
            t_intervals, t_bound, t_len = get_intervals_and_boundaries(t_line)

            # 底层长度不一致说明原文本有差异，直接跳过避免报错
            if e_len != t_len:
                continue

            # 找出那些 专家切了但机器没切，或者 机器切了但专家没切 的坐标
            diff_bounds = sorted(list(e_bound.symmetric_difference(t_bound)))
            if not diff_bounds:
                continue

            # 聚类：把挨得近的错误（相差不到15个字）合并成一个案发现场，避免一句话被反复提取
            clusters = []
            curr_cluster = [diff_bounds[0]]
            for b in diff_bounds[1:]:
                if b - curr_cluster[-1] <= 15: 
                    curr_cluster.append(b)
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [b]
            clusters.append(curr_cluster)

            # 针对每个错误簇进行提取
            for cluster in clusters:
                # 在第一个错误前退 10个字，最后一个错误后进 10个字
                c_start = max(0, min(cluster) - 10)
                c_end = min(e_len, max(cluster) + 10)

                e_snippet = reconstruct_snippet(e_intervals, c_start, c_end)
                t_snippet = reconstruct_snippet(t_intervals, c_start, c_end)

                # 最后再做一道保险：如果拼接出来一模一样，就不输出
                if e_snippet != t_snippet:
                    report.append(f"**案例 {diff_count + 1} (来源行号: {i+1})**")
                    report.append(f"- **标准专家**: `...{e_snippet}...`")
                    report.append(f"- **{model_name}输出**: `...{t_snippet}...`\n")
                    diff_count += 1

        if diff_count == 0:
            report.append("✅ 与标准答案核心切分完全一致，无 Bad Case。\n")
        report.append("---\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"\n🎉 绝对对齐版提取完成！错题本已生成：{os.path.abspath(output_file)}")

if __name__ == '__main__':
    EXPERT_FILE = 'expert_standard_answer.txt' 
    
    TEST_FILES = [
        "result_FMM.txt",
        "result_BMM.txt",
        "result_BiMM.txt",
        "result_MinSeg.txt",
        "result_HMM.txt",
        "result_Ngram.txt",
        "result_CRF.txt"
        "result_CRF_Pro.txt"
    ]
    
    extract_absolute_bad_cases(EXPERT_FILE, TEST_FILES)