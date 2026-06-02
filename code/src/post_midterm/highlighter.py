"""
关键词高亮与上下文片段 —— 第九阶段 #203-#204

功能:
  highlight_keywords(text, query_terms)  → HTML文本, 关键词用<mark>包裹
  generate_snippet(text, query_terms)     → 提取包含关键词的上下文片段

用法:
    from src.post_midterm.highlighter import highlight_keywords, generate_snippet
"""

import re


def highlight_keywords(text, query_terms, tag='mark', css_class='highlight'):
    """
    #203: 对原文中匹配的查询词包裹 <mark> 标签

    处理逻辑:
      1. 对输入的每个查询词，在文本中找到所有出现位置
      2. 用 <mark class="highlight">关键词</mark> 替换
      3. 先处理长词再处理短词，避免短词在长词内部重复匹配
      4. 结果做 HTML 转义（除 mark 标签外）

    Args:
        text: 原始文本
        query_terms: 查询词列表
        tag: HTML 标签名 (默认 mark)
        css_class: CSS 类名

    Returns:
        HTML 安全字符串, 关键词被标签包裹
    """
    if not text or not query_terms:
        return escape_html(text) if text else ''

    # 按词长度从长到短排序，避免短词先匹配破坏长词
    terms = sorted(set(query_terms), key=len, reverse=True)

    # 先转义 HTML
    result = escape_html(text)

    # 逐个替换
    for term in terms:
        if not term:
            continue
        escaped_term = escape_html(term)
        # 使用正则替换，大小写不敏感（英文关键词）
        pattern = re.compile(re.escape(escaped_term), re.IGNORECASE)
        replacement = f'<{tag} class="{css_class}">{escaped_term}</{tag}>'
        result = pattern.sub(replacement, result)

    return result


def generate_snippet(text, query_terms, max_len=150):
    """
    #204: 提取包含第一个关键词的上下文片段

    策略: 找到第一个查询词在文本中的位置，向前后各扩展约 max_len/2 字符

    Args:
        text: 原始文本
        query_terms: 查询词列表
        max_len: 片段最大长度

    Returns:
        HTML 安全的上下文片段, 关键词被高亮
    """
    if not text or not query_terms:
        return escape_html(text[:max_len]) if text else ''

    # 找第一个能匹配到的查询词位置
    best_pos = -1
    best_term = None
    for term in query_terms:
        pos = text.lower().find(term.lower())
        if pos != -1:
            best_pos = pos
            best_term = term
            break

    if best_pos == -1:
        # 没有匹配 → 返回文本开头
        snippet = text[:max_len]
        return highlight_keywords(snippet, query_terms)

    # 计算截取范围
    half = max_len // 2
    start = max(0, best_pos - half)
    end = min(len(text), best_pos + half)

    # 调整到完整中文字符边界
    snippet = text[start:end]

    if start > 0:
        snippet = '...' + snippet
    if end < len(text):
        snippet = snippet + '...'

    return highlight_keywords(snippet, query_terms)


def escape_html(text):
    """HTML 特殊字符转义"""
    if not text:
        return ''
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


# ── 测试 ──────────────────────────────────────────

def main():
    text = ("岗位职责： 1. 参与高比能材料与电芯新技术开发，"
            "支持团队落地项目内研发新技术支持推动所属产品高效和精准研发。"
            " 2. 熟悉电芯领域各项新技术创新和应用，"
            "熟练Python进行数据分析与机器学习模型搭建。"
            " 3. 掌握Python开发常用框架如Django/Flask。")

    query = ['Python', '开发']

    print("Original:")
    print(text)
    print()
    print("Highlighted:")
    print(highlight_keywords(text, query))
    print()
    print("Snippet (max_len=80):")
    print(generate_snippet(text, query, max_len=80))


if __name__ == '__main__':
    main()
