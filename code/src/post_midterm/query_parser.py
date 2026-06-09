"""
智能查询解析器 v2 —— 鲁棒的自然语言检索条件提取

设计原则:
  1. 宁漏勿错: 不确定时不提取, 原样作为关键词
  2. 多策略冗余: 多种写法都能识别
  3. 非破坏性: 提取后原文不清除, 保留用于 VSM 匹配
  4. 上下文校验: 提取值必须在已知词表中
  5. 数字保护: 长数字(手机号)不误识别

用法:
    python src/post_midterm/query_parser.py  # 自测
"""

import re
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CLEANED_CSV


# ══════════════════════════════════════════════════
# 0. 全局词表
# ══════════════════════════════════════════════════

_CITY_NAMES = []
_CITY_ALIASES = {}
_EDU_MAP = {}         # {"全日制本科": "本科", ...}
_INDUSTRY_TERMS = {}   # {"互联网": "互联网和相关服务", ...}


def _init():
    global _CITY_NAMES, _CITY_ALIASES, _EDU_MAP, _INDUSTRY_TERMS
    if _CITY_NAMES:
        return
    try:
        df = pd.read_csv(CLEANED_CSV, encoding='utf-8-sig', nrows=5000)

        # 城市
        raw = df['工作城市'].dropna().unique().tolist()
        _CITY_NAMES = sorted(raw, key=len, reverse=True)
        _CITY_ALIASES = {
            'shanghai': '上海', 'beijing': '北京', 'shenzhen': '深圳',
            'guangzhou': '广州', 'hangzhou': '杭州', 'chengdu': '成都',
            'nanjing': '南京', 'wuhan': '武汉', 'suzhou': '苏州',
            'dongguan': '东莞', 'zhengzhou': '郑州', 'xian': '西安',
            'changsha': '长沙', 'qingdao': '青岛', 'tianjin': '天津',
            'chongqing': '重庆', 'xiamen': '厦门', 'hefei': '合肥',
            'fuzhou': '福州', 'jinan': '济南', 'dalian': '大连',
            'kunming': '昆明', 'shenyang': '沈阳',
            'shang hai': '上海', 'bei jing': '北京',
        }
        # 数据中的 "XX市" → "XX"
        for c in raw:
            s = c.rstrip('市')
            if s != c and len(s) >= 2:
                _CITY_ALIASES.setdefault(s, c)

        # 学历变体 (含基本形式 + 带后缀的变体, 按长度降序)
        _EDU_MAP.update({
            '博士研究生': '博士', '博士及以上': '博士', '博士': '博士',
            '硕士研究生': '硕士', '硕士及以上': '硕士', '硕士以上': '硕士',
            '研究生': '硕士', '硕士': '硕士',
            '大学本科': '本科', '全日制本科': '本科', '本科毕业': '本科',
            '本科以上': '本科', '本科及以上': '本科', '本科或以上': '本科',
            '不低于本科': '本科', '本科': '本科',
            '专科': '大专', '大专及以上': '大专', '大专以上': '大专',
            '大专': '大专',
            '高中': '高中', '中专': '中专', '中技': '中专',
            '学历不限': '不限', '不限学历': '不限', '不限': '不限',
        })

        # 行业
        for ind in df['上市公司行业'].dropna().unique():
            ind = str(ind).strip()
            for p in re.split(r'[/、·,&]', ind):
                p = p.strip()
                if 2 <= len(p) <= 6:
                    _INDUSTRY_TERMS[p] = ind
            if '互联网' in ind: _INDUSTRY_TERMS['互联网'] = ind
            if '金融' in ind and '金融' not in _INDUSTRY_TERMS:
                _INDUSTRY_TERMS['金融'] = ind
            if '制造' in ind: _INDUSTRY_TERMS['制造业'] = ind
            if '建筑' in ind: _INDUSTRY_TERMS['建筑业'] = ind

    except Exception as e:
        print(f"[Parser] init: {e}")


# ══════════════════════════════════════════════════
# 1. 中文数字转阿拉伯
# ══════════════════════════════════════════════════

_CN_NUM = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
           '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
           '十': 10, '百': 100, '千': 1000, '万': 10000}

_CN_NUM_RE = re.compile(
    r'([一二两三四五六七八九十]+)\s*万\s*([一二两三四五六七八九])?(?:\s*千\s*)?'
    r'([一二两三四五六七八九])?\s*[千百]?'
)


def _cn2int(s):
    """中文数字→int, 返回 None 表示无法解析"""
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    total = cur = 0
    for ch in s:
        if ch not in _CN_NUM:
            return None
        v = _CN_NUM[ch]
        if v >= 10:
            if cur == 0:
                cur = 1
            total += cur * v
            cur = 0
        else:
            cur = v
    total += cur
    return total if total > 0 else None


def _replace_cn_numbers(text):
    """将中文数字替换为阿拉伯数字"""
    # 关键: 先处理复合形式, 再处理简单形式, 避免提前截断

    # 1. "X万Y" → XY000  (如 "一万五"→15000, "两万三"→23000)
    text = re.sub(
        r'([一二两三四五六七八九])万([一二三四五六七八九])(?![千百十])',
        lambda m: str(_cn2int(m.group(1)) * 10000 + _cn2int(m.group(2)) * 1000),
        text)

    # 2. "X万以上/及以上/左右" → 保留后缀
    for suffix in ['以上', '及以上', '左右']:
        text = re.sub(
            r'([一二两三四五六七八九]+)万' + suffix,
            lambda m: str(_cn2int(m.group(1)) * 10000) + suffix, text)

    # 3. 孤立的 "X万" (前后无毗连数字) → 纯数字
    text = re.sub(
        r'(?<![0-9一二两三四五六七八九])([一二两三四五六七八九]+)万(?![0-9一二两三四五六七八九以上及左右])',
        lambda m: str(_cn2int(m.group(1)) * 10000), text)

    # 4. "X到Y年" → "X-Y年"
    text = re.sub(
        r'([一二两三四五六七八九]+)到([一二两三四五六七八九]+)年',
        lambda m: f'{_cn2int(m.group(1))}-{_cn2int(m.group(2))}年', text)

    # 5. "差不多/大概/约 X年"
    text = re.sub(
        r'(差不多|大概|约|大约)\s*([一二两三四五六七八九]+)年',
        lambda m: f'{m.group(1)}{_cn2int(m.group(2))}年', text)

    # 6. "X年左右"
    text = re.sub(
        r'([一二两三四五六七八九]+)年(左右)',
        lambda m: f'{_cn2int(m.group(1))}年{m.group(2)}', text)

    return text


# ══════════════════════════════════════════════════
# 2. 薪资
# ══════════════════════════════════════════════════

def _try_parse_salary(text):
    t = _replace_cn_numbers(text.lower().strip())

    def _num(s):
        s = s.strip()
        cn = _cn2int(s)
        if cn and 500 <= cn <= 500000:
            return float(cn)
        # k / K / 千
        m = re.match(r'^(\d+(?:\.\d+)?)\s*[kK千]$', s)
        if m:
            return float(m.group(1)) * 1000
        # 万 / w / W
        m = re.match(r'^(\d+(?:\.\d+)?)\s*[万wW]$', s)
        if m and 0.1 <= float(m.group(1)) <= 50:
            return float(m.group(1)) * 10000
        # 孤立的 >=2000 数字
        m = re.match(r'^(\d+)$', s)
        if m and 2000 <= int(m.group(1)) <= 200000:
            return float(m.group(1))
        return None

    # ── 范围 ──
    sep = r'\s*[-~～到至]\s*'
    rm = re.search(
        r'(?:月?薪\s*(?:不低于|不少于|大于|高于|超过)?\s*)?'
        r'(\d+(?:\.\d+)?\s*[万wWkK千]?)' + sep +
        r'(\d+(?:\.\d+)?\s*[万wWkK千]?)', t)
    if rm:
        lo, hi = _num(rm.group(1)), _num(rm.group(2))
        if lo and hi and 1000 <= lo <= 200000 and 1000 <= hi <= 200000:
            return min(lo, hi), max(lo, hi)

    # ── 下限 (X以上 | X+ | 不低于X | 不少于X) ──
    # 模式A: "不低于/不少于/大于/高于/超过 X万"  (不需要"以上"后缀)
    lm = re.search(
        r'(?:月?薪\s*)?(?:不低于|不少于|大于|高于|超过)\s*'
        r'(\d+(?:\.\d+)?\s*[万wWkK千]?)', t)
    if lm:
        lo = _num(lm.group(1))
        if lo and 1000 <= lo <= 200000:
            return lo, None

    # 模式B: "X万以上" / "X以上" / "X万+" / "Xk+"
    lm = re.search(
        r'(\d+(?:\.\d+)?\s*[万wWkK千]?)\s*(?:以[上高]|\+|以上|及以上)', t)
    if lm:
        lo = _num(lm.group(1))
        if lo and 1000 <= lo <= 200000:
            return lo, None

    # ── 精确月薪(有明确薪资前缀) ──
    em = re.search(r'(?:月薪|月工资|薪资|薪水|工资)\s*[：:为是]?\s*(\d+(?:\.\d+)?\s*[万wWkK千]?)', t)
    if em:
        lo = _num(em.group(1))
        if lo and 1000 <= lo <= 200000:
            return lo, None

    # ── 独立 k/w 数字 + 薪水语境 ──
    if re.search(r'[月薪工资]|薪资|k\b|万\s*(?:以[上高]|\+|左右)', t):
        nm = re.search(r'(?<!\d)(\d+(?:\.\d+)?)\s*[kK](?!\d)', t)
        if nm:
            lo = float(nm.group(1)) * 1000
            if 1000 <= lo <= 200000:
                return lo, None

    return None, None


# ══════════════════════════════════════════════════
# 3. 城市
# ══════════════════════════════════════════════════

def _try_parse_city(text):
    t = text.lower().strip()

    # 英文别名
    for alias, city in sorted(_CITY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in t:
            return city

    # 中文城市名 (长名优先)
    for city in _CITY_NAMES:
        if city not in text:
            continue
        idx = text.find(city)
        at_start = idx == 0
        at_end = idx + len(city) >= len(text)
        before = text[idx - 1:idx] if idx > 0 else ''
        after = (text[idx + len(city):idx + len(city) + 1]
                 if idx + len(city) < len(text) else '')

        # 起首或末尾 → 直接匹配
        if at_start or at_end:
            return city
        # 前后非汉字 → 独立词
        if (not re.match(r'[一-鿿]', before) and
                not re.match(r'[一-鿿]', after)):
            return city
        # 前缀/后缀指示词
        if before in ('在', '去', '到', '于', '来'):
            return city
        if after in ('的', '地', '了', '是', '有', '地区', '附近', '周边'):
            return city

    # "在/去/到/来 上海" 模式 (中间有空格)
    for city in _CITY_NAMES:
        if re.search(rf'(?:在|去|到|于|来)\s+{city}', text):
            return city

    return None


# ══════════════════════════════════════════════════
# 4. 学历
# ══════════════════════════════════════════════════

def _try_parse_education(text):
    for variant, norm in sorted(_EDU_MAP.items(), key=lambda x: -len(x[0])):
        if variant in text:
            return norm
    return None


# ══════════════════════════════════════════════════
# 5. 经验
# ══════════════════════════════════════════════════

def _try_parse_experience(text):
    t = _replace_cn_numbers(text.strip())

    # "应届" "校招" "刚毕业" "毕业生" "无经验" "经验不限"
    if re.search(r'应届|校招|刚毕业|毕业生|无经验|经验不限|不限经验|无需经验', t):
        return "应届生"

    # "X年以上" / "X年及以上"
    m = re.search(r'(\d+)\s*年\s*(?:以[上及]|及以上|以上)', t)
    if m and 1 <= int(m.group(1)) <= 15:
        return f"{m.group(1)}年及以上"

    # "X-Y年"
    m = re.search(r'(\d+)\s*[-~到至]\s*(\d+)\s*年', t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return f"{lo}-{hi}年" if lo > 1 else f"{hi}年及以上"

    # "差不多/大概/约 X年" "X年左右" "X年经验"
    m = re.search(r'(?:差不多|大概|约|大约)\s*(\d+)\s*年', t)
    if m:
        return f"{m.group(1)}年及以上"

    m = re.search(r'(\d+)\s*年\s*(?:左右|经验)', t)
    if m:
        return f"{m.group(1)}年及以上"

    return None


# ══════════════════════════════════════════════════
# 6. 行业
# ══════════════════════════════════════════════════

def _try_parse_industry(text):
    for term, full in sorted(_INDUSTRY_TERMS.items(), key=lambda x: -len(x[0])):
        if term not in text:
            continue
        idx = text.find(term)
        after = text[idx + len(term):idx + len(term) + 3]
        # 行业词后跟 "行业/业/类/方向/领域/相关" 才可信
        if any(s in after for s in ('行业', '业', '类', '方向', '领域', '相关')):
            return full
        # 独立词也可
        before = text[idx - 1:idx] if idx > 0 else ' '
        if not re.match(r'[一-鿿]', before):
            return full
    return None


# ══════════════════════════════════════════════════
# 7. 排除词
# ══════════════════════════════════════════════════

def _try_parse_exclude(text):
    for pat in [r'不要\s*(\S{1,10})', r'不招\s*(\S{1,10})',
                r'排除\s*(\S{1,10})',
                r'非\s*(\S{1,10})(?:岗位|岗|职位|工作)?',
                r'免\s*(\S{1,10})(?:岗|职位)?']:
        m = re.search(pat, text)
        if m:
            ex = re.sub(r'[的，,。.]', '', m.group(1))
            if len(ex) >= 2:
                return ex
    return None


# ══════════════════════════════════════════════════
# 8. 主解析
# ══════════════════════════════════════════════════

def parse_query(query_text):
    _init()
    text = query_text.strip()
    if not text or len(text) > 500:
        text = text[:500] if text else ''

    parsed = {'city': None, 'education': None, 'industry': None,
              'experience': None, 'salary_min': None, 'salary_max': None,
              'exclude': None}
    hints = []

    c = _try_parse_city(text)
    if c:
        parsed['city'] = c
        hints.append(f"城市→{c}")

    lo, hi = _try_parse_salary(text)
    if lo is not None:
        parsed['salary_min'] = int(lo)
        if hi is not None:
            parsed['salary_max'] = int(hi)
            hints.append(f"月薪→{lo/1000:.0f}k-{hi/1000:.0f}k")
        else:
            hints.append(f"月薪→≥{lo/1000:.0f}k")

    edu = _try_parse_education(text)
    if edu:
        parsed['education'] = edu
        hints.append(f"学历→{edu}")

    exp = _try_parse_experience(text)
    if exp:
        parsed['experience'] = exp
        hints.append(f"经验→{exp}")

    ind = _try_parse_industry(text)
    if ind:
        parsed['industry'] = ind
        hints.append(f"行业→{ind}")

    ex = _try_parse_exclude(text)
    if ex:
        parsed['exclude'] = ex
        hints.append(f"排除→{ex}")

    return {'keywords': text, 'parsed': parsed, 'hints': hints}


# ══════════════════════════════════════════════════
# 9. 测试
# ══════════════════════════════════════════════════

def main():
    cases = [
        "上海月薪2万以上的Python开发岗 要本科",
        "北京 数据分析 15k-30k 硕士",
        "shanghai python 20k+",
        "beijing Java 1.5w-3w",
        "深圳 两万以上",
        "杭州 一万五到两万",
        "南京 全日制本科 3年左右",
        "不低于硕士 上海",
        "北京 Java 不要测试",
        "深圳 Python 非销售岗位",
        "互联网行业 产品经理",
        "金融业 风险控制",
        "想找一个上海的python开发工作 最好月薪不低于2万",
        "给我找一个在杭州的 月薪差不多20k的 Java岗 要本科以上学历",
        "广州 数据分析 硕士 3-5年经验 15k-25k",
        "成都 应届生 Python",
        "Python", "", "   ",
    ]
    for t in cases:
        r = parse_query(t)
        hs = " | ".join(r['hints']) if r['hints'] else "—"
        print(f"[{hs}]  ← {t[:70]}")


if __name__ == '__main__':
    main()
