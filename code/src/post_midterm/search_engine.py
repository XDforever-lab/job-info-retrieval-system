"""
检索引擎 —— To Do List 第九阶段 #190 ~ #208

对应理论课: 第2章 向量空间模型 + 布尔逻辑检索 + 第3章 全文检索 + 第6章 检索加权

检索流程:
  用户输入 "Python 开发"
    → parse_query: Jieba分词 + 去停用词 → ["Python", "开发"]
    → query_to_bool: 倒排索引布尔检索 → 候选文档集合
    → query_to_vector: 通过 vocab_map 构建查询向量
    → compute_cosine: 查询向量 × TF-IDF矩阵 → 余弦相似度得分
    → apply_title_boost: 标题命中 ×2.0
    → apply_company_bonus: 企业名匹配 +0.1
    → apply_boolean_filter: 行业/城市/学历/经验/薪资 过滤
    → sort + paginate → 返回结果

用法:
    from src.post_midterm.search_engine import SearchEngine
    engine = SearchEngine()
    results = engine.search("Python 开发", filters={}, sort_by="relevance")

测试:
    python src/post_midterm/search_engine.py
"""

import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, load_npz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (CLEANED_CSV, SEGMENTED_CSV, STOPWORDS_FILE, MODEL_DIR,
                    SEARCH_PAGE_SIZE, TITLE_WEIGHT, CONTENT_WEIGHT, COMPANY_BONUS)

# ── 文件路径 ──────────────────────────────────────
INDEX_FILE = os.path.join(MODEL_DIR, 'inverted_index.pkl')
TFIDF_MATRIX_FILE = os.path.join(MODEL_DIR, 'tfidf_matrix.npz')
VOCAB_MAP_FILE = os.path.join(MODEL_DIR, 'vocab_map.json')
SIMILAR_JOBS_FILE = os.path.join(MODEL_DIR, '..', 'output', '16_similar', 'similar_jobs.json')
ALL_KEYWORDS_FILE = os.path.join(MODEL_DIR, '..', 'output', '10_keywords', 'all_keywords.json')

# 解决 similar_jobs 和 all_keywords 的路径问题
SIMILAR_JOBS_FILE = os.path.normpath(os.path.join(MODEL_DIR, '..', 'output', '16_similar', 'similar_jobs.json'))
ALL_KEYWORDS_FILE = os.path.normpath(os.path.join(MODEL_DIR, '..', 'output', '10_keywords', 'all_keywords.json'))


class SearchEngine:
    """检索引擎 —— 整合倒排索引 + VSM + 布尔过滤"""

    def __init__(self):
        self.ready = False

    def load(self):
        """加载所有依赖数据到内存"""
        import pandas as pd
        import jieba

        t0 = time.perf_counter()

        # 1. 主数据 (DataFrame, 用于获取职位详情 + 布尔过滤)
        print("[SearchEngine] 加载主数据...")
        self.df = pd.read_csv(CLEANED_CSV, encoding='utf-8-sig')
        self.N = len(self.df)

        # 2. 倒排索引 (用于布尔检索快速定位候选文档)
        print("[SearchEngine] 加载倒排索引...")
        with open(INDEX_FILE, 'rb') as f:
            self.lexicon = pickle.load(f)

        # 3. TF-IDF 矩阵 + 词汇表 (用于 VSM 余弦相似度计算)
        print("[SearchEngine] 加载 TF-IDF 矩阵...")
        self.tfidf_matrix = load_npz(TFIDF_MATRIX_FILE)
        with open(VOCAB_MAP_FILE, 'r', encoding='utf-8') as f:
            vm = json.load(f)
        self.feature_names = vm['feature_names']      # 词列表, 索引=矩阵列号
        self.term_to_col = {term: i for i, term in enumerate(self.feature_names)}  # 词→列号映射
        self.idf = vm['df']                            # 词→DF

        # 4. 停用词
        print("[SearchEngine] 加载停用词...")
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            self.stopwords = set(line.strip() for line in f if line.strip())

        # 5. 关键词库 (每条职位的 Top-15 关键词)
        if os.path.exists(ALL_KEYWORDS_FILE):
            with open(ALL_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
                self.all_keywords = json.load(f)
        else:
            self.all_keywords = {}

        # 6. 相似推荐数据
        if os.path.exists(SIMILAR_JOBS_FILE):
            with open(SIMILAR_JOBS_FILE, 'r', encoding='utf-8') as f:
                self.similar_jobs = json.load(f)
        else:
            self.similar_jobs = {}

        # 7. 构建标题词索引 (用于标题加权)
        print("[SearchEngine] 构建标题词索引...")
        self.title_index = defaultdict(set)   # {词 → {doc_id, ...}}
        desc_col = list(self.df.columns)[-2] if '职位描述_分词' in str(self.df.columns) else None
        title_col_orig = '招聘岗位'
        for idx, row in self.df.iterrows():
            title = str(row.get(title_col_orig, ''))
            # Jieba 分词标题
            words = set(jieba.cut(title))
            for w in words:
                w = w.strip()
                if len(w) >= 2 and w not in self.stopwords:
                    self.title_index[w].add(idx)

        elapsed = time.perf_counter() - t0
        self.ready = True
        print(f"[SearchEngine] 加载完成, 耗时 {elapsed:.1f}s, {self.N} 条文档就绪")
        return self

    # ══════════════════════════════════════════════════════
    # 9.1 查询处理 (#190-#193)
    # ══════════════════════════════════════════════════════

    def parse_query(self, query_text):
        """#190: 用户输入 → Jieba 分词 → 去停用词 → 查询词列表"""
        import jieba
        if not query_text or not query_text.strip():
            return []
        words = jieba.cut(query_text.strip())
        return [w.strip() for w in words
                if w.strip() and len(w.strip()) >= 2 and w.strip() not in self.stopwords]

    def query_to_bool(self, query_terms):
        """#191: 倒排索引布尔检索 → 返回包含所有查询词的候选文档集合"""
        if not query_terms:
            return set()

        first = query_terms[0]
        if first not in self.lexicon:
            return set()

        doc_sets = [set(doc_id for doc_id, _, _ in self.lexicon[first]['postings'])]

        for term in query_terms[1:]:
            if term not in self.lexicon:
                return set()
            doc_sets.append(set(doc_id for doc_id, _, _ in self.lexicon[term]['postings']))

        return set.intersection(*doc_sets)

    def query_to_vector(self, query_terms):
        """#192: 查询词 → TF-IDF 稀疏查询向量"""
        vec = np.zeros(len(self.feature_names))
        for term in query_terms:
            if term in self.term_to_col:
                col = self.term_to_col[term]
                # 查询词的 TF-IDF: (1+log(1)) × log(N/df) = 1 × idf
                df = self.idf.get(term, 1)
                vec[col] = math.log(self.N / df) if df > 0 else 0

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return csr_matrix(vec)

    def compute_cosine(self, query_vec):
        """#193: 查询向量 × TF-IDF 矩阵 → 余弦相似度得分数组"""
        # tfidf_matrix 已 L2 归一化, query_vec 已归一化
        # 点积 = 余弦相似度
        scores = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()
        return scores

    # ══════════════════════════════════════════════════════
    # 9.2 多字段联合检索 (#194-#197)
    # ══════════════════════════════════════════════════════

    def apply_title_boost(self, scores, query_terms):
        """#194: 标题字段 2.0 倍权重加成"""
        boosted = scores.copy()
        for term in query_terms:
            if term in self.title_index:
                for doc_id in self.title_index[term]:
                    boosted[doc_id] *= TITLE_WEIGHT  # 默认 2.0
        return boosted

    def apply_company_bonus(self, scores, query_text):
        """#196: 企业名称匹配加分"""
        if not query_text:
            return scores
        boosted = scores.copy()
        query_lower = query_text.strip().lower()
        name_col = '企业名称'
        for idx, row in self.df.iterrows():
            company = str(row.get(name_col, '')).lower()
            if query_lower in company or company in query_lower:
                boosted[idx] += COMPANY_BONUS  # 默认 0.1
        return boosted

    def compute_final_score(self, query_vec, query_terms, query_text):
        """#197: 综合得分 = 余弦相似度 + 标题加权 + 企业名加分"""
        scores = self.compute_cosine(query_vec)
        scores = self.apply_title_boost(scores, query_terms)
        scores = self.apply_company_bonus(scores, query_text)
        return scores

    # ══════════════════════════════════════════════════════
    # 9.3 布尔过滤 (#198-#199)
    # ══════════════════════════════════════════════════════

    def apply_boolean_filter(self, doc_indices, filters):
        """
        #198-#199: 结构化字段布尔过滤

        filters 格式:
          {
            'industry': '互联网/软硬件技术',    # 精确匹配 (行业字段包含即通过)
            'city': '上海',
            'education': '本科',
            'experience': '2年及以上',
            'salary_min': 10000,
            'salary_max': 50000,
            'date_from': '2026-01-01',
            'date_to': '2026-06-01',
          }

        Returns: 过滤后的 doc_id 列表
        """
        if not filters:
            return list(doc_indices)

        result = []
        for doc_id in doc_indices:
            row = self.df.iloc[doc_id]
            if self._doc_passes_filters(row, filters):
                result.append(doc_id)
        return result

    def _doc_passes_filters(self, row, filters):
        """检查单条文档是否通过所有筛选条件"""
        # 行业 (模糊匹配, 因为 9 大类名可能部分匹配原 81 类)
        if 'industry' in filters and filters['industry']:
            industry = str(row.get('上市公司行业', ''))
            if filters['industry'] not in industry:
                return False

        # 城市
        if 'city' in filters and filters['city']:
            city = str(row.get('工作城市', ''))
            if filters['city'] != city:
                return False

        # 学历
        if 'education' in filters and filters['education']:
            edu = str(row.get('学历要求', ''))
            if filters['education'] not in edu:
                return False

        # 经验
        if 'experience' in filters and filters['experience']:
            exp = str(row.get('要求经验', ''))
            if filters['experience'] not in exp:
                return False

        # 最低月薪
        if 'salary_min' in filters and filters['salary_min']:
            try:
                min_salary = float(row.get('最低月薪', 0))
                if min_salary < filters['salary_min']:
                    return False
            except (ValueError, TypeError):
                pass

        # 最高月薪
        if 'salary_max' in filters and filters['salary_max']:
            try:
                max_salary = float(row.get('最高月薪', 0))
                if max_salary > filters['salary_max']:
                    return False
            except (ValueError, TypeError):
                pass

        # 发布日期范围
        if 'date_from' in filters and filters['date_from']:
            pub_date = str(row.get('招聘发布日期', ''))
            if pub_date < filters['date_from']:
                return False

        if 'date_to' in filters and filters['date_to']:
            pub_date = str(row.get('招聘发布日期', ''))
            if pub_date > filters['date_to']:
                return False

        return True

    # ══════════════════════════════════════════════════════
    # 9.4 排序与分页 (#200-#202)
    # ══════════════════════════════════════════════════════

    def sort_results(self, doc_scores, sort_by='relevance'):
        """#200: 四种排序方式"""
        if sort_by == 'relevance':
            return sorted(doc_scores, key=lambda x: -x[1])  # 得分降序

        doc_ids = [d for d, _ in doc_scores]

        if sort_by == 'date':
            # 按发布日期降序
            def get_date(doc_id):
                d = str(self.df.iloc[doc_id].get('招聘发布日期', ''))
                return d
            return sorted(doc_scores, key=lambda x: get_date(x[0]), reverse=True)

        elif sort_by == 'salary_min':
            def get_min_salary(doc_id):
                try:
                    return float(self.df.iloc[doc_id].get('最低月薪', 0))
                except:
                    return 0
            return sorted(doc_scores, key=lambda x: get_min_salary(x[0]), reverse=True)

        elif sort_by == 'salary_max':
            def get_max_salary(doc_id):
                try:
                    return float(self.df.iloc[doc_id].get('最高月薪', 0))
                except:
                    return 0
            return sorted(doc_scores, key=lambda x: get_max_salary(x[0]), reverse=True)

        return doc_scores

    def paginate(self, doc_scores, page=1, page_size=None):
        """#201: 分页"""
        if page_size is None:
            page_size = SEARCH_PAGE_SIZE  # 20
        start = (page - 1) * page_size
        end = start + page_size
        total = len(doc_scores)
        total_pages = max(1, math.ceil(total / page_size))
        return {
            'results': doc_scores[start:end],
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': total_pages,
        }

    # ══════════════════════════════════════════════════════
    # 主检索函数 (#192 整合)
    # ══════════════════════════════════════════════════════

    def search(self, query_text, filters=None, sort_by='relevance',
               page=1, page_size=None):
        """
        主检索函数

        Args:
            query_text: 用户输入的关键词, 如 "Python 开发"
            filters:    结构化筛选条件 dict
            sort_by:    'relevance' | 'date' | 'salary_min' | 'salary_max'
            page:       页码 (从1开始)
            page_size:  每页条数 (默认20)

        Returns:
            {
                'results': [(doc_id, score), ...],
                'total': int,       # 总命中数
                'page': int,
                'page_size': int,
                'total_pages': int,
                'time': float,      # 检索耗时(秒)
                'query_terms': list # 分词结果
            }
        """
        t0 = time.perf_counter()

        # 1. 解析查询
        query_terms = self.parse_query(query_text)

        # 2. 布尔检索 → 候选文档集
        if query_terms:
            candidate_ids = self.query_to_bool(query_terms)
        else:
            # 无查询词 → 返回全部文档 (仅筛选)
            candidate_ids = set(range(self.N))

        # 3. 布尔过滤 (结构化字段)
        if filters:
            candidate_ids = set(self.apply_boolean_filter(candidate_ids, filters))

        if not candidate_ids:
            elapsed = time.perf_counter() - t0
            return {
                'results': [], 'total': 0, 'page': page,
                'page_size': page_size or SEARCH_PAGE_SIZE,
                'total_pages': 0, 'time': round(elapsed, 4),
                'query_terms': query_terms
            }

        # 4. VSM 余弦相似度计算
        if query_terms:
            query_vec = self.query_to_vector(query_terms)
            scores = self.compute_cosine(query_vec)
            # 标题加权 + 企业名加分
            scores = self.apply_title_boost(scores, query_terms)
            scores = self.apply_company_bonus(scores, query_text)
        else:
            # 无查询词 → 给所有候选文档相同得分
            scores = np.ones(self.N)

        # 5. 提取候选文档得分
        doc_scores = [(doc_id, float(scores[doc_id])) for doc_id in candidate_ids]

        # 6. 排序
        doc_scores = self.sort_results(doc_scores, sort_by)

        # 7. 分页
        paginated = self.paginate(doc_scores, page, page_size)

        elapsed = time.perf_counter() - t0
        paginated['time'] = round(elapsed, 4)
        paginated['query_terms'] = query_terms

        return paginated

    # ══════════════════════════════════════════════════════
    # 辅助查询接口
    # ══════════════════════════════════════════════════════

    def _safe_str(self, val):
        """安全转字符串, 处理 NaN"""
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return ''
            return str(val)
        except (ValueError, TypeError):
            return ''

    def _safe_int(self, val, default=0):
        """安全转整数, 处理 NaN"""
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def get_job_by_id(self, doc_id):
        """按内部 ID 获取单条职位完整信息"""
        if doc_id < 0 or doc_id >= self.N:
            return None
        row = self.df.iloc[doc_id]
        return {
            'id': int(doc_id),
            'company': self._safe_str(row.get('企业名称')),
            'stock_name': self._safe_str(row.get('股票简称')),
            'industry': self._safe_str(row.get('上市公司行业')),
            'title': self._safe_str(row.get('招聘岗位')),
            'city': self._safe_str(row.get('工作城市')),
            'salary_min': self._safe_int(row.get('最低月薪')),
            'salary_max': self._safe_int(row.get('最高月薪')),
            'education': self._safe_str(row.get('学历要求')),
            'experience': self._safe_str(row.get('要求经验')),
            'category': self._safe_str(row.get('招聘类别')),
            'publish_date': self._safe_str(row.get('招聘发布日期')),
            'end_date': self._safe_str(row.get('招聘结束日期')),
            'description': self._safe_str(row.get('职位描述')),
            'keywords': self.all_keywords.get(str(doc_id), []) if self.all_keywords else [],
        }

    def get_jobs_by_ids(self, doc_ids):
        """批量获取职位信息"""
        return [self.get_job_by_id(did) for did in doc_ids]

    def get_filter_options(self):
        """返回前端筛选组件所需的所有选项"""
        industries = self.df['上市公司行业'].value_counts().head(30).to_dict()
        cities = self.df['工作城市'].value_counts().head(30).to_dict()
        educations = self.df['学历要求'].value_counts().to_dict()
        experiences = self.df['要求经验'].value_counts().to_dict()

        return {
            'industries': [{'name': k, 'count': int(v)} for k, v in industries.items()],
            'cities': [{'name': k, 'count': int(v)} for k, v in cities.items()],
            'educations': list(educations.keys()),
            'experiences': list(experiences.keys()),
        }

    def get_similar_jobs(self, doc_id, top_k=5):
        """获取相似职位推荐"""
        if not self.similar_jobs:
            return []
        key = str(doc_id)
        similar = self.similar_jobs.get(key, [])[:top_k]
        result = []
        for sim_id, sim_score in similar:
            job = self.get_job_by_id(int(sim_id))
            if job:
                job['similarity'] = round(sim_score, 4)
                result.append(job)
        return result


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("检索引擎测试 —— 第九阶段")
    print("=" * 60)

    engine = SearchEngine()
    engine.load()

    test_queries = [
        ("Python 开发", {}),
        ("财务 审计", {'city': '上海'}),
        ("数据分析", {}),
        ("宁德时代", {}),
        ("电气 工程", {'education': '本科'}),
    ]

    for query_text, filters in test_queries:
        result = engine.search(query_text, filters=filters, sort_by='relevance')
        print(f"\n查询: '{query_text}' | 筛选: {filters}")
        print(f"  分词: {result['query_terms']}")
        print(f"  命中: {result['total']} 条, 耗时 {result['time']*1000:.1f}ms")
        for doc_id, score in result['results'][:3]:
            job = engine.get_job_by_id(doc_id)
            salary = f"{job['salary_min']/1000:.0f}k-{job['salary_max']/1000:.0f}k" if job['salary_min'] else 'mianyi'
            print(f"  [{score:.4f}] {job['title']} | {job['company']} | {job['city']} | {salary}")

    # 排序测试
    print("\n--- 排序测试 ---")
    result = engine.search("Python", sort_by='salary_min')
    print(f"按最低月薪降序 (前5):")
    for doc_id, score in result['results'][:5]:
        job = engine.get_job_by_id(doc_id)
        print(f"  {job['salary_min']} | {job['title']} | {job['company']}")

    # 布尔过滤测试
    print("\n--- 布尔过滤测试 ---")
    result = engine.search("", filters={'city': '上海', 'education': '硕士'})
    print(f"上海 + 硕士: {result['total']} 条")
    result = engine.search("", filters={'salary_min': 30000})
    print(f"月薪>=30k: {result['total']} 条")

    # 分页测试
    print("\n--- 分页测试 ---")
    result = engine.search("开发", page=1, page_size=5)
    print(f"关键词'开发': 共{result['total']}条, {result['total_pages']}页, 第1页{len(result['results'])}条")

    # 相似推荐测试
    print("\n--- 相似推荐测试 ---")
    sim = engine.get_similar_jobs(0, top_k=3)
    if sim:
        print(f"文档#0 的相似职位:")
        for s in sim:
            print(f"  [{s['similarity']:.4f}] {s['title']} | {s['company']}")

    print("\n第九阶段测试完成")


if __name__ == '__main__':
    main()
