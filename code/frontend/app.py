"""上市公司招聘信息检索系统 —— Flask 主应用"""

import json
import os
import sys
import time

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.post_midterm.search_engine import SearchEngine
from src.post_midterm.highlighter import highlight_keywords, generate_snippet

app = Flask(__name__)
app.config['SECRET_KEY'] = 'recruit-search-system-2026'

# ── 启动时加载检索引擎 ─────────────────────────────
print("=" * 50)
print("系统启动中...")
_engine = SearchEngine()
_engine.load()
print(f"系统就绪, {_engine.N} 条文档可用")
print("=" * 50)


# ══════════════════════════════════════════════════
# 页面路由
# ══════════════════════════════════════════════════

@app.route('/')
def index():
    """首页 —— 搜索门户"""
    # 热门行业 Top-8
    industries = _engine.df['上市公司行业'].value_counts().head(8)
    hot_industries = [{'name': k, 'count': int(v)} for k, v in industries.items()]

    # 最新 8 条招聘
    latest = _engine.df.sort_values('招聘发布日期', ascending=False).head(8)
    latest_jobs = []
    for i in range(len(latest)):
        row = latest.iloc[i]
        latest_jobs.append({
            'doc_id': int(latest.index[i]),
            'title': str(row.get('招聘岗位', '')),
            'company': str(row.get('企业名称', '')),
            'city': str(row.get('工作城市', '')),
            'salary_min': int(row.get('最低月薪', 0)) if not _is_nan(row.get('最低月薪')) else 0,
            'salary_max': int(row.get('最高月薪', 0)) if not _is_nan(row.get('最高月薪')) else 0,
            'education': str(row.get('学历要求', '')),
            'date': str(row.get('招聘发布日期', '')),
        })

    # 城市选项
    cities = _engine.df['工作城市'].value_counts().head(20).index.tolist()

    return render_template('index.html',
                         industries=hot_industries,
                         latest_jobs=latest_jobs,
                         cities=cities)


@app.route('/search')
def search():
    """检索结果页"""
    query = request.args.get('q', '')
    industry = request.args.get('industry', '')
    city = request.args.get('city', '')
    education = request.args.get('education', '')
    experience = request.args.get('experience', '')
    salary_min = request.args.get('salary_min', '')
    salary_max = request.args.get('salary_max', '')
    sort_by = request.args.get('sort_by', 'relevance')
    page = request.args.get('page', 1, type=int)

    # 构建筛选条件
    filters = {}
    if industry:
        filters['industry'] = industry
    if city:
        filters['city'] = city
    if education:
        filters['education'] = education
    if experience:
        filters['experience'] = experience
    if salary_min:
        filters['salary_min'] = int(salary_min)
    if salary_max:
        filters['salary_max'] = int(salary_max)

    # 执行检索
    result = _engine.search(query, filters=filters, sort_by=sort_by, page=page)

    # 组装结果
    items = []
    for doc_id, score in result['results']:
        job = _engine.get_job_by_id(doc_id)
        if not job:
            continue
        snippet = generate_snippet(job['description'], result['query_terms'], max_len=150)
        job['doc_id'] = doc_id
        job['score'] = round(score, 4)
        job['snippet'] = snippet
        items.append(job)

    # 筛选选项
    filter_opts = _engine.get_filter_options()

    return render_template('search.html',
                         query=query,
                         results=items,
                         total=result['total'],
                         total_pages=result['total_pages'],
                         page=page,
                         elapsed=round(result['time'] * 1000, 1),
                         query_terms=result['query_terms'],
                         sort_by=sort_by,
                         current_filters={'industry': industry, 'city': city,
                                         'education': education, 'experience': experience,
                                         'salary_min': salary_min, 'salary_max': salary_max},
                         filter_opts=filter_opts)


@app.route('/job/<int:doc_id>')
def job_detail(doc_id):
    """职位详情页"""
    job = _engine.get_job_by_id(doc_id)
    if not job:
        return render_template('detail.html', error='职位不存在'), 404

    # 相似职位
    similar = _engine.get_similar_jobs(doc_id, top_k=5)

    # 关键词
    keywords = job.get('keywords', [])[:10]

    return render_template('detail.html', job=job, doc_id=doc_id,
                         similar=similar, keywords=keywords)


@app.route('/industry')
def industry_browse():
    """行业浏览页 — 9 大行业"""
    industries = _engine.df['上市公司行业'].value_counts().to_dict()
    # 按数量排序取前 9 个行业
    top_industries = sorted(industries.items(), key=lambda x: -x[1])[:9]

    # 为每个行业取代表性关键词 (从聚合数据中取)
    agg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'output', '10_keywords', 'agg_industry.json')
    agg_keywords = {}
    if os.path.exists(agg_file):
        import json
        with open(agg_file, 'r', encoding='utf-8') as f:
            agg_keywords = json.load(f)

    industry_list = []
    for name, count in top_industries:
        kw_list = agg_keywords.get(name, [])[:5]
        industry_list.append({
            'name': name,
            'count': count,
            'keywords': kw_list,
        })

    return render_template('industry.html', industries=industry_list)


@app.route('/stats')
def statistics():
    """数据统计页"""
    # 行业分布
    ind_counts = _engine.df['上市公司行业'].value_counts().head(9)
    industry_data = [{'name': k, 'count': int(v)} for k, v in ind_counts.items()]

    # 城市分布
    city_counts = _engine.df['工作城市'].value_counts().head(10)
    city_data = [{'name': k, 'count': int(v)} for k, v in city_counts.items()]

    # 薪资分布
    salaries = []
    for _, row in _engine.df.iterrows():
        try:
            lo = float(row['最低月薪']) if not _is_nan(row['最低月薪']) else 0
            hi = float(row['最高月薪']) if not _is_nan(row['最高月薪']) else 0
            if lo > 0 and hi > 0:
                avg = (lo + hi) / 2
                salaries.append(avg)
        except (ValueError, TypeError):
            pass
    salary_bins = {'0-5k': 0, '5k-10k': 0, '10k-15k': 0, '15k-20k': 0,
                   '20k-25k': 0, '25k-30k': 0, '30k-40k': 0, '40k+': 0}
    for s in salaries:
        if s < 5000: salary_bins['0-5k'] += 1
        elif s < 10000: salary_bins['5k-10k'] += 1
        elif s < 15000: salary_bins['10k-15k'] += 1
        elif s < 20000: salary_bins['15k-20k'] += 1
        elif s < 25000: salary_bins['20k-25k'] += 1
        elif s < 30000: salary_bins['25k-30k'] += 1
        elif s < 40000: salary_bins['30k-40k'] += 1
        else: salary_bins['40k+'] += 1

    # 学历分布
    edu_counts = _engine.df['学历要求'].value_counts().to_dict()

    # 经验分布
    exp_counts = _engine.df['要求经验'].value_counts().to_dict()

    return render_template('stats.html',
                         industry_data=industry_data,
                         city_data=city_data,
                         salary_bins=salary_bins,
                         edu_data=edu_counts,
                         exp_data=exp_counts)


@app.route('/cluster')
def cluster_view():
    """聚类视图页"""
    # 读取聚类结果
    cluster_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'output', '15_cluster', 'word_cluster_info.json')
    cluster_data = []
    if os.path.exists(cluster_file):
        import json
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)

    return render_template('cluster.html', clusters=cluster_data)


@app.route('/about')
def about():
    """关于页"""
    return render_template('about.html')


# ══════════════════════════════════════════════════
# API 接口 (AJAX 调用)
# ══════════════════════════════════════════════════

@app.route('/api/search')
def api_search():
    """检索 API (JSON)"""
    query = request.args.get('q', '')
    industry = request.args.get('industry', '')
    city = request.args.get('city', '')
    education = request.args.get('education', '')
    experience = request.args.get('experience', '')
    salary_min = request.args.get('salary_min', '')
    salary_max = request.args.get('salary_max', '')
    sort_by = request.args.get('sort_by', 'relevance')
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 20, type=int)

    filters = {}
    if industry: filters['industry'] = industry
    if city: filters['city'] = city
    if education: filters['education'] = education
    if experience: filters['experience'] = experience
    if salary_min: filters['salary_min'] = int(salary_min)
    if salary_max: filters['salary_max'] = int(salary_max)

    result = _engine.search(query, filters=filters, sort_by=sort_by,
                           page=page, page_size=page_size)

    items = []
    for doc_id, score in result['results']:
        job = _engine.get_job_by_id(doc_id)
        if not job:
            continue
        items.append({
            'id': doc_id,
            'title': job['title'],
            'company': job['company'],
            'city': job['city'],
            'salary_min': job['salary_min'],
            'salary_max': job['salary_max'],
            'education': job['education'],
            'experience': job['experience'],
            'date': job['publish_date'],
            'score': round(score, 4),
            'snippet': generate_snippet(job['description'], result['query_terms'], 150),
        })

    return jsonify({
        'results': items,
        'total': result['total'],
        'page': result['page'],
        'page_size': result['page_size'],
        'total_pages': result['total_pages'],
        'time_ms': round(result['time'] * 1000, 1),
    })


@app.route('/api/job/<int:doc_id>')
def api_job_detail(doc_id):
    """职位详情 API"""
    job = _engine.get_job_by_id(doc_id)
    if not job:
        return jsonify({'error': 'not found'}), 404

    job['id'] = doc_id

    # 相似职位
    similar = _engine.get_similar_jobs(doc_id, top_k=5)

    return jsonify({
        'job': job,
        'similar': similar,
    })


@app.route('/api/industries')
def api_industries():
    """行业列表 API"""
    top_n = request.args.get('top', 9, type=int)
    counts = _engine.df['上市公司行业'].value_counts().head(top_n)
    return jsonify([{'name': k, 'count': int(v)} for k, v in counts.items()])


@app.route('/api/cities')
def api_cities():
    """城市列表 API"""
    counts = _engine.df['工作城市'].value_counts().head(30)
    return jsonify([{'name': k, 'count': int(v)} for k, v in counts.items()])


@app.route('/api/stats')
def api_stats():
    """统计数据 API"""
    return jsonify({
        'total_jobs': _engine.N,
        'total_industries': _engine.df['上市公司行业'].nunique(),
        'total_cities': _engine.df['工作城市'].nunique(),
        'total_companies': _engine.df['企业名称'].nunique(),
    })


# ══════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════

def _is_nan(val):
    """检查值是否为 NaN"""
    try:
        import math
        return isinstance(val, float) and math.isnan(val)
    except TypeError:
        return False


@app.template_filter('money')
def money_filter(val):
    """金额格式化: 15000 → '15k'"""
    try:
        v = int(val)
        if v >= 1000:
            return f'{v/1000:.0f}k'
        return str(v)
    except (ValueError, TypeError):
        return str(val)


@app.template_filter('highlight')
def highlight_filter(text, terms):
    """模板内关键词高亮"""
    if not terms:
        return text
    return highlight_keywords(text, terms)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
