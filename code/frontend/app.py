"""上市公司招聘信息检索系统 —— Flask 主应用"""

from flask import Flask, render_template, request, jsonify
import sys
sys.path.insert(0, '..')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'recruit-search-system-2026'


@app.route('/')
def index():
    """首页 —— 搜索门户"""
    return render_template('index.html')


@app.route('/search')
def search():
    """检索结果页"""
    return render_template('search.html')


@app.route('/job/<int:job_id>')
def job_detail(job_id):
    """职位详情页"""
    return render_template('detail.html')


@app.route('/industry')
def industry_browse():
    """行业浏览页"""
    return render_template('industry.html')


@app.route('/stats')
def statistics():
    """数据统计页"""
    return render_template('stats.html')


@app.route('/cluster')
def cluster_view():
    """聚类视图页"""
    return render_template('cluster.html')


@app.route('/about')
def about():
    """关于页"""
    return render_template('about.html')


# ── API 接口 ──────────────────────────────

@app.route('/api/search')
def api_search():
    """检索 API"""
    query = request.args.get('q', '')
    industry = request.args.get('industry', '')
    city = request.args.get('city', '')
    education = request.args.get('education', '')
    experience = request.args.get('experience', '')
    salary_min = request.args.get('salary_min', type=int)
    salary_max = request.args.get('salary_max', type=int)
    sort_by = request.args.get('sort_by', 'relevance')
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 20, type=int)

    # TODO: 接入检索引擎
    return jsonify({
        'query': query,
        'results': [],
        'total': 0,
        'page': page,
        'page_size': page_size,
        'time': 0
    })


@app.route('/api/job/<int:job_id>')
def api_job_detail(job_id):
    """职位详情 API"""
    # TODO: 接入数据查询
    return jsonify({})


@app.route('/api/industries')
def api_industries():
    """行业列表 API"""
    # TODO: 返回所有行业分类及计数
    return jsonify([])


@app.route('/api/cities')
def api_cities():
    """城市列表 API"""
    return jsonify([])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
