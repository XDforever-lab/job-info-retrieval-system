/** 上市公司招聘信息检索系统 —— 前端主脚本 */

document.addEventListener('DOMContentLoaded', () => {
  // 首页：加载热门行业和最新招聘
  if (document.getElementById('hot-industries')) {
    loadHotIndustries();
    loadLatestJobs();
  }
});

/** 加载热门行业 */
async function loadHotIndustries() {
  try {
    const resp = await fetch('/api/industries?top=8');
    const data = await resp.json();
    const container = document.getElementById('hot-industries');
    if (data.length === 0) {
      container.innerHTML = '<div class="loading">暂无行业数据</div>';
      return;
    }
    container.innerHTML = data.map(item => `
      <a href="/search?industry=${encodeURIComponent(item.name)}" class="industry-card">
        <span class="count">${item.count}</span>
        ${item.name}
      </a>
    `).join('');
  } catch (err) {
    console.error('加载行业数据失败:', err);
  }
}

/** 加载最新招聘 */
async function loadLatestJobs() {
  try {
    const resp = await fetch('/api/search?sort_by=date&page_size=12');
    const data = await resp.json();
    const container = document.getElementById('latest-jobs');
    if (!data.results || data.results.length === 0) {
      container.innerHTML = '<div class="loading">暂无最新招聘信息</div>';
      return;
    }
    container.innerHTML = data.results.map(job => renderJobCard(job)).join('');
  } catch (err) {
    console.error('加载最新招聘失败:', err);
  }
}

/** 渲染单张职位卡片 */
function renderJobCard(job, queryTerms = []) {
  const salary = job.salary_min && job.salary_max
    ? `¥${(job.salary_min / 1000).toFixed(0)}k-${(job.salary_max / 1000).toFixed(0)}k`
    : '薪资面议';

  let desc = (job.desc_snippet || '').substring(0, 150);
  queryTerms.forEach(term => {
    const re = new RegExp(`(${escapeRegExp(term)})`, 'gi');
    desc = desc.replace(re, '<span class="highlight">$1</span>');
  });

  return `
    <a href="/job/${job.id}" class="job-card">
      <div class="job-card-header">
        <span class="job-card-title">${escapeHTML(job.title || '')}</span>
        <span class="job-card-salary">${salary}</span>
      </div>
      <div class="job-card-company">
        ${escapeHTML(job.company || '')}
        ${job.stock_symbol ? ' · ' + escapeHTML(job.stock_symbol) : ''}
        ${job.stock_code ? ' (' + escapeHTML(job.stock_code) + ')' : ''}
      </div>
      <div class="job-card-meta">
        <span>📍 ${escapeHTML(job.city || '未知')}</span>
        <span>🎓 ${escapeHTML(job.education || '不限')}</span>
        <span>💼 ${escapeHTML(job.experience || '不限')}</span>
        <span>📅 ${job.publish_date || ''}</span>
      </div>
      <div class="job-card-desc">${desc}</div>
    </a>
  `;
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
