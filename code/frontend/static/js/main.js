/** 上市公司招聘信息检索系统 —— 前端脚本 */

document.addEventListener('DOMContentLoaded', function() {
  // 首页：如果有动态加载需求，使用 API
  // 目前首页使用服务端渲染，无需额外 JS
});

/** 加载热门行业 (AJAX) */
async function loadHotIndustries() {
  try {
    var resp = await fetch('/api/industries?top=8');
    var data = await resp.json();
    var container = document.getElementById('hot-industries');
    if (!container) return;
    if (!data.length) {
      container.innerHTML = '<div class="loading">暂无数据</div>';
      return;
    }
    container.innerHTML = data.map(function(item) {
      return '<a href="/search?industry=' + encodeURIComponent(item.name) + '" class="industry-card">' +
        '<span class="ind-count">' + item.count + '</span>' +
        '<span class="ind-name">' + item.name + '</span></a>';
    }).join('');
  } catch (e) { console.error('加载行业失败:', e); }
}

/** 加载最新招聘 (AJAX) */
async function loadLatestJobs() {
  try {
    var resp = await fetch('/api/search?sort_by=date&page_size=12');
    var data = await resp.json();
    var container = document.getElementById('latest-jobs');
    if (!container) return;
    if (!data.results || !data.results.length) {
      container.innerHTML = '<div class="loading">暂无最新招聘</div>';
      return;
    }
    container.innerHTML = data.results.map(function(j) {
      return renderJobCard(j);
    }).join('');
  } catch (e) { console.error('加载最新招聘失败:', e); }
}

/** 渲染单张职位卡片 */
function renderJobCard(job) {
  var salary = '面议';
  if (job.salary_min && job.salary_max) {
    salary = (job.salary_min/1000).toFixed(0) + 'k-' + (job.salary_max/1000).toFixed(0) + 'k';
  }
  var snippet = (job.snippet || '').substring(0, 150);

  return '<a href="/job/' + job.id + '" class="job-card">' +
    '<div class="job-card-header">' +
      '<span class="job-title">' + escapeHTML(job.title || '') + '</span>' +
      '<span class="job-card-salary">' + salary + '</span>' +
    '</div>' +
    '<div class="job-card-company">' + escapeHTML(job.company || '') + '</div>' +
    '<div class="job-card-meta">' +
      '<span>' + escapeHTML(job.city || '') + '</span>' +
      '<span>' + escapeHTML(job.education || '') + '</span>' +
      '<span>' + (job.date || '') + '</span>' +
    '</div>' +
    '<div class="job-card-desc">' + snippet + '</div>' +
  '</a>';
}

function escapeHTML(str) {
  var div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
