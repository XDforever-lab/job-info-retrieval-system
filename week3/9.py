import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免中文显示为方块）
# Windows 一般使用 SimHei，Mac 可以使用 Arial Unicode MS 或 Heiti TC
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
labels = ['FMM', 'BMM', 'BiMM', 'MinSeg', 'N-gram', 'HMM', 'CRF', 'BiMM1\n(迭代优化后)']
p_scores = [59.75, 60.02, 60.05, 60.02, 59.81, 71.15, 84.98, 75.14]
r_scores = [72.59, 72.91, 72.95, 72.91, 72.66, 70.57, 83.60, 81.16]
f1_scores = [65.55, 65.84, 65.88, 65.84, 65.61, 70.86, 84.28, 78.04]

x = np.arange(len(labels))
width = 0.25  # 柱子宽度

# 创建图表对象
fig, ax = plt.subplots(figsize=(14, 7))

# 绘制三组柱状图
rects1 = ax.bar(x - width, p_scores, width, label='精准率 (Precision)', color='#5D9CE8', edgecolor='white')
rects2 = ax.bar(x, r_scores, width, label='召回率 (Recall)', color='#FFB03B', edgecolor='white')
rects3 = ax.bar(x + width, f1_scores, width, label='F1 测度 (F1-Measure)', color='#4ECDC4', edgecolor='white')

# 细节修饰与标注
ax.set_ylabel('百分比 (%)', fontsize=12)
ax.set_title('张琪琛_自动分词实验各模型性能宏观对比图', fontsize=16, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(50, 90) # 限定Y轴范围让对比更明显

# 在柱子上方标注具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 增加网格线并设置图例
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=11)

# 调整布局并展示图表
fig.tight_layout()
plt.show()