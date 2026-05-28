"""项目全局配置"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_CSV = os.path.join(DATA_DIR, '上市公司招聘数据2026.csv')

# 模型路径
MODEL_DIR = os.path.join(BASE_DIR, 'models')
INDEX_FILE = os.path.join(MODEL_DIR, 'inverted_index.pkl')
TFIDF_MATRIX = os.path.join(MODEL_DIR, 'tfidf_matrix.npz')
VOCAB_MAP = os.path.join(MODEL_DIR, 'vocab_map.json')
CLASSIFIER_MODEL = os.path.join(MODEL_DIR, 'classifier.joblib')
CLUSTER_MODEL = os.path.join(MODEL_DIR, 'cluster.joblib')
KEYWORDS_FILE = os.path.join(MODEL_DIR, 'keywords.json')

# 输出根目录 + 各脚本子目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DIR_1_EXPLORE = os.path.join(OUTPUT_DIR, '1_exploration')
DIR_2_PREPROC = os.path.join(OUTPUT_DIR, '2_preprocess')
DIR_3_CLEANER = os.path.join(OUTPUT_DIR, '3_cleaner')
DIR_4_SAMPLE = os.path.join(OUTPUT_DIR, '4_sample')
DIR_6_PREP_SEG = os.path.join(OUTPUT_DIR, '6_prep_seg')
DIR_7_SEG_EVAL = os.path.join(OUTPUT_DIR, '7_seg_eval')
DIR_8_SEG = os.path.join(OUTPUT_DIR, '8_full_seg')
DIR_9_NGRAM = os.path.join(OUTPUT_DIR, '9_ngram')
DIR_10_KW = os.path.join(OUTPUT_DIR, '10_keywords')
DIR_12_CLASSIFY = os.path.join(OUTPUT_DIR, '12_classify')
DIR_13_CLASSIFY = os.path.join(OUTPUT_DIR, '13_classify')
DIR_15_CLUSTER = os.path.join(OUTPUT_DIR, '15_cluster')
DIR_16_SIMILAR = os.path.join(OUTPUT_DIR, '16_similar')

# 各脚本输出文件路径
# 1_exploration
STATS_SUMMARY = os.path.join(DIR_1_EXPLORE, 'stats_summary.json')

# 2_preprocess
PROCESSED_CSV = os.path.join(DIR_2_PREPROC, '上市公司招聘数据_processed.csv')
PREPROCESS_REPORT = os.path.join(DIR_2_PREPROC, 'preprocess_report.json')

# 3_cleaner
CLEANED_CSV = os.path.join(DIR_3_CLEANER, '上市公司招聘数据_cleaned.csv')
CLEANER_REPORT = os.path.join(DIR_3_CLEANER, 'cleaner_report.json')
QUALITY_HTML = os.path.join(DIR_3_CLEANER, '数据质量报告.html')

# 4_sample
SAMPLED_CSV = os.path.join(DIR_4_SAMPLE, '分词采样结果_200条.csv')

# 6_prep_seg
SEG_RAW_TXT = os.path.join(DIR_6_PREP_SEG, 'seg_200_raw.txt')
SEG_JIEBA_TXT = os.path.join(DIR_6_PREP_SEG, 'seg_200_jieba.txt')
SEG_META_JSON = os.path.join(DIR_6_PREP_SEG, 'seg_200_meta.json')

# 7_seg_eval
DICT_FILE = os.path.join(DIR_7_SEG_EVAL, 'dic_cleaned.txt')
GOLD_STANDARD = os.path.join(DIR_7_SEG_EVAL, 'gold_standard.txt')
EVAL_JSON = os.path.join(DIR_7_SEG_EVAL, 'seg_eval_report.json')
EVAL_PNG = os.path.join(DIR_7_SEG_EVAL, 'seg_eval_chart.png')
EVAL_SUMMARY = os.path.join(DIR_7_SEG_EVAL, 'seg_eval_summary.txt')

# 通用输出文件
STOPWORDS_FILE = os.path.join(OUTPUT_DIR, 'stopwords.txt')
NGRAM_REPORT = os.path.join(OUTPUT_DIR, 'ngram_report.json')
SEGMENTED_CSV = os.path.join(DIR_8_SEG, '上市公司招聘数据_segmented.csv')
CLUSTER_VIS_DATA = os.path.join(OUTPUT_DIR, 'cluster_vis.json')
EVAL_REPORT = os.path.join(OUTPUT_DIR, 'eval_report.json')

# 检索参数
SEARCH_PAGE_SIZE = 20
TITLE_WEIGHT = 2.0
CONTENT_WEIGHT = 1.0
COMPANY_BONUS = 0.1
CACHE_TTL = 300

# 分类与聚类参数
CLASSIFICATION_K = 30
CLUSTER_N_CLUSTERS = 5
LDA_N_TOPICS = 10
TEXTRANK_WINDOW = 5
TEXTRANK_DAMPING = 0.85

# Jieba 配置
JIEBA_DICT = None
