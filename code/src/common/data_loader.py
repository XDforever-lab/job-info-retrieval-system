"""数据加载与预处理工具"""

import pandas as pd
import os

# 项目根目录 (code/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_CSV = os.path.join(DATA_DIR, '上市公司招聘数据2026.csv')


def detect_encoding(filepath, sample_size=100000):
    """自动检测文件编码（处理 GBK/UTF-8 等）"""
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-16']
    for enc in encodings:
        try:
            with open(filepath, 'rb') as f:
                raw = f.read(sample_size)
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return 'utf-8'  # fallback


def load_raw_data(filepath=None):
    """加载原始招聘数据 CSV

    Returns:
        pd.DataFrame: 包含所有字段的原始数据
    """
    fp = filepath or RAW_CSV

    if not os.path.exists(fp):
        raise FileNotFoundError(f"数据文件不存在: {fp}")

    encoding = detect_encoding(fp)
    print(f"[data_loader] 检测到编码: {encoding}，开始加载...")

    df = pd.read_csv(fp, encoding=encoding)
    print(f"[data_loader] 加载完成: {len(df)} 条记录, {len(df.columns)} 个字段")

    return df


def get_data_summary(df):
    """生成数据摘要信息

    Returns:
        dict: 包含行数、字段列表及各字段缺失率等信息
    """
    summary = {
        'total_rows': len(df),
        'fields': list(df.columns),
        'field_count': len(df.columns),
        'missing': {},
        'dtypes': {}
    }

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)
        summary['missing'][col] = {
            'count': missing_count,
            'pct': missing_pct
        }
        summary['dtypes'][col] = str(df[col].dtype)

    return summary
