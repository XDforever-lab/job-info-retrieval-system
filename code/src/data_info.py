"""数据基本信息查看脚本 —— 运行后输出完整的数据探索报告"""

import sys
sys.path.insert(0, '..')

from src.data_loader import load_raw_data, get_data_summary


def print_separator(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    # 1. 加载数据
    print_separator("加载原始数据")
    df = load_raw_data()

    # 2. 基本信息
    print_separator("基本信息")
    print(f"  总记录数: {len(df):,}")
    print(f"  字段数量: {len(df.columns)}")
    print(f"  字段列表: {', '.join(df.columns)}")

    # 3. 各字段缺失率
    print_separator("字段缺失统计")
    summary = get_data_summary(df)
    for col, info in summary['missing'].items():
        flag = " ⚠️" if info['pct'] > 10 else ""
        print(f"  {col}: {info['count']} 缺失 ({info['pct']}%){flag}")

    # 4. 各字段数据类型
    print_separator("字段类型")
    for col, dtype in summary['dtypes'].items():
        print(f"  {col}: {dtype}")

    # 5. 前5行预览
    print_separator("前 5 行数据预览")
    
    print(df.head(5).to_string())

    # 6. 数值列统计
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print_separator("数值列统计")
        print(df[numeric_cols].describe().to_string())

    # 7. 唯一值多的分类列
    print_separator("分类列基数统计")
    for col in df.select_dtypes(include=['object']).columns:
        nunique = df[col].nunique()
        print(f"  {col}: {nunique} 个唯一值")


if __name__ == '__main__':
    main()
