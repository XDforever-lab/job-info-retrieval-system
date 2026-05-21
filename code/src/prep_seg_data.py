"""
分词评测数据准备脚本 —— 覆盖 To Do List #52 ~ #53

    #52: 从 200 条抽样预分词数据（output/分词采样结果_200条.csv）
            提取作为分词评测样本
    #53: 将 200 条职位描述转为纯文本 txt，换行分隔，
            同时生成"去分词符的原文"和"保留分词符的机切版"两份 txt

用法:
    conda activate my_project
    python src/prep_seg_data.py

输出:
    output/seg_200_raw.txt        纯原文（无分词符），供人工精标
    output/seg_200_jieba.txt      机切版（保留/分隔符），供评测对比
    output/seg_200_meta.json      元信息（编号映射表）
"""

import json
import os
import sys
import re

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

# ── 配置 ────────────────────────────────────────

INPUT_CSV = os.path.join(OUTPUT_DIR, "分词采样结果_200条.csv")
OUTPUT_RAW = os.path.join(OUTPUT_DIR, "seg_200_raw.txt")
OUTPUT_JIEBA = os.path.join(OUTPUT_DIR, "seg_200_jieba.txt")
OUTPUT_META = os.path.join(OUTPUT_DIR, "seg_200_meta.json")


# ── 主逻辑 ──────────────────────────────────────

def main():
    print("=" * 60)
    print("  分词评测数据准备 (#52 ~ #53)")
    print("=" * 60)

    # 加载
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"\n  [加载] {INPUT_CSV}")
    print(f"  行数: {len(df)}, 字段: {list(df.columns)}")

    if "职位描述" not in df.columns:
        raise KeyError(f"CSV 中缺少'职位描述'列，当前列: {list(df.columns)}")

    # 提取职位描述列，去掉空值
    descs = df["职位描述"].dropna().astype(str).tolist()
    descs = [d.strip() for d in descs if d.strip()]
    print(f"  有效职位描述: {len(descs)} 条")

    # ── 生成纯原文 txt ──
    # 去掉 jieba 分词符 "/"，还原为纯文本
    raw_lines = []
    for d in descs:
        # 把 "/" 分隔符去掉，还原原文
        # 注意：数字/英文 中的 "/" 不删（如 "15k/月"）
        raw = re.sub(r'(?<=[一-鿿，。；：！？、""''）\)】\]])\s*/\s*(?=[一-鿿])', '', d)
        raw = re.sub(r'(?<=[一-鿿])\s*/\s*(?=[一-鿿，。；：！？、])', '', raw)
        raw = re.sub(r'(?<=[\x00-\x7f])\s*/\s*(?=[一-鿿])', '', raw)
        raw = re.sub(r'(?<=[一-鿿])\s*/\s*(?=[\x00-\x7f])', '', raw)
        raw = re.sub(r' {2,}', ' ', raw)
        raw = raw.strip()
        raw_lines.append(raw)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        f.write("\n".join(raw_lines))
    print(f"\n  #53 [保存] 纯原文 txt → {OUTPUT_RAW}")
    print(f"      {len(raw_lines)} 行，每行一条职位描述（已去除分词符）")
    print(f"      用途：人工逐条精标分词金标准")

    # ── 生成机切版 txt ──
    # 保留 "/" 分隔符，每行一条
    jieba_lines = [d for d in descs]
    with open(OUTPUT_JIEBA, "w", encoding="utf-8") as f:
        f.write("\n".join(jieba_lines))
    print(f"\n  #52 [保存] 机切版 txt → {OUTPUT_JIEBA}")
    print(f"      {len(jieba_lines)} 行，保留 / 分词分隔符")
    print(f"      用途：与人工精标结果进行评测对比")

    # ── 元信息 JSON ──
    meta = {
        "数据来源": INPUT_CSV,
        "样本数": len(descs),
        "输出文件": {
            "纯原文": OUTPUT_RAW,
            "机切版": OUTPUT_JIEBA,
        },
        "编号映射": {
            str(i): str(df.iloc[i].get("编号", i + 1))
            for i in range(len(descs))
        },
        "说明": (
            "seg_200_raw.txt: 去除分词符后的纯文本，用于人工精标金标准。"
            "seg_200_jieba.txt: 保留 / 分隔符的机器分词结果，用于自动评测对比。"
            "两条文件的第 N 行对应同一条职位描述，行号保持一致。"
        ),
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n  [保存] 元信息 → {OUTPUT_META}")

    # ── 预览 ──
    print(f"\n{'─' * 40}")
    print("  原文预览（第 1 条前 120 字）:")
    print(f"  {raw_lines[0][:120]}...")
    print(f"\n  机切预览（第 1 条前 120 字）:")
    print(f"  {jieba_lines[0][:120]}...")

    print(f"\n{'=' * 60}")
    print("  数据准备完成 ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
