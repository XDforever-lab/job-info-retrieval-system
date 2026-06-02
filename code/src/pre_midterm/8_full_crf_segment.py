"""
CRF 全量分词脚本 —— 覆盖 To Do List #82 ~ #85

    #82: 用 CRF 对全量 93k 条职位描述进行分词
    #83: 对全量 93k 条招聘岗位名称进行分词
    #84: 保存全量分词结果（追加 _seg 字段）
    #85: 生成分词后全量数据集 data_segmented.csv

用法:
    conda activate my_project
    python src/8_full_crf_segment.py
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pycrfsuite
import sklearn_crfsuite
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (CLEANED_CSV, GOLD_STANDARD, OUTPUT_DIR, SEGMENTED_CSV,
                    DIR_7_SEG_EVAL, DICT_FILE)

# ── CRF 分词器 (复用 7_segment_and_evaluate.py 的 CRFSeg) ──

def get_word_tags(word):
    if len(word) == 1:
        return ["S"]
    elif len(word) == 2:
        return ["B", "E"]
    else:
        return ["B"] + ["I"] * (len(word) - 2) + ["E"]


def split_text_blocks(text):
    blocks = re.split(r"([一-龥]+)", text)
    return [b for b in blocks if b]


class CRFSegmenter:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=0.1, c2=0.1,
            max_iterations=100, all_possible_transitions=True
        )

    def _feat(self, text, i):
        ch = text[i]
        f = {"bias": 1.0, "char": ch}
        if i > 0:
            f["char-1"] = text[i - 1]
            f["char-1:0"] = text[i - 1] + ch
        else:
            f["BOS"] = True
        if i < len(text) - 1:
            f["char+1"] = text[i + 1]
            f["char0:+1"] = ch + text[i + 1]
        else:
            f["EOS"] = True
        return f

    def train(self, gold_path):
        print("  从金标准训练 CRF 模型...")
        X, y = [], []
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                words = [w for w in line.strip().split("/") if w.strip()
                         and re.match(r"^[一-龥]+$", w)]
                if not words:
                    continue
                tags = []
                for w in words:
                    tags.extend(get_word_tags(w))
                chars = "".join(words)
                if chars:
                    X.append([self._feat(chars, i) for i in range(len(chars))])
                    y.append(tags)
        if X:
            self.model.fit(X, y)
        print(f"  训练完成，{len(X)} 条样本")

    def segment_text(self, text):
        """对任意文本（含中英文混合）进行 CRF 分词"""
        if not text or not isinstance(text, str):
            return ""
        result = []
        for block in split_text_blocks(text):
            if re.match(r"^[一-龥]+$", block):
                chars = block
                feats = [self._feat(chars, i) for i in range(len(chars))]
                tags = self.model.predict_single(feats)
                word = ""
                for ch, tag in zip(chars, tags):
                    word += ch
                    if tag in ("E", "S"):
                        result.append(word)
                        word = ""
            else:
                result.append(block)
        return "/".join(result)


# ── 主流程 ──

def main():
    t0 = time.time()

    print("=" * 60)
    print("  CRF 全量分词 #82 ~ #85")
    print("=" * 60)

    # 1. 训练 CRF
    print("\n[1] 训练 CRF 模型")
    crf = CRFSegmenter()
    crf.train(GOLD_STANDARD)

    # 2. 加载清洗后数据
    print(f"\n[2] 加载数据: {CLEANED_CSV}")
    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig", low_memory=False)
    print(f"    记录数: {len(df):,}")
    print(f"    字段: {list(df.columns)}")

    # 3. 分词 职位描述 (#82)
    print(f"\n[3] #82: CRF 分词 - 职位描述 ({len(df):,} 条) ...")
    desc_seg = []
    for i in tqdm(range(len(df)), desc="  职位描述分词", ncols=80):
        text = str(df.iloc[i].get("职位描述", ""))
        desc_seg.append(crf.segment_text(text) if text and text != "nan" else "")
    df["职位描述_分词"] = desc_seg

    # 4. 分词 招聘岗位 (#83)
    print(f"\n[4] #83: CRF 分词 - 招聘岗位 ({len(df):,} 条) ...")
    title_seg = []
    for i in tqdm(range(len(df)), desc="  岗位名称分词", ncols=80):
        text = str(df.iloc[i].get("招聘岗位", ""))
        title_seg.append(crf.segment_text(text) if text and text != "nan" else "")
    df["招聘岗位_分词"] = title_seg

    # 5. 保存 (#84, #85)
    os.makedirs(os.path.dirname(SEGMENTED_CSV), exist_ok=True)
    out_path = SEGMENTED_CSV
    print(f"\n[5] 保存: {out_path}")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    out_size = os.path.getsize(out_path)
    print(f"    大小: {out_size / 1024 / 1024:.1f} MB")

    # 预览
    print(f"\n{'=' * 60}")
    print("  分词样例（前 3 条）：")
    for i in range(min(3, len(df))):
        raw = str(df.iloc[i]["职位描述"])[:80]
        seg = str(df.iloc[i]["职位描述_分词"])[:80]
        print(f"\n  [{i+1}] 原文: {raw}...")
        print(f"      分词: {seg}...")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  #82 ~ #85 完成! 耗时 {elapsed / 60:.1f} 分钟")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
