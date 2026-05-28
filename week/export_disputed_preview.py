"""
导出争议文章前200词预览，供人工逐篇判断类别
输出：disputed_for_manual_review.csv
"""
import pandas as pd

pool = pd.read_csv("pooling_for_annotation.csv")
seg  = pd.read_csv("crf_seg_result.csv")

# 序号 -> 词列表
doc_words = {}
for _, row in seg.iterrows():
    words = [w.strip() for w in str(row["分词后内容"]).split("/") if w.strip()]
    doc_words[row["序号"]] = words

# 只取争议文章
disputed = pool[pool["完全一致"] == "否"].copy()

rows = []
for _, r in disputed.iterrows():
    seq_id = r["序号"]
    words = doc_words.get(seq_id, [])
    preview = " ".join(words[:200])  # 前200个词
    rows.append({
        "序号": seq_id,
        "文章预览(前200词)": preview,
        "原模型共识簇": int(r["共识簇"]),
        "人工判定类别(填0-4)": "",
    })

df = pd.DataFrame(rows)
df.to_csv("disputed_for_manual_review.csv", index=False, encoding="utf-8-sig")

print(f"已导出 {len(df)} 篇争议文章 → disputed_for_manual_review.csv")
print(f"类别对照: 0=党政人事  1=社会民生  2=思想文化  3=经济科技  4=国际外交")
