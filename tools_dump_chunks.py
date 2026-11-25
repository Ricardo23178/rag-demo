"""调试用：导出集合里的全部切块，便于设计评测问题与检查切块质量。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb

from config import CHROMA_DIR, COLLECTION

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_collection(COLLECTION)
data = col.get(include=["documents", "metadatas"])

rows = sorted(zip(data["ids"], data["documents"], data["metadatas"]), key=lambda x: x[0])
out = []
for cid, doc, meta in rows:
    out.append(f"--- {cid} | {meta['section']} | p.{meta['pages']} | {len(doc)} 字 ---")
    out.append(doc[:300].replace("\n", " / "))
    out.append("")
out.append(f"总计 {len(rows)} 块")
print("\n".join(out))
