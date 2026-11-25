"""离线建库：PDF -> 清洗 -> 按小节切块 -> 向量化 -> 写入 Chroma。

用法：
    python ingest.py                # 增量重建（先清空同名集合）
    python ingest.py --keep         # 不清空，直接追加
"""
import argparse
import re
from collections import Counter
from pathlib import Path

# 必须先导入 config：它在 fastembed / huggingface_hub 加载前设置 HF_ENDPOINT 镜像
from config import CHROMA_DIR, COLLECTION, DATA_DIR, EMBED_MODEL  # noqa: E402

import chromadb
from fastembed import TextEmbedding
from pypdf import PdfReader

# 形如 "7.1  数据库设计概述" / "7.1.1  数据库设计的特点"
HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+(\S.{1,40})$")
# 句子结束符：遇到这些才允许换行，否则视为排版折行，接回上一行
SENTENCE_END = tuple("。！？；：.!?;:")
# 列表项开头：这些行要单独成行，不能被上一行吞掉
LIST_START_RE = re.compile(r"^([（(]?\d+[)）.、]|[①-⑩]|[-•·●▪]|（续）)")
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 120


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def detect_boilerplate(pages: list[str], ratio: float = 0.5) -> set[str]:
    """页眉/页脚识别：在超过一半页面上重复出现的行，视为模板噪声。"""
    counter: Counter[str] = Counter()
    for text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            counter[line] += 1
    threshold = max(3, int(len(pages) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def clean_lines(text: str, boilerplate: set[str]) -> list[str]:
    kept = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in boilerplate:
            continue
        if len(line) <= 2 and not line.isdigit():
            continue
        kept.append(line)
    return kept


def build_chunks(
    pages: list[str], boilerplate: set[str], source: str, merge: bool = True
) -> list[dict]:
    """按小节聚块；merge=True 时把课件里的排版折行接回整句。"""
    chunks: list[dict] = []
    section = "开篇"
    paras: list[str] = []
    pages_used: list[int] = []
    buf_len = 0

    def flush() -> None:
        nonlocal paras, pages_used, buf_len
        body = "\n".join(paras).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            page_start, page_end = pages_used[0], pages_used[-1]
            pages_label = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
            chunks.append(
                {
                    "text": f"【{section}】\n{body}",
                    "section": section,
                    "pages": pages_label,
                    "page_start": page_start,
                    "source": source,
                }
            )
        paras, pages_used, buf_len = [], [], 0

    for page_no, text in enumerate(pages, start=1):
        for line in clean_lines(text, boilerplate):
            match = HEADING_RE.match(line)
            if match:
                flush()
                section = f"{match.group(1)} {match.group(2)}".strip()
                continue
            # 上一行没有句末标点、本行也不是列表项 → 说明是排版折行，接回去
            if (
                merge
                and paras
                and not paras[-1].endswith(SENTENCE_END)
                and not LIST_START_RE.match(line)
            ):
                prev = paras[-1]
                # 短行多半是小标题或列表项，补一个分隔符，避免两句粘成一句
                separator = "：" if len(prev) <= 8 else ("；" if len(prev) <= 18 else "")
                paras[-1] = prev + separator + line
            else:
                paras.append(line)
            buf_len += len(line)
            if not pages_used or pages_used[-1] != page_no:
                pages_used.append(page_no)
            if buf_len >= MAX_CHUNK_CHARS:
                flush()

    flush()
    return chunks


def dedupe(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for chunk in chunks:
        key = chunk["text"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="不清空已有集合")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="对照组：不剔除页眉页脚、不合并折行，写入 <集合>_raw 用于对比实验",
    )
    args = parser.parse_args()
    collection_name = COLLECTION + "_raw" if args.raw else COLLECTION

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"{DATA_DIR} 下没有找到 PDF")

    all_chunks: list[dict] = []
    for pdf in pdfs:
        pages = extract_pages(pdf)
        boiler = set() if args.raw else detect_boilerplate(pages)
        chunks = dedupe(build_chunks(pages, boiler, pdf.name, merge=not args.raw))
        total_chars = sum(len(p) for p in pages)
        kept_chars = sum(len(c["text"]) for c in chunks)
        print(
            f"[{pdf.name}] 页数={len(pages)} 原始字符={total_chars} "
            f"剔除模板行={len(boiler)} 切块={len(chunks)} 保留字符={kept_chars}"
        )
        if boiler:
            for line in list(boiler)[:3]:
                print(f"    - 模板行样例: {line[:40]}")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("没有切出任何内容，检查 PDF 是否为扫描件")

    print(f"共 {len(all_chunks)} 个块，开始向量化（{EMBED_MODEL}）...")
    model = TextEmbedding(model_name=EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    vectors = [vec.tolist() for vec in model.embed(texts, batch_size=16)]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if not args.keep:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"c{i:04d}" for i in range(len(all_chunks))],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "section": c["section"],
                "pages": c["pages"],
                "page_start": c["page_start"],
                "source": c["source"],
            }
            for c in all_chunks
        ],
    )
    print(f"写入完成，集合 {collection_name} 当前条数={collection.count()}")


if __name__ == "__main__":
    main()
