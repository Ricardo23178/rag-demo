"""在线问答：检索 Chroma -> 组装 Prompt -> 调用 DeepSeek 生成带引用的回答。"""
from __future__ import annotations

from dataclasses import dataclass

# 必须先导入 config：它在 fastembed / huggingface_hub 加载前设置 HF_ENDPOINT 镜像
from config import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBED_MODEL,
    MIN_SCORE,
    TOP_K,
)

import chromadb
from fastembed import TextEmbedding
from openai import OpenAI

SYSTEM_PROMPT = """你是《数据库系统概论》课件的问答助手。必须遵守：
1. 只能依据【资料】回答，不得使用资料以外的知识；
2. 资料不足以回答时，直接说“课件中未提及”，并说明缺什么信息，不要猜测；
3. 用简体中文，分点作答，每点尽量简短；
4. 每条件结论后用 [小节 p.页码] 标注来源，例如 [7.1 数据库设计概述 p.3]；
5. 不要编造资料中没有的数字、术语或结论。"""


@dataclass
class Hit:
    text: str
    section: str
    pages: str
    score: float

    @property
    def label(self) -> str:
        return f"{self.section} p.{self.pages}"


class RagEngine:
    def __init__(self, collection_name: str | None = None) -> None:
        self._name = collection_name or COLLECTION
        self._model: TextEmbedding | None = None
        self._collection = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=EMBED_MODEL)
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = client.get_collection(self._name)
        return self._collection

    def retrieve(self, question: str, k: int = TOP_K) -> list[Hit]:
        vector = list(self.model.query_embed([question]))[0].tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                Hit(
                    text=doc,
                    section=str(meta.get("section", "")),
                    pages=str(meta.get("pages", "")),
                    score=round(1 - float(distance), 4),
                )
            )
        return hits

    def build_messages(
        self, question: str, hits: list[Hit], history: list[dict] | None = None
    ) -> list[dict]:
        context = "\n\n".join(
            f"[资料{i}] 来源：{hit.label}\n{hit.text}" for i, hit in enumerate(hits, start=1)
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-4:])
        messages.append(
            {"role": "user", "content": f"【资料】\n{context}\n\n【问题】\n{question}"}
        )
        return messages

    def is_low_confidence(self, hits: list[Hit], min_score: float = MIN_SCORE) -> bool:
        """top1 相似度低于阈值时视为「资料里没有」，避免拿无关片段硬答。"""
        return not hits or hits[0].score < min_score

    def stream_answer(
        self, question: str, hits: list[Hit], history: list[dict] | None = None
    ):
        if self.is_low_confidence(hits):
            top = hits[0].score if hits else 0.0
            yield (
                f"资料中未找到相关内容（最高相似度 {top:.2f}，低于阈值 {MIN_SCORE:.2f}）。"
                "请换个问法，或确认该内容是否在这份资料范围内。"
            )
            return
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY")
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        stream = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=self.build_messages(question, hits, history),
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def answer_text(
        self, question: str, hits: list[Hit], history: list[dict] | None = None
    ) -> str:
        return "".join(self.stream_answer(question, hits, history))


def main() -> None:
    engine = RagEngine()
    print("RAG 已就绪，输入问题（空行退出）")
    while True:
        question = input("\n问：").strip()
        if not question:
            break
        hits = engine.retrieve(question)
        print("\n命中片段：")
        for i, hit in enumerate(hits, start=1):
            print(f"  [{i}] {hit.label}  相似度={hit.score}")
        print("\n答：", end="", flush=True)
        for token in engine.stream_answer(question, hits):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
