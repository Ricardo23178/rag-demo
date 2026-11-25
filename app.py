"""Streamlit 界面：左侧提问，右侧/下方展示回答与命中的原文片段。"""
import streamlit as st

from rag import RagEngine

st.set_page_config(page_title="课件问答 · RAG Demo", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="正在加载向量模型...")
def get_engine() -> RagEngine:
    return RagEngine()


engine = get_engine()

with st.sidebar:
    st.subheader("参数")
    top_k = st.slider("召回片段数 top_k", min_value=1, max_value=8, value=4)
    show_context = st.checkbox("显示命中原文", value=True)
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("生成：DeepSeek（deepseek-chat）")
    st.caption("向量：BAAI/bge-small-zh-v1.5")
    st.caption("向量库：Chroma（本地持久化）")

st.title("课件问答 Demo")
st.caption("语料：《数据库系统概论》第七章 数据库设计（94 页课件）")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_context and message.get("hits"):
            with st.expander(f"命中片段 {len(message['hits'])} 条"):
                for i, hit in enumerate(message["hits"], start=1):
                    st.markdown(f"**[{i}] {hit['label']}** · 相似度 `{hit['score']}`")
                    st.text(hit["text"][:500])

question = st.chat_input("例如：数据库设计分为哪几个阶段？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = engine.retrieve(question, top_k)
        placeholder = st.empty()
        answer = ""
        try:
            for token in engine.stream_answer(question, hits):
                answer += token
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
        except Exception as exc:  # 缺 key、网络异常等
            answer = f"调用失败：{exc}"
            placeholder.error(answer)

        hit_payload = [
            {"label": h.label, "score": h.score, "text": h.text} for h in hits
        ]
        if show_context:
            with st.expander(f"命中片段 {len(hits)} 条"):
                for i, hit in enumerate(hit_payload, start=1):
                    st.markdown(f"**[{i}] {hit['label']}** · 相似度 `{hit['score']}`")
                    st.text(hit["text"][:500])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hit_payload}
    )
