"""全局配置。所有密钥只从环境变量读取，不落盘、不进仓库。"""
import os
from pathlib import Path

# 国内网络下载 embedding 模型走镜像，必须在 import fastembed 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
EVAL_OUT = BASE_DIR / "eval_report.md"

COLLECTION = "courseware_ch7"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# BGE 中文模型检索时建议给 query 加指令前缀，能明显提升召回
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
