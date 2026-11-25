"""全局配置。密钥只从环境变量或本地 .env 读取，不进仓库。"""
import os
from pathlib import Path

# 国内网络下载 embedding 模型走镜像，必须在 import fastembed 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
EVAL_OUT = BASE_DIR / "eval_report.md"

# 本地 .env 优先注入环境变量（.env 已被 .gitignore 忽略）
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

COLLECTION = os.getenv("RAG_COLLECTION", "courseware_ch7")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# BGE 中文模型检索时建议给 query 加指令前缀，能明显提升召回
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
# 拒答门控：top1 相似度低于该值时不调用模型，直接回「未找到」
# 实测文档内问题 >=0.60，文档外 <=0.50，取中点
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.55"))

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
