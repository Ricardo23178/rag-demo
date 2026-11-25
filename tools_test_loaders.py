"""自测：验证多格式加载器（docx / md / txt）与切块结果。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document

from ingest import build_chunks, load_document

tmp = Path(tempfile.mkdtemp(prefix="rag_loader_"))

# --- markdown ---
md = tmp / "guide.md"
md.write_text(
    "# 家属绑定说明\n\n"
    "打开家属协同页，点击发起绑定即可生成邀请码。邀请码有效期 24 小时，"
    "过期后需要重新生成。绑定成功后，家属可在授权范围内查看被照护方的用药计划状态"
    "与异常事件，但不能修改对方的用药计划。\n\n"
    "## 解绑规则\n\n"
    "解绑后家属将无法继续查看被照护方的异常事件，但历史记录保留以便追溯。"
    "被照护方可以随时重新发起绑定，同一对用户解绑后再次绑定不受限制。"
    "如果家属主动解绑，被照护方会收到一条系统通知。\n",
    encoding="utf-8",
)

# --- docx（带标题样式 + 表格） ---
docx_path = tmp / "manual.docx"
document = Document()
document.add_heading("异常事件处理", level=1)
document.add_paragraph(
    "异常事件由提醒任务到期未确认时自动生成，状态依次为未处理、已通知、已知晓、已关闭。"
    "家属在授权范围内可以查看到被照护方的异常事件列表，并根据事件类型决定是否联系对方。"
    "超过设定时长仍未处理的事件，会在管理端统计中标记为待跟进。"
)
table = document.add_table(rows=2, cols=2)
table.cell(0, 0).text = "状态"
table.cell(0, 1).text = "含义"
table.cell(1, 0).text = "OPEN"
table.cell(1, 1).text = "尚未通知家属"
document.save(str(docx_path))

# --- txt ---
txt = tmp / "faq.txt"
txt.write_text(
    "问：忘记密码怎么办？\n"
    "答：在登录页点击找回密码，通过手机号验证后重置。若手机号也已停用，"
    "需要联系平台管理员在管理端重置密码，重置后首次登录会要求修改初始密码。\n",
    encoding="utf-8",
)

for path in (md, docx_path, txt):
    pages = load_document(path)
    chunks = build_chunks(pages, set(), path.name)
    print(f"\n=== {path.name}：{len(pages)} 个块，{len(chunks)} 个 chunk ===")
    for chunk in chunks:
        print(f"  [{chunk['section']}] p.{chunk['pages']} {len(chunk['text'])} 字")
        print("   ", chunk["text"].replace("\n", " / ")[:120])
