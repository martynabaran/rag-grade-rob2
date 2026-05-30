"""
app.py — Standalone Gradio chatbot for GRADE and RoB2.

Run locally:
    python app.py
Then open http://127.0.0.1:7860 in your browser.

Features:
  - Per-user session isolation (gr.State, one UUID per browser tab)
  - Source links embedded in every assistant reply
  - Per-message thumbs up / down (hover over any reply to see them)
  - Optional free-text comment saved alongside the rating
  - All feedback appended to feedback.jsonl
"""

import json
import os
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("MODEL_NAME", "google/gemini-2.5-flash")

import gradio as gr
from langchain_community.vectorstores import FAISS

from rag_core import get_embeddings, rag_answer_with_sources, build_llm

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GRADE_SYSTEM_PROMPT = (
    "You are an expert assistant specialising in the GRADE methodology for evidence-based "
    "healthcare guidelines. Answer the user's question based ONLY on the retrieved GRADE Book "
    "content below. Do not fabricate or speculate beyond what is in the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Use precise GRADE terminology (certainty of evidence, rating down/up, domains, etc.)."
)

ROB2_SYSTEM_PROMPT = (
    "You are an expert assistant specialising in the Cochrane Risk of Bias tool 2.0 (RoB2). "
    "RoB2 assesses risk of bias in randomized trials across five domains: "
    "(1) randomization process, (2) deviations from intended interventions, "
    "(3) missing outcome data, (4) outcome measurement, (5) selection of the reported result. "
    "Answer the user's question based ONLY on the retrieved RoB2 content below. "
    "Do not fabricate or speculate beyond what is in the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Use precise RoB2 terminology (signalling questions, bias judgments: Low/Some concerns/High, etc.)."
)

# ---------------------------------------------------------------------------
# Load indices and LLM once at startup
# ---------------------------------------------------------------------------

print("Loading GRADE index...")
grade_index = FAISS.load_local(
    ".faiss_cache/grade", get_embeddings(), allow_dangerous_deserialization=True
)
print("Loading RoB2 index...")
rob2_index = FAISS.load_local(
    ".faiss_cache/rob2", get_embeddings(), allow_dangerous_deserialization=True
)
print("Building LLM...")
llm = build_llm()
print("Ready — open http://127.0.0.1:7860")

# ---------------------------------------------------------------------------
# Feedback persistence
# ---------------------------------------------------------------------------

FEEDBACK_FILE = Path("feedback.jsonl")


def _save_feedback(
    *,
    session_id: str,
    domain: str,
    question: str,
    answer: str,
    sources: list[str],
    rating: str,        # "up" | "down" | "comment"
    comment: str = "",
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "domain": domain,
        "question": question,
        "answer": answer,
        "sources": sources,
        "rating": rating,
        "comment": comment,
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n📚 **Sources:**"]
    for src in sources:
        lines.append(f"- [{src}]({src})" if src.startswith("http") else f"- `{src}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI factory — one call per domain tab
# ---------------------------------------------------------------------------

def build_chat_tab(*, domain: str, index, system_prompt: str, examples: list[str]):
    # Per-session state — each browser tab gets its own copy
    session_id = gr.State(lambda: str(uuid.uuid4()))
    last_qa = gr.State({"question": "", "answer": "", "sources": []})

    # Chatbot — thumbs appear on hover when .like() is bound below
    chatbot = gr.Chatbot(
        show_copy_button=True,
        height=500,
        type="messages",
        label=f"{domain} Assistant",
    )

    # Message input
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your question and press Enter…",
            show_label=False,
            scale=9,
            autofocus=True,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)

    gr.Examples(examples=examples, inputs=msg_box, label="Example questions")

    # Comment box — always visible, submitted together with the thumb rating
    with gr.Row():
        comment_box = gr.Textbox(
            label="💬 Optional comment on the last reply",
            placeholder="Add a comment before clicking 👍 or 👎 on the message above…",
            lines=2,
            scale=1,
        )

    feedback_status = gr.Markdown("")

    # ---- Event: send message ------------------------------------------------

    def respond(message, history, sid, qa):
        if not message.strip():
            return history, "", qa

        answer, _ctx, sources = rag_answer_with_sources(
            message, index, llm, system_prompt=system_prompt
        )
        full_answer = answer + _format_sources(sources)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_answer},
        ]
        new_qa = {"question": message, "answer": answer, "sources": sources}
        return history, "", new_qa

    for trigger in (send_btn.click, msg_box.submit):
        trigger(
            respond,
            inputs=[msg_box, chatbot, session_id, last_qa],
            outputs=[chatbot, msg_box, last_qa],
        )

    # ---- Event: thumb up / down on any message bubble ----------------------

    def on_like(event: gr.LikeData, comment, sid, qa):
        rating = "up" if event.liked else "down"
        icon = "👍" if event.liked else "👎"
        if qa.get("question"):
            _save_feedback(
                session_id=sid,
                domain=domain,
                question=qa["question"],
                answer=qa["answer"],
                sources=qa.get("sources", []),
                rating=rating,
                comment=comment.strip(),
            )
            return gr.update(value=f"✅ {icon} saved — thank you!")
        return gr.update(value="")

    chatbot.like(on_like, inputs=[comment_box, session_id, last_qa], outputs=[feedback_status])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

with gr.Blocks(title="Evidence-Based Medicine Assistants", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏥 Evidence-Based Medicine RAG Assistants\n"
        "Answers are grounded strictly in official methodology documentation. "
        "**Hover over any assistant reply** to rate it 👍 / 👎. "
        "Add an optional comment in the box below the chat before clicking the thumb."
    )

    with gr.Tabs():
        with gr.TabItem("📗 GRADE Methodology"):
            gr.Markdown(
                "Ask questions about the **GRADE** methodology — certainty of evidence, "
                "rating domains, evidence profiles, and more."
            )
            build_chat_tab(
                domain="GRADE",
                index=grade_index,
                system_prompt=GRADE_SYSTEM_PROMPT,
                examples=[
                    "What are the four levels of certainty of evidence in GRADE?",
                    "What are the five domains for rating down certainty of evidence?",
                    "When can certainty of evidence be rated up?",
                    "What is the difference between imprecision and inconsistency in GRADE?",
                ],
            )

        with gr.TabItem("📘 RoB2 Tool"):
            gr.Markdown(
                "Ask questions about the **Cochrane RoB2** tool — five domains, "
                "signalling questions, domain-level and overall bias judgments."
            )
            build_chat_tab(
                domain="RoB2",
                index=rob2_index,
                system_prompt=ROB2_SYSTEM_PROMPT,
                examples=[
                    "What are the five domains assessed in RoB2?",
                    "What does 'Some concerns' mean in an overall RoB2 judgment?",
                    "If one domain is rated High, what is the overall bias judgment?",
                    "What are the signalling questions for the randomization domain?",
                ],
            )

if __name__ == "__main__":
    demo.launch()
