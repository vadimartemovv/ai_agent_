import os
from io import BytesIO
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None


APP_TITLE = "Report AI Agent"
MODEL_PATH_ENV = "LLAMA_MODEL_PATH"


class SummaryResponse(BaseModel):
    summary: str


class QAResponse(BaseModel):
    answer: str


app = FastAPI(title=APP_TITLE)


_llm = None


def _load_llm() -> "Llama":
    global _llm
    if _llm is not None:
        return _llm
    if Llama is None:
        raise HTTPException(
            status_code=500,
            detail="llama-cpp-python is not available. Install dependencies.",
        )
    model_path = os.getenv(MODEL_PATH_ENV)
    if not model_path:
        raise HTTPException(
            status_code=500,
            detail=f"Set {MODEL_PATH_ENV} to a local GGUF model file.",
        )
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {model_path}",
        )
    _llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    return _llm


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)
    return "\n\n".join(pages_text).strip()


def _chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def _generate(llm: "Llama", prompt: str, max_tokens: int = 512) -> str:
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.95,
        stop=["</s>", "###"],
    )
    return (response["choices"][0]["text"] or "").strip()


def _build_summary_prompt(text: str) -> str:
    return (
        "Ты помощник-аналитик. Сделай саммари на 5-10 предложений. "
        "Только по содержанию отчета, без выдумок.\n\n"
        f"ОТЧЕТ:\n{text}\n\nСАММАРИ:\n"
    )


def _build_qa_prompt(text: str, question: str) -> str:
    return (
        "Ты помощник-аналитик. Отвечай только по содержанию отчета. "
        "Если ответа нет в тексте, скажи, что в отчете это не указано.\n\n"
        f"ОТЧЕТ:\n{text}\n\nВОПРОС: {question}\nОТВЕТ:\n"
    )


@app.post("/summary", response_model=SummaryResponse)
async def summarize_report(file: UploadFile = File(...)) -> SummaryResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_bytes = await file.read()
    text = _extract_text_from_pdf(file_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    llm = _load_llm()
    chunks = _chunk_text(text)
    if len(chunks) == 1:
        summary = _generate(llm, _build_summary_prompt(chunks[0]), max_tokens=400)
        return SummaryResponse(summary=summary)

    # Map-reduce for long reports
    partial_summaries = []
    for chunk in chunks:
        partial = _generate(llm, _build_summary_prompt(chunk), max_tokens=250)
        if partial:
            partial_summaries.append(partial)
    combined = "\n".join(partial_summaries)
    final_summary = _generate(llm, _build_summary_prompt(combined), max_tokens=400)
    return SummaryResponse(summary=final_summary)


@app.post("/qa", response_model=QAResponse)
async def answer_question(
    question: str = Form(...),
    file: UploadFile = File(...),
) -> QAResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    file_bytes = await file.read()
    text = _extract_text_from_pdf(file_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    llm = _load_llm()
    chunks = _chunk_text(text)

    # If long, create a short context summary first
    if len(chunks) > 1:
        partial_summaries = []
        for chunk in chunks:
            partial = _generate(llm, _build_summary_prompt(chunk), max_tokens=200)
            if partial:
                partial_summaries.append(partial)
        context = "\n".join(partial_summaries)
    else:
        context = chunks[0]

    answer = _generate(llm, _build_qa_prompt(context, question), max_tokens=300)
    return QAResponse(answer=answer)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
