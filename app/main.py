import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None


APP_TITLE = "Report AI Agent"
MODEL_PATH_ENV = "LLAMA_MODEL_PATH"
MODEL_URL_ENV = "LLAMA_MODEL_URL"
DEFAULT_MODEL_PATH = "/models/model.gguf"


class SummaryResponse(BaseModel):
    summary: str


class QAResponse(BaseModel):
    answer: str


class DebugTextResponse(BaseModel):
    text: str
    length: int


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
    model_path = os.getenv(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    if not model_path:
        raise HTTPException(
            status_code=500,
            detail=f"Set {MODEL_PATH_ENV} to a local GGUF model file.",
        )
    if not os.path.exists(model_path):
        model_url = os.getenv(MODEL_URL_ENV)
        if not model_url:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Model file not found: {model_path}. "
                    f"Set {MODEL_URL_ENV} to auto-download a GGUF model."
                ),
            )
        _download_model(model_url, model_path)
    _llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    return _llm


def _download_model(url: str, dest_path: str) -> None:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download model: HTTP {response.status}",
                )
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model: {exc}",
        )


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


def _generate(
    llm: "Llama",
    prompt: str,
    max_tokens: int = 512,
    repeat_penalty: float = 1.2,
    frequency_penalty: float = 0.0,
) -> str:
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        stop=["</s>", "###"],
    )
    return (response["choices"][0]["text"] or "").strip()


def _build_summary_prompt(text: str) -> str:
    return (
        "You are an analyst assistant. Write in English. Produce EXACTLY 5–10 sentences. "
        "No meta commentary (e.g., 'the report says...'). "
        "No lists or numbering, only coherent prose. "
        "Use only facts from the report, no speculation. Do not repeat yourself.\n\n"
        f"REPORT:\n{text}\n\nSUMMARY (5–10 sentences):\n"
    )


def _build_qa_prompt(text: str, question: str) -> str:
    return (
        "You are an analyst assistant. Write in English. Answer only using the report content. "
        "No meta commentary or speculation. Do not repeat yourself. "
        "If the answer is not in the text, say: \"The report does not specify this.\".\n\n"
        f"REPORT:\n{text}\n\nQUESTION: {question}\nANSWER:\n"
    )


def _split_sentences(text: str) -> List[str]:
    # Very simple sentence splitter for RU/EN punctuation.
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            sentence = "".join(buf).strip()
            if sentence:
                parts.append(sentence)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _sanitize_summary(text: str) -> str:
    # Remove common numbering like "1." or "8." at line starts.
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped[:2].isdigit() and stripped[1:2] == ".":
            stripped = stripped[2:].lstrip()
        elif len(stripped) >= 3 and stripped[0].isdigit() and stripped[1:2].isdigit() and stripped[2:3] == ".":
            stripped = stripped[3:].lstrip()
        lines.append(stripped)
    return " ".join([l for l in lines if l]).strip()


def _is_repetitive(text: str, threshold: int = 6) -> bool:
    tokens = text.lower().split()
    if len(tokens) < 20:
        return False
    counts = {}
    for i in range(len(tokens) - 2):
        gram = " ".join(tokens[i : i + 3])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= threshold:
            return True
    return False


def _ensure_summary_quality(llm: "Llama", text: str, summary: str) -> str:
    cleaned = _sanitize_summary(summary)
    sentences = _split_sentences(cleaned)
    if 5 <= len(sentences) <= 10 and not _is_repetitive(cleaned):
        return cleaned
    # Retry once with an even stricter prompt
    retry_prompt = (
        "Перепиши саммари СТРОГО в 5–10 предложениях, "
        "без нумерации, без списков, без мета-ответов. "
        "Пиши по-русски. Только факты из отчета. Не повторяйся.\n\n"
        f"ОТЧЕТ:\n{text}\n\nСАММАРИ:\n"
    )
    retry = _generate(
        llm,
        retry_prompt,
        max_tokens=420,
        repeat_penalty=1.35,
        frequency_penalty=0.2,
    )
    cleaned_retry = _sanitize_summary(retry)
    sentences = _split_sentences(cleaned_retry)
    if len(sentences) > 10:
        return " ".join(sentences[:10]).strip()
    if len(sentences) < 5:
        return cleaned_retry
    return cleaned_retry


def _summarize_text(llm: "Llama", text: str) -> str:
    chunks = _chunk_text(text)
    if len(chunks) == 1:
        raw = _generate(llm, _build_summary_prompt(chunks[0]), max_tokens=400)
        return _ensure_summary_quality(llm, chunks[0], raw)

    partial_summaries = []
    for chunk in chunks:
        partial = _generate(llm, _build_summary_prompt(chunk), max_tokens=250)
        if partial:
            partial_summaries.append(partial)
    combined = "\n".join(partial_summaries)
    raw = _generate(llm, _build_summary_prompt(combined), max_tokens=400)
    return _ensure_summary_quality(llm, combined, raw)


def _answer_question(llm: "Llama", text: str, question: str) -> str:
    chunks = _chunk_text(text)
    if len(chunks) > 1:
        partial_summaries = []
        for chunk in chunks:
            partial = _generate(llm, _build_summary_prompt(chunk), max_tokens=200)
            if partial:
                partial_summaries.append(partial)
        context = "\n".join(partial_summaries)
    else:
        context = chunks[0]
    return _generate(llm, _build_qa_prompt(context, question), max_tokens=300)


@app.post("/summary", response_model=SummaryResponse)
async def summarize_report(file: UploadFile = File(...)) -> SummaryResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_bytes = await file.read()
    text = _extract_text_from_pdf(file_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    llm = _load_llm()
    summary = _summarize_text(llm, text)
    return SummaryResponse(summary=summary)


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
    answer = _answer_question(llm, text, question)
    return QAResponse(answer=answer)


@app.post("/debug_text", response_model=DebugTextResponse)
async def debug_text(file: UploadFile = File(...)) -> DebugTextResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_bytes = await file.read()
    text = _extract_text_from_pdf(file_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")
    preview = text[:4000]
    return DebugTextResponse(text=preview, length=len(text))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Report AI Agent</title>
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }
      .card { max-width: 720px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 12px; }
      h1 { margin: 0 0 12px; font-size: 24px; }
      label { display: block; margin: 12px 0 6px; font-weight: 600; }
      input[type="file"], input[type="text"] { width: 100%; padding: 8px; }
      button { margin-top: 12px; padding: 10px 16px; cursor: pointer; }
      pre { white-space: pre-wrap; background: #f6f6f6; padding: 12px; border-radius: 8px; }
      .row { display: flex; gap: 12px; }
      .row > button { flex: 1; }
      .muted { color: #666; font-size: 13px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Report AI Agent</h1>
      <div class="muted">Загрузи PDF, получи саммари или задай вопрос.</div>

      <label>PDF файл</label>
      <input id="file" type="file" accept="application/pdf"/>

      <div class="row">
        <button id="btn-summary">Сделать саммари</button>
        <button id="btn-debug">Показать извлечённый текст</button>
      </div>

      <label>Вопрос</label>
      <input id="question" type="text" placeholder="Например: Какой прогноз выручки на 2025 год?"/>
      <div class="row">
        <button id="btn-qa">Задать вопрос</button>
      </div>

      <label>Ответ</label>
      <pre id="output">—</pre>
    </div>

    <script>
      const fileInput = document.getElementById('file');
      const questionInput = document.getElementById('question');
      const output = document.getElementById('output');

      async function postForm(url, extraFields = {}) {
        const file = fileInput.files[0];
        if (!file) throw new Error('Выберите PDF файл.');
        const form = new FormData();
        form.append('file', file);
        for (const [k, v] of Object.entries(extraFields)) form.append(k, v);
        const res = await fetch(url, { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Ошибка запроса');
        return data;
      }

      async function postStream(url, extraFields = {}) {
        const file = fileInput.files[0];
        if (!file) throw new Error('Выберите PDF файл.');
        const form = new FormData();
        form.append('file', file);
        for (const [k, v] of Object.entries(extraFields)) form.append(k, v);
        const res = await fetch(url, { method: 'POST', body: form });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Ошибка запроса');
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buffer.indexOf('\\n')) >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line) continue;
            try {
              const msg = JSON.parse(line);
              if (msg.status) output.textContent = msg.status;
              if (msg.summary) output.textContent = msg.summary;
              if (msg.answer) output.textContent = msg.answer;
            } catch (_) {
              output.textContent = line;
            }
          }
        }
      }

      document.getElementById('btn-summary').addEventListener('click', async () => {
        output.textContent = 'Загрузка файла...';
        try {
          await postStream('/summary_stream');
        } catch (e) {
          output.textContent = String(e.message || e);
        }
      });

      document.getElementById('btn-debug').addEventListener('click', async () => {
        output.textContent = 'Извлекаю текст...';
        try {
          const data = await postForm('/debug_text');
          output.textContent = data.text || '—';
        } catch (e) {
          output.textContent = String(e.message || e);
        }
      });

      document.getElementById('btn-qa').addEventListener('click', async () => {
        output.textContent = 'Загрузка файла...';
        const q = questionInput.value.trim();
        if (!q) { output.textContent = 'Введите вопрос.'; return; }
        try {
          await postStream('/qa_stream', { question: q });
        } catch (e) {
          output.textContent = String(e.message || e);
        }
      });
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/summary_stream")
async def summarize_report_stream(file: UploadFile = File(...)) -> StreamingResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_bytes = await file.read()

    async def _gen():
        yield '{"status":"Читаю PDF..."}\n'
        yield '{"status":"Извлекаю текст..."}\n'
        text = _extract_text_from_pdf(file_bytes)
        if not text:
            yield '{"status":"В PDF нет извлекаемого текста."}\n'
            return
        yield '{"status":"Загружаю модель..."}\n'
        llm = _load_llm()
        chunks = _chunk_text(text)
        if len(chunks) == 1:
            yield '{"status":"Делаю саммари..."}\n'
            summary = _summarize_text(llm, text)
            yield f'{{"summary":{_json_escape(summary)}}}\n'
            return
        partial_summaries = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            yield f'{{"status":"Суммирую блок {idx}/{total}..."}}\n'
            partial = _generate(llm, _build_summary_prompt(chunk), max_tokens=250)
            if partial:
                partial_summaries.append(partial)
        combined = "\n".join(partial_summaries)
        yield '{"status":"Делаю итоговое саммари..."}\n'
        raw = _generate(llm, _build_summary_prompt(combined), max_tokens=400)
        summary = _ensure_summary_quality(llm, combined, raw)
        yield f'{{"summary":{_json_escape(summary)}}}\n'

    return StreamingResponse(_gen(), media_type="text/plain")


@app.post("/qa_stream")
async def answer_question_stream(
    question: str = Form(...),
    file: UploadFile = File(...),
) -> StreamingResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    file_bytes = await file.read()

    async def _gen():
        yield '{"status":"Читаю PDF..."}\n'
        yield '{"status":"Извлекаю текст..."}\n'
        text = _extract_text_from_pdf(file_bytes)
        if not text:
            yield '{"status":"В PDF нет извлекаемого текста."}\n'
            return
        yield '{"status":"Загружаю модель..."}\n'
        llm = _load_llm()
        chunks = _chunk_text(text)
        if len(chunks) > 1:
            partial_summaries = []
            total = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                yield f'{{"status":"Готовлю контекст {idx}/{total}..."}}\n'
                partial = _generate(llm, _build_summary_prompt(chunk), max_tokens=200)
                if partial:
                    partial_summaries.append(partial)
            context = "\n".join(partial_summaries)
        else:
            context = chunks[0]
        yield '{"status":"Ищу ответ..."}\n'
        answer = _generate(llm, _build_qa_prompt(context, question), max_tokens=300)
        yield f'{{"answer":{_json_escape(answer)}}}\n'

    return StreamingResponse(_gen(), media_type="text/plain")


def _json_escape(value: str) -> str:
    escaped = (
        value.replace("\\\\", "\\\\\\\\")
        .replace("\\n", "\\\\n")
        .replace("\\r", "\\\\r")
        .replace("\\t", "\\\\t")
        .replace('"', '\\"')
    )
    return f'"{escaped}"'
