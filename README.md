# Report AI Agent (FastAPI + local LLM)

Веб‑сервис принимает PDF‑отчет, делает саммари (5–10 предложений) и отвечает на вопросы по содержимому.
Есть простая веб‑страница для запуска всех функций из браузера и потоковый статус выполнения.

## Быстрый старт

1) Установи зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Укажи модель:

```bash
export LLAMA_MODEL_PATH="/path/to/model.gguf"
```

Либо задай прямую ссылку и сервис сам скачает модель:
```bash
export LLAMA_MODEL_URL="https://.../model.gguf"
```

Рекомендованный вариант (качество/скорость): **Qwen2.5‑3B‑Instruct Q4_K_M** (~2.1 ГБ):
```bash
export LLAMA_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
```

3) Запусти сервер:

```bash
uvicorn app.main:app --reload
```

## API

### GET /
Веб‑страница с загрузкой PDF и кнопками для саммари и Q&A.

### POST /summary
- form-data: `file` (PDF)

Ответ:
```json
{ "summary": "..." }
```

### POST /qa
- form-data: `file` (PDF)
- form-data: `question` (строка)

Ответ:
```json
{ "answer": "..." }
```

### POST /debug_text
Возвращает первые ~4000 символов извлечённого текста и общую длину.

### POST /summary_stream
Возвращает прогресс в виде JSON‑строк (status/summary).

### POST /qa_stream
Возвращает прогресс в виде JSON‑строк (status/answer).

### GET /health
Ответ:
```json
{ "status": "ok" }
```

## Пример запроса

```bash
curl -X POST http://127.0.0.1:8000/summary \
  -F "file=@report.pdf"

curl -X POST http://127.0.0.1:8000/qa \
  -F "file=@report.pdf" \
  -F "question=Какой прогноз выручки на 2025 год?"
```

## Примечания
- Сервис работает полностью локально.
- Для больших отчетов используется map‑reduce саммари по чанкам.
- Если в отчете нет ответа, модель просится это явно сказать.
- В `/` доступна веб‑страница с формой и статусом выполнения.
