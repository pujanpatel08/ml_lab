"""Darvis query router: a FastAPI service in front of the fine-tuned model from module 07.

Run from this folder:
    uv run uvicorn app:app --port 8000 --reload

It expects an OpenAI-compatible model server (mlx_lm.server, LM Studio, Ollama, or vLLM).
Point it somewhere else with environment variables, for example:
    MODEL_BASE_URL=http://localhost:1234/v1 MODEL_NAME=darvis-router-1.5b-4bit uv run uvicorn app:app
"""

import json
import os
import re
import time
from typing import Literal

from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://localhost:8080/v1")
# mlx_lm.server maps "default_model" to whatever model it was started with.
MODEL_NAME = os.getenv("MODEL_NAME", "default_model")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "local")
SYSTEM_PROMPT = "You are the Darvis query router. Convert the student's question into a JSON query."
MAX_ATTEMPTS = 2

client = AsyncOpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)
app = FastAPI(title="Darvis router")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Route(BaseModel):
    intent: Literal[
        "grade_distribution", "professor_compare", "professor_recommend",
        "prerequisites", "course_info", "schedule", "unsupported",
    ]
    courses: list[str]
    professors: list[str]
    term: Literal["Fall 2026", "Spring 2027", "Summer 2027"] | None


class RouteResponse(BaseModel):
    route: Route
    attempts: int
    fallback: bool
    latency_ms: float


FALLBACK = Route(intent="unsupported", courses=[], professors=[], term=None)


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@app.get("/health")
async def health():
    try:
        models = await client.models.list()
        return {"status": "ok", "model_server": MODEL_BASE_URL, "models": [m.id for m in models.data][:5]}
    except Exception as exc:  # the model server is down or unreachable
        return {"status": "degraded", "model_server": MODEL_BASE_URL, "error": str(exc)[:200]}


@app.post("/route", response_model=RouteResponse)
async def route(body: Question):
    start = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body.question},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        parsed = extract_json(response.choices[0].message.content or "")
        if parsed is not None:
            try:
                result = Route.model_validate(parsed)
                return RouteResponse(route=result, attempts=attempt, fallback=False,
                                     latency_ms=(time.perf_counter() - start) * 1000)
            except ValidationError:
                pass  # well-formed JSON with the wrong shape counts as a failed attempt
    # Never let a bad model output become a 500 error for the user.
    return RouteResponse(route=FALLBACK, attempts=MAX_ATTEMPTS, fallback=True,
                         latency_ms=(time.perf_counter() - start) * 1000)
