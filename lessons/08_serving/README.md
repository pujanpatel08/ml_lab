# Module 08: Serving your model

New dependencies (run from `ML_Lab`):

```bash
uv add fastapi uvicorn
```

Prerequisite: module 07, which produced `lessons/07_mlx_lora/models/darvis-router-1.5b-4bit`.

A model on disk isn't a product. This module puts your fine-tuned router behind an API, wraps it in a backend service the way Darvis would use it, and measures what it costs to run.

## The stack you're building

```
student question
   -> FastAPI service (app.py): validation, retries, fallback, timing
   -> OpenAI-compatible model server: mlx_lm.server on your Mac (or LM Studio, Ollama, vLLM)
   -> your fine-tuned model
```

Because every model server here speaks the same OpenAI-style API (module 01), the backend doesn't care which one is underneath. You can develop against your Mac and deploy against a rented GPU by changing one environment variable.

## Step 1: start the model server

In an iTerm2 tab, from `ML_Lab`:

```bash
uv run mlx_lm.server --model lessons/07_mlx_lora/models/darvis-router-1.5b-4bit --port 8080
```

Leave it running. It's an OpenAI-compatible server built into `mlx-lm`, the quickest way to serve an MLX model.

## Step 2: work through `serving.ipynb`

It calls the server directly, measures latency and throughput, then calls the FastAPI service.

## Step 3: start the FastAPI service when the notebook tells you to

In a second iTerm2 tab:

```bash
cd lessons/08_serving
uv run uvicorn app:app --port 8000 --reload
```

Then open http://localhost:8000/docs. FastAPI generates interactive API docs from the code, so you can try the `/route` endpoint in the browser.

## Other ways to serve the same model

LM Studio. LM Studio can load MLX models from its models folder, which expects a `publisher/model-name` layout. Find the folder in LM Studio's My Models view, then copy your model in, for example to `~/.lmstudio/models/pujanpatel8/darvis-router-1.5b-4bit`. Run `lms ls` to confirm LM Studio sees it, load it, and start the server on port 1234. Then start the FastAPI app with `MODEL_BASE_URL=http://localhost:1234/v1 MODEL_NAME=<the id lms ls shows>`.

Ollama (optional, more steps). Ollama runs GGUF files, the llama.cpp format, so you convert the 16-bit fused model first. Commands below reflect llama.cpp's current tooling. If a script name has changed, check the llama.cpp README.

```bash
brew install llama.cpp                                    # gives you llama-quantize
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/Developer/llama.cpp
cd ~/Developer/ML_Lab/lessons/07_mlx_lora
uv run --with-requirements ~/Developer/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  python ~/Developer/llama.cpp/convert_hf_to_gguf.py models/darvis-router-1.5b \
  --outfile models/darvis-router-f16.gguf --outtype f16
llama-quantize models/darvis-router-f16.gguf models/darvis-router-q4_k_m.gguf Q4_K_M
```

Then make a file called `Modelfile` next to the GGUF:

```
FROM ./darvis-router-q4_k_m.gguf
PARAMETER temperature 0
SYSTEM You are the Darvis query router. Convert the student's question into a JSON query.
```

and run `ollama create darvis-router -f Modelfile` and `ollama run darvis-router "prereqs for systems?"`. Ollama serves it at `http://localhost:11434/v1` with `MODEL_NAME=darvis-router`.

vLLM on a rented GPU (production). vLLM is the standard open-source server for production traffic. It batches many requests together on the GPU (continuous batching) and manages KV-cache memory efficiently (PagedAttention), so one GPU can serve many users at once. On a rented Linux GPU box (RunPod, Lambda, and others), copy up the 16-bit fused model and run:

```bash
uv venv && uv pip install vllm
uv run vllm serve ./darvis-router-1.5b --served-model-name darvis-router --port 8000
```

Point the FastAPI service at it with `MODEL_BASE_URL=http://<box address>:8000/v1 MODEL_NAME=darvis-router`. Put it behind authentication before exposing it to the internet.

## Why this matters

This module is the full loop for Darvis: train a specialized model, serve it, put a reliable backend in front of it, and know the cost per request. The backend patterns here (schema validation, retries, a safe fallback, timing every request) are what make an unreliable component like an LLM safe to put in front of users. The cost math is what tells you whether self-hosting is worth it at your traffic level, and that answer changes as you grow.

You're done when you can answer, with your own measured numbers: what's the p95 latency of the router, how many requests per second can your Mac handle, and at what monthly request volume a rented GPU beats a hosted API.
