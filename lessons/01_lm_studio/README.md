# Module 01: LM Studio and local models

By the end of this module you'll understand what a "local model" physically is, why quantization exists, how to pick a model that fits your Mac, and how to call a local model from Python the same way you'd call OpenAI. Do parts 1 to 3 in the LM Studio app, then part 4 in `lm_studio_api.ipynb`.

## Part 1: What you're actually downloading

A model is two things: a file of weights (billions of numbers learned during training) and a runtime that does the math to turn your prompt into output. LM Studio is the runtime plus a GUI. The model files come from Hugging Face.

Three properties decide whether a model is useful to you.

Parameter count (the "7B" or "8B" in the name) is how many weights it has. More parameters generally means more capable, and also more memory and slower output.

Base vs. instruct. A base model was only trained to predict the next token on internet text. Ask it a question and it may just continue your text with more questions. An instruct (or "chat", "it") model was further trained to follow instructions and hold a conversation. For chatting, always pick instruct. Base models matter later when you fine-tune.

Format. You'll see two kinds in LM Studio. GGUF runs on the llama.cpp engine and works on any computer. MLX is Apple's format and is usually faster on Apple Silicon. On your Mac, prefer MLX when both exist.

## Part 2: Quantization, and why it decides what fits

Each weight is normally stored as a 16-bit number. Quantization stores each one in fewer bits, which shrinks the model and speeds it up at a small cost in quality.

Rough sizes for a 7B to 8B model:

| Precision | Name you'll see | Approximate size | Quality |
|---|---|---|---|
| 16-bit | F16, BF16 | ~15 GB | Reference |
| 8-bit | Q8_0, 8bit | ~8 GB | Almost identical |
| ~5-bit | Q5_K_M | ~5.5 GB | Very close |
| ~4-bit | Q4_K_M, 4bit | ~4.5 GB | Small drop, usually the best trade |
| 3-bit and below | Q3, Q2 | ~3 GB or less | Noticeably worse |

The rule of thumb: bytes ≈ parameters × bits ÷ 8, plus some overhead.

The memory rule for your Mac: the model file plus a few GB for the context (the KV cache, which grows with conversation length) has to fit in the memory macOS lets the GPU use. On Apple Silicon that's roughly two-thirds to three-quarters of your total RAM. Check your RAM under Apple menu > About This Mac.

| Your RAM | Comfortable model size |
|---|---|
| 8 GB | 1B to 3B at 4-bit |
| 16 GB | up to 8B at 4-bit |
| 24 to 32 GB | up to 14B at 4-bit, or 8B at 8-bit |
| 48 GB+ | 30B-class at 4-bit |

LM Studio shows a fit estimate next to each download, so you don't have to calculate it every time. Now you know where that estimate comes from.

## Part 3: Hands-on in the app

1. Download a first model. Open the search (Discover) view and look for a small instruct model in MLX 4-bit, in the 3B to 8B range that fits your RAM. Qwen and Llama instruct models are good starting points. Before you download, read the model card: who made it, its context length, and its license.

2. Chat with it. Load the model and open the chat view. Find the settings panel and locate the system prompt, temperature, and context length.

3. Watch the numbers. After each reply LM Studio shows tokens per second and time to first token. Write down your numbers. Those two metrics are what you'll care about when you serve a model to real users.

4. Temperature experiment. Ask "Write one sentence describing a basketball game." five times at temperature 0, then five times at temperature 1.5. At 0 the model always picks the most likely next token, so the outputs come out the same or nearly the same. At higher temperature it samples from more of the probability distribution, so you get more variety and more nonsense.

5. System prompt experiment. Set the system prompt to "You are a terse senior engineer. Answer in at most two sentences." and ask a question. Then clear it and ask again. The system prompt is the cheapest way to change a model's behavior. Fine-tuning (module 07) is the expensive way.

6. Quantization experiment. Download the same model at a different quantization (for example 8-bit if you used 4-bit). Ask both the same hard question, like a tricky logic puzzle or a piece of code to debug. Compare speed, memory use (watch Activity Monitor), and answer quality.

7. Find the limits. Paste in something long (a full lecture transcript or a long article) and ask for a summary. Watch speed drop as context grows. That's the KV cache at work.

## Part 4: The local server

This is the most important part for building. LM Studio can serve the loaded model through an API that matches OpenAI's format exactly, so any code written for OpenAI works against your local model by changing one URL.

Start it in the app: open the Developer view (the terminal-style icon in the left sidebar) and switch the server on. It runs at `http://localhost:1234`.

Or start it from the terminal with the `lms` CLI that ships with LM Studio. If `lms` isn't found, run `~/.lmstudio/bin/lms bootstrap` and open a new terminal tab.

```bash
lms ls                 # models you've downloaded
lms load               # pick a model to load (interactive)
lms ps                 # what's loaded right now
lms server start       # start the API server
lms server status
lms unload --all       # free the memory when you're done
```

Then open `lm_studio_api.ipynb` in this folder. Make sure you ran `uv add openai matplotlib` in module 00 first.

## Why this matters for Darvis

Darvis pays per request for model inference today. Once you can serve a model behind an OpenAI-compatible API, swapping the hosted model for your own fine-tuned one is a config change in your backend, not a rewrite. Module 08 does exactly that.
