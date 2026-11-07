# Module 03: micrograd

Open `micrograd.ipynb`. You'll build a tiny autograd engine in plain Python, check it against PyTorch, and use it to train a small neural network.

Watch Andrej Karpathy's video "The spelled-out intro to neural networks and backpropagation: building micrograd" (YouTube, about 2.5 hours) alongside it. A good rhythm: watch a section, pause, write that part of the code in a blank cell yourself, then compare with the notebook.

This is the most important module in the first half of the course. Once backprop stops being magic, every later topic (transformers, fine-tuning, LoRA) is engineering on top of something you already understand.

You're done when you can rebuild the `Value` class with `+`, `*`, `tanh`, and `backward()` in a blank file from memory.
