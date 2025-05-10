# LLM Distribution Shifts
--- 

This work aims to research distribution shifts in LLMs, and how LLMs behave when encountering data that differs from their training distribution, distinguishing between "out-of-support" shifts (requiring extrapolation to new concepts) and "in-support" shifts (involving rewordings or spurious correlations). 

Our initial methodology involves fine-tuning pre-trained LLMs on the GSM8K math dataset to specialize them in solving these problems. We then rigorously evaluate their performance on other math datasets designed to represent realistic shifts from the training data. This includes testing on problems with subtle variations in wording (like SVAMP, representing "in-support" shifts) and problems introducing new concepts or requiring deeper reasoning (like MATH, representing "out-of-support" shifts).

This repo contains my initial code for my notebooks, logs, and scripts for running experiments. Within the logs contains logs from fine-tuning runs like Qwen3-0.6B and TinyLlama-1.1B-Chat-v1.0 fine-tuned on GSM8K. It also contains comparison evals on datasets before and after fine-tuning.  

