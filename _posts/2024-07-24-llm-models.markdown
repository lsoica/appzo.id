---
layout: post
title:  "LLM Models"
date:   2024-07-24 11:08:03 +0200
categories: ai
---

[Source Video](https://www.youtube.com/watch?v=bZQun8Y4L2A)

# Stages

Pre-training -> Base model
Supervised training -> SFT model (assistants, for example the ChatGPT is an assistent on top of the GPT4 base model)
Reward modeling  -> RM model (humans choosing from multiple completions for a prompt, ranking them and then the model is retrained to match the human's choice)
Reinforcement learning -> RL model (based on the rewards given for each completion, lower the tokens part of a non rewarded completion and increase the rewarded ones)