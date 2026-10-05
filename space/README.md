---
title: Multimodal Food Recognition
emoji: 🍜
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# Multimodal Food Understanding from Images and Text

Demo of the Deep Learning final project (Group 12, USTH). Upload a food photo and, optionally, a text such as a
recipe or a description. The app shows the top-5 dishes (101 UPMC Food-101 classes) predicted from the image only,
from the text only, and by the cross-attention fusion of both. All three heads run on frozen CLIP ViT-B/16 features.
The text is cleaned and every dish name is masked before it reaches the models, as in training.

This Space runs on CPU, so the vision-language extraction (Qwen3-VL) of the full project is disabled here; see the
repository for the complete pipeline: <https://github.com/viethoang2503/food-recognize-extract>.
