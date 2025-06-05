# Artificial Intelligence Course Project
This repository contains a web application for sentiment analysis of Indonesian text using two pre-trained models from Hugging Face. The application allows users to input text and receive sentiment predictions from both models, enabling easy comparison and analysis.

### Models Used
- **[`crypter70/IndoBERT-Sentiment-Analysis`](https://huggingface.co/crypter70/IndoBERT-Sentiment-Analysis)**  
  Fine-tuned from `indobenchmark/indobert-base-p1` on the IndoNLU dataset, achieving 94.52% accuracy.
- **[`agufsamudra/indo-sentiment-analysis`](https://huggingface.co/agufsamudra/indo-sentiment-analysis)**  
  Binary sentiment classifier (Positive/Negative) fine-tuned from `indobenchmark/indobert-base-p1`.
