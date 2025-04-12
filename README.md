
# Sentiment Analysis on IMDb Dataset Using BERT

This project demonstrates how to fine-tune a pre-trained BERT model for binary sentiment classification (positive or negative) on the IMDb movie review dataset. The goal is to classify movie reviews based on their sentiment, leveraging BERT's powerful language understanding capabilities. This project was developed as part of my journey to learn how to apply BERT to natural language processing (NLP) tasks.

---

## Dataset

The [IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb) consists of **50,000 movie reviews**, evenly split into:
- **25,000 training samples**
- **25,000 testing samples**

Each review is labeled as either **positive (1)** or **negative (0)**, making it a balanced dataset ideal for sentiment analysis.

---

## Model

The project uses the **`bert-base-uncased`** model from [Hugging Face's Transformers library](https://huggingface.co/transformers/). BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on a large corpus of text. By fine-tuning it on the IMDb dataset, we adapt BERT to the specific task of sentiment classification, taking advantage of its bidirectional context understanding.

---

## Results

After fine-tuning, the model achieves competitive performance on the test set. Example metrics include:
- **Accuracy**: ~88%
- **F1 Score**: ~88%

These results were obtained after training for a few epochs on the IMDb dataset. Performance can vary depending on hyperparameter settings and training duration.

---

## Learning Outcomes

This project was a key part of my learning journey with BERT. Here’s what I gained:

- **BERT Fundamentals**: Understood how BERT’s bidirectional architecture enhances NLP tasks.
- **Fine-Tuning Skills**: Learned to fine-tune a pre-trained BERT model using Hugging Face’s `transformers` library.
- **Text Preprocessing**: Mastered tokenization and data preparation for BERT-specific inputs.
- **Model Evaluation**: Explored metrics like accuracy and F1 score to assess performance.

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [IMDb Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

---
