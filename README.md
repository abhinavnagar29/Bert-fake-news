# NewsGauge — Fake News Classification with BERT

A deployed fake-news headline classification model built on the [Fakeddit](https://github.com/entitize/Fakeddit) dataset. This repository contains the **Streamlit web application** for the production model.

**🚀 Try the live demo**: Deploy the Streamlit app from this repository to Streamlit Community Cloud.

## Overview

NewsGauge uses a fine-tuned DistilBERT model to classify news headlines as "real" or "fake". The model was trained on a balanced subset of the Fakeddit dataset (5,000 posts: 2,500 real, 2,500 fake) and achieves approximately 80% accuracy on held-out test data.

The model features:
- **Temperature-scaled calibration** for more reliable confidence scores
- **Hugging Face Hub integration** for easy model loading
- **Streamlit web interface** for interactive predictions

## Model Performance

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| DistilBERT (fine-tuned) | 0.798 | 0.798 |

The model was trained with:
- **Dataset**: Balanced Fakeddit subset (5,000 posts)
- **Split**: 3,500 train / 500 validation / 1,000 test
- **Calibration**: Temperature scaling (T = 1.110)

## Deployment

### Streamlit Community Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Configure:
   - Repository: `abhinavnagar29/NewsGauge`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - Requirements file path: `app/requirements.txt`
5. Click "Deploy"

### Local Testing

```bash
git clone https://github.com/abhinavnagar29/NewsGauge.git
cd NewsGauge
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Repository Structure

```
NewsGauge/
├── app/
│   ├── streamlit_app.py     # Streamlit web application
│   └── requirements.txt     # Deployment dependencies
└── README.md                # This file
```

## Model Details

- **Model**: DistilBERT (distilbert-base-uncased)
- **Hugging Face**: https://huggingface.co/abhinav-29/fakeddit-bert-fake-news
- **Task**: Binary classification (real vs fake news headlines)
- **Input**: News headlines (max 64 tokens)
- **Output**: Label + calibrated confidence score

## Limitations

- **Not a general-purpose fact-checker**: The model learns headline-style correlates from Fakeddit's distant-supervision labels, not verified claim truth
- **Training data specific**: Trained on Reddit post titles from Fakeddit dataset
- **Confidence calibration**: Calibration reduces average overconfidence but doesn't guarantee individual prediction correctness
- **English only**: Model trained on English text only

## License

This project builds on coursework for CS60075 at IIT Kharagpur. Use as a reference for learning and portfolio development.
