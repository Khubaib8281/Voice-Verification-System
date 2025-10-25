# 🔊 ECAPA Voice Verification System

This repository contains a **voice verification system** built using **ECAPA-TDNN embeddings** and a **Logistic Regression classifier**.  
The model verifies whether an input voice sample belongs to a specific enrolled user or not — achieving **100% accuracy** on the test set.

---

## 🧠 Overview

The system extracts **speaker embeddings** using the pretrained **ECAPA-TDNN** model from [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).  
Then, it trains a **Logistic Regression classifier** on these embeddings to distinguish between:
- ✅ **Authorized voice (target speaker)**
- ❌ **Other/random voices**

---

## ⚙️ Workflow

1. Extract embeddings from audio using ECAPA-TDNN  
2. Train a Logistic Regression model on the embeddings  
3. Save the trained model as `voice_auth.pkl`  
4. Use `voice_auth.py` for real-time verification  

---

## 📂 Repository Structure

```
voice_verification/
│
├── voice_auth.py                # Voice verification module
├── voice_verification_system.ipynb     # Training and testing notebook
├── voice_verification_model.pkl # Trained classifier
├── requirements.txt             # Minimal dependencies
```

---

## 🧩 Requirements

```bash
torch
speechbrain
numpy
scikit-learn
joblib
```

> Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
import joblib
from speechbrain.pretrained import EncoderClassifier
import torch
import numpy as np
import warnings

# Load models
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
clf = joblib.load("voice_auth.pkl")

def verify_voice(audio_path, clf_model):
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    signal = ecapa.load_audio(audio_path)
    embd = ecapa.encode_batch(signal).detach().cpu().numpy().mean(axis=1)
    prediction = clf_model.predict(embd)
    proba = clf_model.predict_proba(embd)
    print(f"Prediction: {prediction[0]}")
    print(f"Class probabilities: {proba[0]}")

# Example
verify_voice("test_voice.wav", clf)
```

---

## 🎯 Example Output

```
Prediction: Authorized
Class probabilities: [0.02, 0.98]
```

---

## 🧪 Model Performance

- Accuracy: **100%**
- Classifier: Logistic Regression
- Embeddings: ECAPA-TDNN (SpeechBrain)
- Verification Type: Binary (Authorized vs Unauthorized)

---

## 🌍 Applications

- Voice-based login systems  
- Access control verification  
- Smart assistants personalization  
- Speaker authentication for APIs

---

## 📚 Citation / Credit

If you use this work in your research or projects, please cite:

`Muhammad Khubaib Ahmad (2025). Voice Verification System using ECAPA Embeddings`.\
Model and methodology inspired by SpeechBrain ECAPA-TDNN:
> SpeechBrain ECAPA-TDNN – https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

> Star the repository if you found it useful
---

## 🧑‍💻 Author

**Muhammad Khubaib Ahmad**  
AI/ML Engineer | Data Scientist | AI Researcher  
[Hugging Face Profile](https://huggingface.co/Khubaib01)  
[GitHub Profile](https://github.com/Khubaib8281)

---
