import os
import warnings  
import re      
warnings.filterwarnings("ignore", message=".*list_audio_backends has been deprecated.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")   
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torchaudio._backend.list_audio_backends")     
    
warnings.filterwarnings(   
    "ignore",
    category=UserWarning,
    message=re.escape("Trying to unpickle estimator LogisticRegression from version")
)

warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
       
import torch
import torchaudio
import numpy as np
import sounddevice as sd
from joblib import load
from speechbrain.inference import EncoderClassifier

# Load model and ECAPA encoder

print("🔐 Initializing Voice Unlock System...")   
clf = load("voice_auth.pkl")
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Record voice from microphone      

def record_voice(seconds=3, fs=16000):
    print(f"\n🎤 Please speak for {seconds} seconds...")       
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()    
    print("✅ Recording complete.\n")
    return torch.tensor(audio.T)    


# Extract ECAPA embedding

def extract_embedding(signal, fs=16000):     
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
    emb = encoder.encode_batch(signal).squeeze().detach().cpu().numpy()
    return emb

# Authenticate and unlock folder
def authenticate_and_unlock(folder_path):
    signal = record_voice(seconds=4)
    emb = extract_embedding(signal)

    pred = clf.predict([emb])[0]
    proba = clf.predict_proba([emb])[0]
    conf = proba.max()

    print(f"🔎 Predicted: {pred} ({conf:.2f} confidence)")

    if pred == "me" and conf > 0.75:
        print("✅ Voice verified! Unlocking folder...")
        os.startfile(folder_path)  # open the folder (Windows)
    else:
        print("❌ Access denied.")

# Run authentication
if __name__ == "__main__":
    folder_to_unlock = r"D:\Media\Music\ma stuff"       
    authenticate_and_unlock(folder_to_unlock)
        