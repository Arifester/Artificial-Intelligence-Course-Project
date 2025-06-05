# models_loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("Memuat model dan tokenizer, ini mungkin memakan waktu...")

# --- Model 1: crypter70/IndoBERT-Sentiment-Analysis ---
try:
    tokenizer_crypter = AutoTokenizer.from_pretrained("crypter70/IndoBERT-Sentiment-Analysis")
    model_crypter = AutoModelForSequenceClassification.from_pretrained("crypter70/IndoBERT-Sentiment-Analysis")
    model_crypter.eval() # Set ke mode evaluasi

    # Pemetaan label berdasarkan config.json yang Anda berikan
    # "id2label": {"0": "POSITIVE", "1": "NEUTRAL", "2": "NEGATIVE"}
    labels_crypter_display = model_crypter.config.id2label
    # Jika ingin memastikan urutan atau nama yang lebih custom:
    # labels_crypter_display = {
    #     0: "Positif (Crypter)",
    #     1: "Netral (Crypter)",
    #     2: "Negatif (Crypter)"
    # }
    print("Model crypter70 berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model crypter70: {e}")
    tokenizer_crypter, model_crypter, labels_crypter_display = None, None, {}


# --- Model 2: agufsamudra/indo-sentiment-analysis ---
try:
    tokenizer_aguf = AutoTokenizer.from_pretrained("agufsamudra/indo-sentiment-analysis")
    model_aguf = AutoModelForSequenceClassification.from_pretrained("agufsamudra/indo-sentiment-analysis")
    model_aguf.eval() # Set ke mode evaluasi

    # Pemetaan label berdasarkan hasil pengujian Anda:
    # Indeks 1 adalah "Positif"
    # Indeks 0 adalah "Negatif"
    # id2label dari config: {0: 'LABEL_0', 1: 'LABEL_1'}
    labels_aguf_display = {
        0: "Negatif",  # Sesuai hasil tes: 'Produk ini sangat buruk...' -> Indeks 0
        1: "Positif"   # Sesuai hasil tes: 'Saya sangat senang...' -> Indeks 1
    }
    print("Model agufsamudra berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model agufsamudra: {e}")
    tokenizer_aguf, model_aguf, labels_aguf_display = None, None, {}

print("Semua model yang tersedia telah selesai dimuat.")
