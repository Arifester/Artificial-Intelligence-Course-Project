# app.py
from flask import Flask, request, render_template
from models_loader import (
    tokenizer_crypter, model_crypter, labels_crypter_display,
    tokenizer_aguf, model_aguf, labels_aguf_display
)
import torch

app = Flask(__name__)

def get_sentiment_prediction(tokenizer, model, text_input, label_map):
    """
    Fungsi untuk mendapatkan prediksi sentimen dari model.
    """
    if tokenizer is None or model is None:
        return "Model tidak tersedia", {}

    try:
        # Tokenisasi
        inputs = tokenizer(text_input, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Prediksi
        with torch.no_grad(): # Penting untuk inferensi
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist() # squeeze() untuk batch size 1
        predicted_index = torch.argmax(logits, dim=-1).item()

        predicted_label_name = label_map.get(predicted_index, f"LABEL_{predicted_index}")

        # Membuat dictionary probabilitas dengan nama label yang benar
        prob_dict = {}
        for i, prob_value in enumerate(probabilities):
            label_name = label_map.get(i, f"LABEL_{i}")
            prob_dict[label_name] = round(prob_value * 100, 2) # Dalam persentase

        return predicted_label_name, prob_dict
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return "Error prediksi", {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_to_analyze = request.form.get('text_input', '')
        results = {'text_input': text_to_analyze}

        if text_to_analyze:
            # Prediksi dengan Model Crypter70
            if model_crypter and tokenizer_crypter:
                pred_label_crypter, probs_crypter = get_sentiment_prediction(
                    tokenizer_crypter, model_crypter, text_to_analyze, labels_crypter_display
                )
                results['prediction_crypter'] = pred_label_crypter
                results['probabilities_crypter'] = probs_crypter
            else:
                results['prediction_crypter'] = "Model Crypter70 tidak tersedia"
                results['probabilities_crypter'] = {}

            # Prediksi dengan Model Agufsamudra
            if model_aguf and tokenizer_aguf:
                pred_label_aguf, probs_aguf = get_sentiment_prediction(
                    tokenizer_aguf, model_aguf, text_to_analyze, labels_aguf_display
                )
                results['prediction_aguf'] = pred_label_aguf
                results['probabilities_aguf'] = probs_aguf
            else:
                results['prediction_aguf'] = "Model Agufsamudra tidak tersedia"
                results['probabilities_aguf'] = {}

            return render_template('index.html', **results)

    return render_template('index.html', text_input='', prediction_crypter=None, probabilities_crypter=None, prediction_aguf=None, probabilities_aguf=None)

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False untuk production
