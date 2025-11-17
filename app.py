from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

# Load detoxification model and tokenizer
DETOX_MODEL_PATH = "t5-small-detox-finetuned"
detox_tokenizer = AutoTokenizer.from_pretrained(DETOX_MODEL_PATH)
detox_model = AutoModelForSeq2SeqLM.from_pretrained(DETOX_MODEL_PATH)

# Load classification model and tokenizer
CLASSIFY_MODEL_PATH = "classifyModel"
classify_tokenizer = AutoTokenizer.from_pretrained(CLASSIFY_MODEL_PATH)
classify_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFY_MODEL_PATH)
classify_model.eval()

# Homepage route
@app.route("/", methods=["GET"])
def home():
    return """
    <html>
        <head><title>Detoxification & Classification API</title></head>
        <body style="font-family: Arial; text-align: center; padding: 40px;">
            <h1>Detoxification & Classification API is Live</h1>
            <p>Use the endpoints below by sending <code>POST</code> requests with JSON input.</p>
            <div style="margin: 20px auto; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #fff; max-width: 600px;">
                <h2>/detoxify</h2>
                <p>Send text to be detoxified:</p>
                <pre>{ "text": "your input here" }</pre>
                <p>Response:</p>
                <pre>{ "detoxified": "cleaned text" }</pre>
            </div>
            <div style="margin: 20px auto; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #fff; max-width: 600px;">
                <h2>/classify</h2>
                <p>Send text to be classified as toxic or non-toxic:</p>
                <pre>{ "text": "your input here", "threshold": 0.5 }</pre>
                <p>Response:</p>
                <pre>{
  "classification": "toxic",
  "confidence": {
    "non-toxic": 0.03,
    "toxic": 0.97
  }
}</pre>
            </div>
            <p style="margin-top:40px; font-size:0.9em; color:#888;">
                Flask API running on <code>0.0.0.0:7860</code>
            </p>
        </body>
    </html>
    """

# Detoxify endpoint
@app.route("/detoxify", methods=["POST"])
def detoxify():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        inputs = detox_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = detox_model.generate(**inputs, max_new_tokens=128)
        detoxified = detox_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"detoxified": detoxified})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Classification endpoint with confidence and threshold
@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        threshold = float(data.get("threshold", 0.5))

        if not text:
            return jsonify({"error": "No text provided"}), 400

        inputs = classify_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = classify_model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze()

        # Confidence dictionary
        confidence = {
            "non-toxic": round(probs[0].item(), 4), "toxic": round(probs[1].item(), 4)}

        # Threshold-based classification
        prediction_label = "toxic" if probs[1].item() > threshold else "non-toxic"

        return jsonify({
            "classification": prediction_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)