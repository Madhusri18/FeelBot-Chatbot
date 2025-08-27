from flask import Flask, render_template, request, jsonify
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask
app = Flask(__name__)

# Load model and tokenizer
model = load_model("saved/chatbot_dl.h5")
with open("saved/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("saved/label_encoder.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)

max_len = 100  # adjust based on your preprocessing

# ---------- Helper Functions ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def preprocess_input(text, tokenizer, max_len=100):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

def detect_negation(user_input):
    negations = ["not", "isnt", "isn't", "didnt", "didn't", "never", "cannot", "cant", "can't"]
    negation_map = {
        "happy": "sadness",
        "sadness": "happy",
        "anger": "neutral",
        "fear": "neutral",
        "surprise": "neutral",
        "disgust": "neutral"
    }

    text = clean_text(user_input)
    words = text.split()
    neg_pos = [i for i, w in enumerate(words) if w in negations]

    emo_pos = []
    for i, w in enumerate(words):
        for emo in negation_map:
            if emo in w:
                emo_pos.append((i, emo))

    for n in neg_pos:
        for e_idx, emo in emo_pos:
            if e_idx > n:
                return negation_map[emo]

    return None

# Greetings & farewells
greetings = ["hi", "hello", "hey", "hiya"]
farewells = ["bye", "okay", "done", "thank you", "thanks"]

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message").strip().lower()
    words = user_input.split()

    # Greetings
    if any(word in greetings for word in words):
        return jsonify({"response": "Hello! How are you feeling today?"})

    # Farewells
    if any(word in farewells for word in words):
        return jsonify({"response": "Goodbye! Take care. 👋"})

    # Negation detection
    label = detect_negation(user_input)

    if label is None:
        x = preprocess_input(user_input, tokenizer, max_len)
        prediction = model.predict(x)[0]
        max_prob = np.max(prediction)
        label = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]

        if max_prob < 0.5:
            return jsonify({"response": "Hmm, I’m not sure how you feel 🤔"})

    return jsonify({"response": f"I sense you are feeling {label}"})


if __name__ == "__main__":
    app.run(debug=True)
