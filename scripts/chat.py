import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

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
    """
    Detects negation in a sentence and flips the emotion.
    Handles negations anywhere in the sentence.
    """
    # Normalize text
    text = clean_text(user_input)
    
    # Negation words
    negations = ["not", "isnt", "isn't", "didnt", "didn't", "never", "cannot", "cant", "can't"]
    
    # Map emotions to opposite or neutral
    negation_map = {
        "happy": "sadness",
        "sadness": "happy",
        "anger": "neutral",
        "fear": "neutral",
        "surprise": "neutral",
        "disgust": "neutral"
    }
    
    words = text.split()
    
    # Find positions of negation words
    neg_pos = [i for i, w in enumerate(words) if w in negations]
    
    # Find positions of emotion words
    emo_pos = []
    for i, w in enumerate(words):
        for emo in negation_map:
            if emo in w:
                emo_pos.append((i, emo))
    
    # Check if any emotion comes after a negation
    for n in neg_pos:
        for e_idx, emo in emo_pos:
            if e_idx > n:  # emotion appears after negation
                return negation_map[emo]
    
    return None

# ---------- Load Model & Preprocessors ----------
model = load_model("saved/chatbot_dl.h5")

with open("saved/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("saved/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("🤖 Emotion Chatbot is ready! Type 'quit' to exit.\n")

# ---------- Greeting/Farewell Lists ----------
greetings = ["hi", "hello", "hey", "hiya"]
farewells = ["bye", "okay", "done", "thank you", "thanks"]

# ---------- Chat Loop ----------
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        print("Bot: Goodbye! Take care. 👋")
        break

    user_input_lower = user_input.lower()
    words = user_input_lower.split()

    # Handle greetings
    if any(word in greetings for word in words):
        print("Bot: Hello! How are you feeling today?\n")
        continue

    # Handle farewells
    if any(word in farewells for word in words):
        print("Bot: Goodbye! Take care. 👋\n")
        break

    # ---------- Check for negation ----------
    label = detect_negation(user_input)

    if label is None:
        # Use model prediction if no negation detected
        x = preprocess_input(user_input, tokenizer, max_len=100)
        prediction = model.predict(x)[0]
        max_prob = np.max(prediction)
        label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        # Neutral/fallback if model is not confident
        if max_prob < 0.5:
            print("Bot: Hmm, I’m not sure how you feel 🤔\n")
            continue

    print(f"Bot: I sense you are feeling **{label}**.\n")
