# train.py

import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from models.utils import save_pickle

# Load dataset
data = pd.read_csv("data/emotion_final.csv")

print("\n📊 Dataset Info:")
print("Shape:", data.shape)       # (rows, columns)
print("\nFirst 5 rows:\n", data.head())
print("\nLabel distribution:\n", data['Emotion'].value_counts())


# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

data["Text"] = data["Text"].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["Emotion"])
labels_onehot = to_categorical(labels)

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data["Text"])
sequences = tokenizer.texts_to_sequences(data["Text"])

max_len = 100
X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Train/test split (important for evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_onehot, test_size=0.2, random_state=42, stratify=labels
)

# Save preprocessing tools
save_pickle(tokenizer, "saved/tokenizer.pkl")
save_pickle(label_encoder, "saved/label_encoder.pkl")

# Also save test data for evaluation.py
np.save("saved/X_test.npy", X_test)
np.save("saved/y_test.npy", y_test)

# Build deep learning model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_len))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save model
model.save("saved/chatbot_dl.h5")
print("✅ Deep Learning model trained and saved successfully!")
