import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def train_and_predict(description, input_text, num_words):
    # Tokenize the training text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([description])
    total_words = len(tokenizer.word_index) + 1

    # Create training sequences
    token_list = tokenizer.texts_to_sequences([description])[0]
    sequences = []
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(x) for x in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

    X = sequences[:, :-1]
    y = sequences[:, -1]

    # Build model
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len - 1),
        LSTM(128, return_sequences=False),
        Dense(total_words, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)

    # Reverse lookup for words
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

    # Predict multiple words
    text = input_text.strip()
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_word = reverse_word_index.get(predicted_index, "")

        # Skip empty or consecutive duplicate words
        if not predicted_word:
            break
        if text.split() and predicted_word == text.split()[-1]:
            # Try next likely word if the same word was predicted
            predicted_probs[0][predicted_index] = 0
            predicted_index = np.argmax(predicted_probs)
            predicted_word = reverse_word_index.get(predicted_index, "")
            if not predicted_word:
                break

        text += " " + predicted_word

    return text


# --- 🎨 Gradio Interface ---
custom_css = """
body {
    background-color: #0b0c10;
    color: #c5c6c7;
}
h1, h2, h3 {
    color: #66fcf1 !important;
}
.gradio-container {
    background: linear-gradient(145deg, #0b0c10 0%, #1f2833 100%);
    border-radius: 12px;
    padding: 25px;
}
.gr-button-primary {
    background-color: #45a29e !important;
    border-color: #45a29e !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}
.gr-button-secondary {
    background-color: #c5c6c7 !important;
    color: #1f2833 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}
textarea {
    background-color: #1f2833 !important;
    color: #c5c6c7 !important;
    border-radius: 8px !important;
}
"""

description = gr.Textbox(
    label="📝 Training Text (Description)",
    placeholder="Enter text to train your model, e.g. 'the quick brown fox jumps over the lazy dog'",
    lines=3
)
input_text = gr.Textbox(
    label="💬 Input Text (Prompt)",
    placeholder="Enter starting words, e.g. 'quick'",
    lines=1
)
num_words = gr.Slider(
    label="🔢 Number of Words to Predict",
    minimum=1,
    maximum=20,
    value=5,
    step=1
)
output = gr.Textbox(
    label="✨ Predicted Output",
    placeholder="Your generated text will appear here...",
    lines=3
)

demo = gr.Interface(
    fn=train_and_predict,
    inputs=[description, input_text, num_words],
    outputs=output,
    title="🧩 Text Word Predictor",
    description="Train a mini LSTM model on your text and generate the next few words (clean predictions, no duplicates).",
    theme="gradio/soft",
    css=custom_css
)

if __name__ == "__main__":
    demo.launch()
