import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import re

# Hyperparameters
MAX_LEN = 40
BATCH_SIZE = 64
NUM_HEADS = 8
D_MODEL = 512
FFN_UNITS = 2048
DROPOUT = 0.1
NUM_LAYERS = 4
EPOCHS = 300
VOCAB_SIZE = 8000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CustomTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word_count = {}

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                self.word_count[word] = self.word_count.get(word, 0) + 1
        sorted_vocab = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 4]
        for idx, (word, _) in enumerate(sorted_vocab, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text.split()]
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            text = " ".join([self.idx2word.get(idx, "<unk>") for idx in seq])
            texts.append(text)
        return texts

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

loaded_tokenizer = load_tokenizer('model/tokenizer.pkl')

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output
  
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(attention)
      
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ffn_units, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_units)
        self.linear2 = nn.Linear(ffn_units, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)
      
      
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, ffn_units, dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output = self.attention(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout(ffn_output))
      
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, ffn_units, dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.attention1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout(attn1))

        attn2 = self.attention2(out1, enc_output, enc_output, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout(attn2))

        ffn_output = self.ffn(out2)
        return self.layernorm3(out2 + self.dropout(ffn_output))
      
      
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)
      
      
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ffn_units, num_layers, dropout_rate, max_len):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_units, dropout_rate)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_units, dropout_rate)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 1

    def forward(self, encoder_input, decoder_input, encoder_mask=None, decoder_mask=None):
        # Encoder
        encoder_embedded = self.embedding(encoder_input)
        encoder_embedded = self.positional_encoding(encoder_embedded)

        encoder_output = encoder_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, encoder_mask)

        # Decoder
        decoder_embedded = self.embedding(decoder_input)
        decoder_embedded = self.positional_encoding(decoder_embedded)

        look_ahead_mask = self.create_look_ahead_mask(decoder_input.size(1)).to(decoder_input.device)

        decoder_output = decoder_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, look_ahead_mask, encoder_mask)

        return self.fc(decoder_output)
      


def chat_response(question, tokenizer, model, max_len=40):
    model.eval()
    question_seq = tokenizer.texts_to_sequences([question])[0]
    question_seq = [1] + question_seq[:max_len - 2] + [2]  # Add <start> and <end> tokens
    question_seq = question_seq + [0] * (max_len - len(question_seq))  # Pad to max_len
    question_tensor = torch.tensor([question_seq]).to(device)

    decoder_input = torch.tensor([[1]]).to(device)  # Start token
    response = []

    for _ in range(max_len):
        with torch.no_grad():
            output = model(question_tensor, decoder_input)
        predicted_id = torch.argmax(output[:, -1, :], dim=-1).item()
        if predicted_id == 2:  # End token
            break
        response.append(predicted_id)
        decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]]).to(device)], dim=-1)

    return tokenizer.sequences_to_texts([response])[0].replace("<start>", "").replace("<end>", "").strip()

# Load the Trained Model
model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, FFN_UNITS, NUM_LAYERS, DROPOUT, MAX_LEN)
model = model.to(device)
model.load_state_dict(torch.load("model/transformer_chatbot_gpu_deco_1.pth"))
model.eval()
print(1)


# Streamlit app interface
# Streamlit app interface
st.title("Chatbot (Livermore) with Transformer")

# Initialize the conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Initialize the user input state if not already initialized
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def submit():
    st.session_state.user_input = st.session_state.widget
    st.session_state.widget = ''

# Display conversation history
for i, message in enumerate(st.session_state.conversation_history):
    if message["role"] == "user":
        st.write(f"You: {message['text']}")
    else:
        st.write(f"Livermore: {message['text']}")

# User input for the next message
st.text_input("You:", key="widget", on_change=submit)

# If the user submits a message
if st.session_state.user_input:
    # Add user's message to history
    st.session_state.conversation_history.append({"role": "user", "text": st.session_state.user_input})
    
    # Get bot's response
    bot_response = chat_response(st.session_state.user_input, loaded_tokenizer, model)
    
    # Add bot's message to history
    st.session_state.conversation_history.append({"role": "bot", "text": bot_response})
    
    # Clear input field by resetting the session state variable
    st.session_state.user_input = ""  # Reset the value of the input field
    
    # Trigger a rerun of the app
    st.rerun()  # This will cause the input field to reset and the new message to appear