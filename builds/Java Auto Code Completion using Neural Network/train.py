import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
import logging
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data files (if not already downloaded)
nltk.download('punkt')

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Example tokenization function
def tokenize(text):
    return word_tokenize(text.lower())

# Load your dataset
with open('mix.json', 'r') as f:
    data = json.load(f)

# Prepare your data
texts = []
labels = []

for intent in data['intents']:
    label = intent['tag']
    patterns = intent['patterns']
    texts.extend(patterns)
    labels.extend([label] * len(patterns))

# Tokenize and stem the words
tokenized_texts = [tokenize(text) for text in texts]

# Train Word2Vec model
word2vec_model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec_model.bin")

# Load the trained model
word2vec_model = Word2Vec.load("word2vec_model.bin")

# Convert text to Word2Vec embeddings
X = []
for text in tokenized_texts:
    embeddings = []
    for word in text:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))
    X.append(np.array(embeddings))

unique_labels = list(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = [label_to_index[label] for label in labels]

input_size = word2vec_model.vector_size
hidden_size = 512
output_size = len(unique_labels)

class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_batch(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) > 0:
            padded_seq = torch.zeros(max_len, input_size)
            padded_seq[:len(seq)] = torch.tensor(seq)
        else:
            padded_seq = torch.zeros(max_len, input_size)
        padded_sequences.append(padded_seq)
    padded_sequences = torch.stack(padded_sequences, dim=0)
    numeric_labels = torch.tensor(labels)
    return padded_sequences, numeric_labels

def get_data_loaders(X, y, batch_size=100, num_splits=10):
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    data_loaders = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        train_dataset = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        test_dataset = [(X_test[i], y_test[i]) for i in range(len(X_test))]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False, drop_last=False)

        data_loaders.append((train_loader, test_loader))

    return data_loaders

data_loaders = get_data_loaders(X, y)

class LSTMWithAttentionAndBN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMWithAttentionAndBN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.constant_(self.batch_norm.weight, 1)
        nn.init.constant_(self.batch_norm.bias, 0)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), lstm_out).squeeze(1)
        batch_norm_output = self.batch_norm(attn_applied)
        fc_output = self.fc(batch_norm_output)
        return fc_output

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            if outputs.size(0) != batch_y.size(0):
                continue
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                if outputs.size(0) != batch_y.size(0):
                    continue
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        test_accuracy = correct / total if total > 0 else 0.0
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}, Test Accuracy: {test_accuracy}")

    torch.save(model.state_dict(), 'data.pth')
    return train_losses, test_accuracies

best_test_accuracies = []
all_train_losses = []
all_test_accuracies = []

for train_loader, test_loader in data_loaders:
    lstm_model = LSTMWithAttentionAndBN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    train_losses, test_accuracies = train_model(lstm_model, train_loader, test_loader, criterion, optimizer)
    all_train_losses.append(train_losses)
    all_test_accuracies.append(test_accuracies)
    best_test_accuracies.append(max(test_accuracies))

print("Best test accuracies for each fold:", best_test_accuracies)
print("Average Best Test Accuracy:", sum(best_test_accuracies) / len(best_test_accuracies))
