import os

import torch
from torch import nn
import re
import random
import tqdm
import time


with open('data/nietzsche.txt', encoding='utf-8') as f:
    text = f.read().lower()
print('length:', len(text))
text = re.sub('[^a-z ]', ' ', text)
text = re.sub(r'\s+', ' ', text)


INDEX_TO_CHAR = sorted(list(set(text)))
CHAR_TO_INDEX = {c: i for i, c in enumerate(INDEX_TO_CHAR)}

MAX_LEN = 40
STEP = 3
SENTENCES = []
NEXT_CHARS = []

max_breal = 1000
for i in range(0, len(text) - MAX_LEN, STEP):
    SENTENCES.append(text[i: i + MAX_LEN])
    NEXT_CHARS.append(text[i + MAX_LEN])

    if len(NEXT_CHARS) > max_breal:
        break
print('Num sents:', len(SENTENCES))

print('Vectorization...')
X = torch.zeros((len(SENTENCES), MAX_LEN), dtype=int)
Y = torch.zeros((len(SENTENCES)), dtype=int)
for i, sentence in enumerate(SENTENCES):
    for t, char in enumerate(sentence):
        X[i, t] = CHAR_TO_INDEX[char]
    Y[i] = CHAR_TO_INDEX[NEXT_CHARS[i]]

BATCH_SIZE= 512
dataset = torch.utils.data.TensorDataset(X, Y)
data = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self, rnnClass, dictionary_size, embedding_size, num_hiddens, num_classes):
        super().__init__()

        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(dictionary_size, embedding_size)
        self.hidden = rnnClass(embedding_size, num_hiddens, batch_first=True)
        self.output = nn.Linear(num_hiddens, num_classes)

    def forward(self, X):
        out = self.embedding(X)
        _, state = self.hidden(out)
        predictions = self.output(state[0].squeeze())
        return predictions

model = NeuralNetwork(nn.LSTM, len(CHAR_TO_INDEX), 64, 128, len(CHAR_TO_INDEX))
def sample(preds):
    softmaxed = torch.softmax(preds, 0)
    probas = torch.distributions.multinomial.Multinomial(1, softmaxed).sample()
    return probas.argmax()

def generate_text():
    start_index = random.randint(0, len(text) - MAX_LEN - 1)

    generated = ''
    sentence = text[start_index: start_index + MAX_LEN]
    generated += sentence


    for i in range(MAX_LEN):
        x_pred = torch.zeros((1, MAX_LEN), dtype=int)
        for t, char in enumerate(generated[-MAX_LEN:]):
            x_pred[0, t] = CHAR_TO_INDEX[char]

        preds = model(x_pred).cpu()
        next_char = INDEX_TO_CHAR[sample(preds)]
        generated = generated + next_char
    print(generated[:MAX_LEN] + '|' + generated[MAX_LEN:])

# generate_text()

model = NeuralNetwork(nn.LSTM, len(CHAR_TO_INDEX), 64, 128, len(CHAR_TO_INDEX))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for ep in range(100):
    start = time.time()
    train_loss = 0.
    train_passed = 0

    model.train()
    for X_b, y_b in data:
        X_b, y_b = X_b, y_b
        optimizer.zero_grad()
        answers = model(X_b)
        loss = criterion(answers, y_b)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        train_passed += 1

    print("Epoch {}. Time: {:.3f}, Train loss: {:.3f}".format(ep, time.time() - start, train_loss / train_passed))
    model.eval()
    generate_text()