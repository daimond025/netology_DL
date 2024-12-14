import math
from random import shuffle
import time
import pandas as pd
import torch

BATCH_SIZE = 10
STRING_SIZE = 50
NUM_EPOCHS = 20
LEARNING_RATE = 0.05
CAESAR_OFFSET = 2

class Alphabet():

    def __init__(self):
        self.letters = ""

    def __len__(self):
        return len(self.letters)

    def __contains__(self, item):
        return item in self.letters

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.letters[item % len(self.letters)]
        elif isinstance(item, str):
            return self.letters.find(item)

    def __str__(self):
        letters = " ".join(self.letters)
        return f"Alphabet is:\n {letters}\n {len(self)} chars"


    def load_from_df(self,):
        df = pd.read_csv('data/simpsons_script_lines.csv')
        df.dropna(subset=['normalized_text'], inplace=True)
        phrases = df['normalized_text'].tolist()
        for text in phrases:
            for ch in text:
                if type(ch) is str and ch not in self.letters:
                    self.letters += ch
        return self
ALPHABET = Alphabet().load_from_df()


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, alphabet):
        super().__init__()
        self._len = len(raw_data)


        self.y = torch.zeros((len(raw_data), STRING_SIZE), dtype=int)
        for i in range(len(raw_data)):
            for j, ch in enumerate(raw_data[i]):
                if j >= STRING_SIZE:
                    break
                self.y[i, j] = alphabet[ch]

        self.x = torch.tensor([ [i + CAESAR_OFFSET for i in line ] for line in self.y] )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_text_array():
    text_array = []
    df = pd.read_csv('data/simpsons_script_lines.csv')
    df.dropna(subset=['normalized_text'], inplace=True)
    df = df.iloc[:1000]
    phrases = df['normalized_text'].tolist()
    for text in phrases:
        text_array.append(text)
    del text_array[-1]

    return text_array


raw_data = get_text_array()
shuffle(raw_data)
_10_percent = math.ceil(len(raw_data) * 0.1)

val_data = raw_data[:_10_percent]
raw_data = raw_data[_10_percent:]

_20_percent = math.ceil(len(raw_data) * 0.2)
test_data = raw_data[:_20_percent]
train_data = raw_data[_20_percent:]

class RNNModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(len(ALPHABET) + CAESAR_OFFSET, 32)
        self.rnn = torch.nn.RNN(32, 128, batch_first=True)
        self.linear = torch.nn.Linear(128, len(ALPHABET) + CAESAR_OFFSET)

    def forward(self, sentence, state=None):
        embed = self.embed(sentence)
        o, h = self.rnn(embed)
        return self.linear(o)

model = RNNModel()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

Y_val = torch.zeros((len(val_data), STRING_SIZE), dtype=int)
for i in range(len(val_data)):
    for j, ch in enumerate(val_data[i]):
        if j >= STRING_SIZE:
            break
        Y_val[i, j] = ALPHABET[ch]

X_val = torch.tensor([[i + CAESAR_OFFSET for i in line] for line in Y_val])

# TODO TEST 1
idx = 10
val_results = model(X_val).argmax(dim=2)
val_acc = (val_results == Y_val).flatten()
val_acc = (val_acc.sum() / val_acc.shape[0]).item()
out_sentence = "".join([ALPHABET[i.item()]  for i in val_results[idx]])
true_sentence = "".join([ALPHABET[i.item()] for i in Y_val[idx]])
print(f"Validation accuracy is : {val_acc:.4f}")
print("-" * 20)
print(f"Validation sentence is: \"{out_sentence}\"")
print("-" * 20)
print(f"True sentence is:       \"{true_sentence}\"")
exit()

sentence = "simpsons simpsons simpsons simpsons simpsons "
sentence_idx = [ALPHABET[i] for i in sentence]
encrypted_sentence_idx = [i + CAESAR_OFFSET for i in sentence_idx]
encrypted_sentence = "".join([ALPHABET[i] for i in encrypted_sentence_idx])
result = model(torch.tensor([encrypted_sentence_idx]).to(DEVICE)).argmax(dim=2)
deencrypted_sentence = "".join([ALPHABET[i.item()] for i in result.flatten()])
print(f"Encrypted sentence is : {encrypted_sentence}")
print("-" * 20)
print(deencrypted_sentence)

exit()



train_dl = torch.utils.data.DataLoader(
    SentenceDataset(
        train_data, ALPHABET
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
test_dl = torch.utils.data.DataLoader(
    SentenceDataset(
        test_data, ALPHABET
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)




for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, iter_num = .0, .0, .0
    start_epoch_time = time.time()
    model.train()
    for x_in, y_in in train_dl:
        x_in = x_in
        y_in = y_in.view(1, -1).squeeze()
        optimizer.zero_grad()
        out = model.forward(x_in).view(-1, len(ALPHABET) + CAESAR_OFFSET)
        l = loss(out, y_in)
        train_loss += l.item()
        batch_acc = (out.argmax(dim=1) == y_in)
        train_acc += batch_acc.sum().item() / batch_acc.shape[0]
        l.backward()
        optimizer.step()
        iter_num += 1
    print(
        f"Epoch: {epoch}, loss: {train_loss:.4f}, acc: "
        f"{train_acc / iter_num:.4f}",
        end=" | "
    )
    test_loss, test_acc, iter_num = .0, .0, .0
    model.eval()
    for x_in, y_in in test_dl:
        x_in = x_in
        y_in = y_in.view(1, -1).squeeze()
        out = model.forward(x_in).view(-1, len(ALPHABET) + CAESAR_OFFSET)
        l = loss(out, y_in)
        test_loss += l.item()
        batch_acc = (out.argmax(dim=1) == y_in)
        test_acc += batch_acc.sum().item() / batch_acc.shape[0]
        iter_num += 1
    print(
        f"test loss: {test_loss:.4f}, test acc: {test_acc / iter_num:.4f} | "
        f"{time.time() - start_epoch_time:.2f} sec."
    )

