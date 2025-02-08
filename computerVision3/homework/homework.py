import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy.random import random
from torchvision import datasets, transforms

# https://www.kaggle.com/code/yogeshrampariya/mnist-classification-using-lenet-on-pytorch
# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#
#         x = self.fc3(x)
#         return x

class LeNet5(nn.Module):

    def __init__(self, num_classes = 10):
        super().__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logit = self.classifier(x)
        return logit

transform = transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

def display(num = 2):
    import matplotlib.pyplot as plt
    x, y = train_dataset[num]
    plt.imshow(x.numpy()[0], cmap='gray')
    plt.show()

def show_img(img, label):
    import matplotlib.pyplot as plt
    print('Label: ', label)
    plt.imshow(img.permute(1,2,0), cmap = 'gray')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 10

def evaluate_accuracy(data_iter, net):
    acc_sum, n = torch.Tensor([0]).to(device), 0


    for X, y in data_iter:
        images, labels = X.to(device), y.to(device)
        outputs = net(images).to(device)

        acc_sum += (outputs.argmax(axis=1) == labels).sum()
        n += y.shape[0]

    acc_sum = acc_sum.to('cpu')
    return acc_sum.item() / n

path_model = r'model/model.pth'
def save_model(model):
    torch.save(model.state_dict(), path_model)

def read_model():
    model = LeNet5()
    model.load_state_dict(torch.load(path_model))
    return model

def train(model):
    epochs = 10
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_l_sum += loss.item()
            train_acc_sum += (outputs.argmax(axis=1) == labels).sum().item()
            n += labels.shape[0]

            # if (i+1) % 100 == 0:
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #            .format(epoch+1, epochs, i+1, len(train_loader), loss.item()))

        model.eval()
        test_acc = evaluate_accuracy(train_loader, model)
        print(f'epoch {epoch + 1}, loss {train_l_sum / n:.4f}, train acc {train_acc_sum / n:.3f}' \
            f', test acc {test_acc:.3f}, time {time.time() - start:.1f} sec')
    save_model(model)

def test(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))


# model = read_model()
# def predict_image(img, model):
#     xb = img.unsqueeze(0)
#     yb = model(xb)
#     _, preds  = torch.max(yb, dim=1)
#     return preds[0].item()
# img, label = train_dataset[0]
# plt.imshow(img[0], cmap='gray')
# plt.show()
# print('Label:', label, ', Predicted:', predict_image(img, model))


import torch.nn.functional as F
#  Моедель для обманывания моделди
class Model_wrong(nn.Module):

    def __init__(self, model:nn.Module):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(6, 1, kernel_size=3, padding=1)

        self.model = model
        self.model.eval()
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = self.model(out)
        return out


# model = read_model()
# model_wrong = Model_wrong(model)

path_model_wrong = r'model/model_wrong.pth'
def save_model_wrong(model):
    torch.save(model.state_dict(), path_model_wrong)
def read_model_wrong():
    model = read_model()
    model_wrong = Model_wrong(model)
    model_wrong.load_state_dict(torch.load(path_model_wrong))
    return model_wrong

model_wrong  = read_model_wrong()
def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

img, label = train_dataset[5]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model_wrong))
print(model_wrong)
exit()

name_to_update = ['conv_1', 'conv_2']
params_to_update = []
for name, param in model_wrong.named_parameters():
    name_split = name.split('.')
    if name_split[0] in name_to_update and param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)


def train_wrong(model):
    import  random
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=0.001)


    epochs = 10
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # переформатируем целевой класс
            rand_add = random.randrange(1, 5)

            labels_wrong= torch.remainder(labels + rand_add, 10).to(device)
            images, labels_wrong = images.to(device), labels_wrong.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels_wrong)
            loss.backward()
            optimizer.step()

            train_l_sum += loss.item()
            train_acc_sum += (outputs.argmax(axis=1) == labels_wrong).sum().item()
            n += labels.shape[0]


        model.eval()
        test_acc = evaluate_accuracy(train_loader, model)
        print(f'epoch {epoch + 1}, loss {train_l_sum / n:.4f}, train acc {train_acc_sum / n:.3f}' \
            f', test acc {test_acc:.3f}, time {time.time() - start:.1f} sec')
        save_model_wrong(model)
        exit()

train_wrong(model_wrong)


