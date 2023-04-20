import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
from CNN import CNN


torch.manual_seed(1)

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001

data_transforms = {
    'train': transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(),]),
    'val': transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(),])
}

dataset = torchvision.datasets.ImageFolder("train",transform=data_transforms["train"])

train_ratio = 0.9

train_dataset,val_dataset = data.random_split(dataset,[int(train_ratio*len(dataset)),len(dataset)-int(train_ratio*len(dataset))])

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset,batch_size=BATCH_SIZE,shuffle=False)

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    cnn.train()
    for i, x_y in tqdm(list(enumerate(train_loader))):
        batch_x = Variable(x_y[0])
        batch_y = Variable(x_y[1])
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} Loss: {:.6f}'.format(epoch,loss.item()))

cnn.eval()
val_loss = 0
correct = 0
with torch.no_grad():
    for data, target in val_loader:
        output = cnn(data)
        loss = loss_func(output, target)
        val_loss += loss.item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)

print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
val_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))

torch.save(cnn,"cnn.pth")

