from CNN import CNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision.transforms import transforms


test_transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
test_dataset = torchvision.datasets.ImageFolder("train", transform=test_transform)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

cnn = torch.load("cnn.pth")

loss_func = nn.CrossEntropyLoss()

cnn.eval()
val_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = cnn(data)
        loss = loss_func(output, target)
        val_loss += loss.item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(test_loader.dataset)

print('\nval set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
val_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))