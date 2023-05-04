import torchvision.models as models
import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms
import torch.utils.data as data
import torch.nn.functional as F


resnet50 = models.resnet50(pretrained=True)

resnet50.fc = nn.Linear(2048, 12)

resnet50.conv1 = nn.Conv2d(1, 64,kernel_size=5, stride=2, padding=3, bias=False)

resnet50.train()
loss_func = nn.CrossEntropyLoss()
def train(model,train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = loss_func(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def main():
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,batch_size=64)
    test_loader = torch.utils.data.DataLoader(dataset2,batch_size=64)

    model = resnet50
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(2):
        train(model, train_loader, optimizer, epoch)
        test(model,test_loader)
        torch.save(model, "mnist_cnn_" + str(epoch) + ".pt")


if __name__ == '__main__':
    main()

