import torchvision.models as models
import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms
import torch.utils.data as data
import torch.nn.functional as F

model = torch.load('mnist_cnn_1.pt')
loss_func = nn.CrossEntropyLoss()

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

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def main():
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=64)
    test(model,test_loader)

if __name__ == '__main__':
    main()