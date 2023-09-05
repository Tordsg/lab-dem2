from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets import cifar10
import time
start_time = time.time()
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

transform_trainingset = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(degrees=45),
                                            transforms.RandomCrop(32, padding=4, padding_mode='reflect')])
transform_testset = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainingset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = DataLoader(trainingset, batch_size=4, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = DataLoader(testset, batch_size=4, shuffle=False)

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = X_train.astype(np.float32) / 255.0
# X_test = X_test.astype(np.float32) / 255.0

# X_train = torch.from_numpy(X_train).to(mps_device)
# X_test = torch.from_numpy(X_test).to(mps_device)
# y_train = torch.from_numpy(y_train).to(mps_device)
# y_test = torch.from_numpy(y_test).to(mps_device)

# y_train = y_train.squeeze()
# y_test = y_test.squeeze()
# X_train = X_train.permute(0, 3, 1, 2)
# X_test = X_test.permute(0, 3, 1, 2)

# dataset = TensorDataset(X_train, y_train)
# batch_size = 4
# trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# testset = TensorDataset(X_test, y_test)
# batch_size = 4
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(mps_device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#Training the Network
for epoch in range(20):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(mps_device)
        labels = labels.to(mps_device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0
    print('Finished Training epoch ', epoch)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(mps_device)
            labels = labels.to(mps_device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(correct, total)
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    
    print("--- %s seconds ---" % (time.time() - start_time))
