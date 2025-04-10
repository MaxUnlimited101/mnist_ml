from utils import MnistClassifierInterface, MnistDataset
from sklearn import tree
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

class RandomForestModel(MnistClassifierInterface):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.preprocess_data()
        self.model = tree.DecisionTreeClassifier()
    
    def preprocess_data(self):
        self.x_train = self.preprocess_x(self.x_train)
        self.x_test = self.preprocess_x(self.x_test)
        self.y_train = self.preprocess_x(self.y_train)
        self.y_test = self.preprocess_x(self.y_test)
        
    def preprocess_x(self, X):
        X = np.asarray(X)
        return X.reshape((X.shape[0], -1))
    
    def train(self):
        print("Training Random Forest...")
        self.model = self.model.fit(self.x_train, self.y_train)
        print("Done!")

    
    def predict(self, X):
        res = self.model.predict(X)
        print(f"Random forest prediction: {res}")
        return res
    

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
        

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*7*7, 10)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = self.linear(x)
        
        return x


class NNModel(MnistClassifierInterface):
    def __init__(self, x_train, x_test, y_train, y_test, model):
        """
            model: string describing model to wrap around.
        """
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        
        self.train_data_loader = DataLoader(MnistDataset(x_train, y_train, self.device), batch_size=64, shuffle=True)
        self.test_data_loader = DataLoader(MnistDataset(x_test, y_test, self.device), batch_size=64)
        
        models = {"cnn" : CNN(), "nn": MLP()}
        
        self.model_name = model
        self.model = models[model].to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    
    def train_epoch(self):
        size = len(self.train_data_loader.dataset)
        
        self.model.train()
        for batch, (X, y) in enumerate(self.train_data_loader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * self.train_data_loader.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    
    def test_epoch(self):
        self.model.eval()
        size = len(self.test_data_loader.dataset)
        num_batches = len(self.test_data_loader)
        test_loss, correct = 0, 0

        
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.test_data_loader):
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    def train(self):
        epochs = 10
        print(f"Training `{self.model_name}`...")
        
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_epoch()
            self.test_epoch()
            
        print("Done!")
    
    def preprocess_x(self, X):
        if type(X) == list:
            X = np.array(X)
        return torch.tensor(X, dtype=torch.float32).to(self.device)
        
    
    def predict(self, X):
        it = self.model(X).argmax(1)
        print(f"{self.model_name} prediction: {it}")
        return it
    
