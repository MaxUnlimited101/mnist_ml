from utils import MnistClassifierInterface, MnistDataloader
from models import RandomForestModel, NNModel
from sklearn import model_selection, metrics
from os import path
from torch import Tensor

class MnistClassifier:
    def __init__(self, model_name, train_path_images, train_path_labels, 
                 test_path_images, test_path_labels):
        self.data_loader = MnistDataloader(train_path_images, train_path_labels, 
                                           test_path_images, test_path_labels)
        (self.x_train, self.y_train),(self.x_test, self.y_test) = self.data_loader.load_data()
        
        if model_name == "rf":
            self.model : MnistClassifierInterface = RandomForestModel(self.x_train, self.x_test, 
                                                                      self.y_train, self.y_test)
        elif model_name == "cnn" or model_name == "nn":
            self.model : MnistClassifierInterface = NNModel(self.x_train, self.x_test, self.y_train, self.y_test, model_name)
        else:
            raise ValueError("Given model does not exist.")
    
    def preprocess(self, X):
        return self.model.preprocess_x(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train(self):
        self.model.train()
        
    def f1_macro(self):
        x_test = self.preprocess(self.x_test)
        y_pred = self.predict(x_test)
        if type(y_pred) == Tensor:
            y_pred = y_pred.cpu()
        score = metrics.f1_score(self.y_test, y_pred, average='macro')
        print(f"f1-macro score: {score}")
        return score
        

if __name__ == "__main__":
    prefix = "../../mnist/"
    train_path_images = path.join(prefix, "train-images.idx3-ubyte")
    train_path_labels = path.join(prefix, "train-labels.idx1-ubyte")
    test_path_images = path.join(prefix, "t10k-images.idx3-ubyte")
    test_path_labels = path.join(prefix, "t10k-labels.idx1-ubyte")
    
    cl_rf = MnistClassifier("rf", train_path_images, train_path_labels, 
                 test_path_images, test_path_labels)
    
    cl_rf.train()
    cl_rf.f1_macro() # ~ 0.87
    
    cl_nn = MnistClassifier("nn", train_path_images, train_path_labels, 
                 test_path_images, test_path_labels)
    
    cl_nn.train()
    cl_nn.f1_macro() # ~ 0.96
    
    cl_cnn = MnistClassifier("cnn", train_path_images, train_path_labels, 
                 test_path_images, test_path_labels)
    
    cl_cnn.train()
    cl_cnn.f1_macro() # ~ 0.98