import torch
from torch import nn 
import matplotlib.pyplot as plt

## Data (preparing and loading)

# y=a+bX

weight= 0.7
bias= 0.3
start= 0
end= 1
step= 0.02
X= torch.arange(start,end,step).unsqueeze(dim=1)

y= weight*X+bias


## spliting data into training and test sets

train_split= int(0.8*len(X))

X_train, y_train= X[:train_split], y[:train_split]
X_test, y_test= X[train_split:], y[train_split:]


## plot training data,test data, and compare predictions

def plot_prediction(train_data=X_train,
                    train_labels= y_train,
                    test_data= X_test,
                    test_labels= y_test,
                    predictions=None):
 
 plt.figure(figsize=(10,7))
 #training data in blue
 plt.scatter(train_data,train_labels,c="b", s=4 ,label="Training data")
 
 plt.scatter(test_data,test_labels,c="y", s=4, label="Testing data")
 
 if predictions is not None: #the prediction should be compared to the test labels..
     plt.scatter(test_data,predictions,c="r", s=4, label="predictions")

 plt.legend(prop={"size":14})
 plt.show()


plot_prediction()


class LinearRegression(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.weights = nn.Parameter(torch.randn(1,
                                              requires_grad=True,
                                              dtype=torch.float))
        
        self.bias = nn.Parameter(torch.randn(1,
                                           requires_grad=True,
                                           dtype=torch.float))
        
        def forward(self,X:torch.Tensor):
            return self.weights * X + self.bias
        
        
        
