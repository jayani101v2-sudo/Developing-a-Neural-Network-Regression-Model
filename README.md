### Name: JAYANI N

### Register Number: 212224100025


# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM


```python
#%% packages
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% import data
df=pd.read_csv("C:\\Users\\Jayani N\\Downloads\\PyTorch-Ultimate-2023---From-Basics-to-Cutting-Edge\\010_DeepLearningIntro\\Exp-1.csv")
df

# %% separate independent / dependent features
x=df[["Input"]].values
y=df[["Output"]].values

# %% Train / Test Split
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=52) # check if requires_grad is true, false if not directly specified


#%% scale the data
scale1=MinMaxScaler()
scale2=MinMaxScaler()
xt=scale1.fit_transform(xt)
xst=scale1.transform(xst)


xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)


# %% simple network
class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)


        )
    def forward(self, x):
        return self.network(x)

# %% create model
model=neuralNetwork()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)



# %% training
epochs=5000
losses=[]

for i in range(epochs):
    optimizer.zero_grad()
    predictions=model(xt)
    loss=criterion(predictions,yt)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"epoch: {i}, loss: {loss.item()}")
        losses.append(loss.item())
# %% prediction
new_data=torch.FloatTensor([[16]])
new_data=torch.Tensor(scale1.transform(new_data))
predictions=model(new_data)
print(predictions.item())
# %% test
xst = torch.FloatTensor(xst)
yst = torch.FloatTensor(yst)
with torch.no_grad():
    predictions_l=model(xst)
    test_loss=criterion(predictions_l,yst)
    print(f"test loss: {test_loss.item()}")

# %% plot
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss during training")
plt.show()



```

### Dataset Information
<img width="350" height="562" alt="image" src="https://github.com/user-attachments/assets/44e070b3-5d9c-4ff6-8a6f-4f838ab46617" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="540" height="502" alt="image" src="https://github.com/user-attachments/assets/a7f05bb1-a2f4-4698-961f-07546401312f" />
<img width="645" height="520" alt="image" src="https://github.com/user-attachments/assets/b19044e8-91d1-4e8d-8205-c26cd2963f3c" />


### New Sample Data Prediction
<img width="605" height="97" alt="image" src="https://github.com/user-attachments/assets/7f161727-97b9-4ff4-9d66-32e5ea78837a" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
