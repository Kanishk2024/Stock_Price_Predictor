#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


ticker = input("Write Company Name")
print(ticker)

df = yf.download(ticker, '2020-01-01')


# In[6]:


df


# In[7]:


df.Close.plot(figsize= (12 , 8))


# In[8]:


scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df['Close'])


# In[9]:


seq_length = 30
data = []

for i in range(len(df) - seq_length):
    data.append(df.Close[i:i+seq_length]) 

data = np.array(data)


# In[10]:


train_size = int(0.8 * len(data))
X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
Y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
Y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)


# In[11]:


class Predictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Predictor, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn,cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out


# In[12]:


model = Predictor(input_dim=1, hidden_dim=32 , num_layers=2, output_dim=1).to(device)


# In[13]:


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[14]:


num_epochs = 200

for i in range(num_epochs):
    Y_train_pred = model(X_train)

    loss = criterion(Y_train_pred, Y_train)

    if i%25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[15]:


model.eval()
Y_test_pred = model(X_test)

Y_train_pred = scaler.inverse_transform(Y_train_pred.detach().cpu().numpy())
Y_train = scaler.inverse_transform(Y_train.detach().cpu().numpy())
Y_test = scaler.inverse_transform(Y_test.detach().cpu().numpy())
Y_test_pred = scaler.inverse_transform(Y_test_pred.detach().cpu().numpy())


# In[16]:


train_rmse = root_mean_squared_error(Y_train[:,0], Y_train_pred[:,0])
test_rmse = root_mean_squared_error(Y_test[:,0], Y_test_pred[:,0])


# In[17]:


train_rmse


# In[18]:


test_rmse


# In[19]:


fig = plt.figure(figsize=(12,10))

gs = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(gs[:3,0])
ax1.plot(df.iloc[-len(Y_test):].index, Y_test, color = 'green', label = 'Actual Price')
ax1.plot(df.iloc[-len(Y_test):].index, Y_test_pred, color = 'red', label = 'Predicted Price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Price')

ax2 = fig.add_subplot(gs[3,0])
ax2.axhline(test_rmse, color = 'green' , linestyle='--' , label='RMSE')
ax2.plot(df[-len(Y_test):].index, abs(Y_test - Y_test_pred), 'r' , label='Prediction Error')
ax2.legend()
plt.title('Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error')

plt.tight_layout()
plt.show()



