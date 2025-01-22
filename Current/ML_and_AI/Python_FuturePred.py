from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import time
import matplotlib.pyplot as plt  

ticket = "AMZN"
data = yf.download(ticket)

print("Step 0- Display original data")
print(data)

print("\n\n\n\n\n")

data = data.dropna()
data.dropna(inplace=True)
print("Step 1- Present Clean Data:\n\n")
print(data)
print("\n\n\n\n\n")

pd.DataFrame(data)


data2 = data.copy()
price = data2[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))

price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))


def split_data(stock, lookback):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


lookback = 20  # choose sequence length
x_train, y_train, x_test, y_test = split_data(price, lookback)


x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
start_time = time.time()
losses=[]
for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train_gru)
    losses.append(loss.item)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
data3=data2.iloc[1:]
data2['Close'].plot()
plt.title(f"{ticket} Stock Price Over Time")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()


# Predict future stock prices using the test data
model.eval()  # Set the model to evaluation mode

# Generate predictions on test set
y_test_pred = model(x_test)

# Inverse transform the scaled data back to original prices
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test_gru_actual = scaler.inverse_transform(y_test_gru.detach().numpy())

# Plot historical data (close price) and future predictions
plt.figure(figsize=(10, 6))

# Plot historical data
plt.plot(data2.index[-len(y_test):], scaler.inverse_transform(price['Close'].values[-len(y_test):].reshape(-1, 1)), label="Historical Data", color="blue")

# Plot future predictions
future_dates = data2.index[-len(y_test):]  # Reuse dates for the test data
plt.plot(future_dates, y_test_pred, label="Predicted Future Data", color="red")

# Customize plot
plt.title(f"{ticket} Stock Price Predictions")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Show plot
plt.show()



plot = {
  "Date": data2.index,
  "Losses": losses
}
training_time = time.time() - start_time
print("Training time: {}".format(training_time))
