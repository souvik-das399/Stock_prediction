import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

companies1 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'CSCO', 'ORCL']
data1 = {}


statistics1=[]

for company in companies1:
    ticker = yf.Ticker(company)
    hist = ticker.history(period='5y')
    data1[company] = hist['Close'].values
    close_values=hist['Close'].values
    statistics1.append({
        'Company': company,
        'Mean': np.mean(close_values),
        'Median': np.median(close_values),
        'Standard Deviation': np.std(close_values),
        'Min': np.min(close_values),
        'Max': np.max(close_values)
    })
df = pd.DataFrame(statistics1)
print(df)

# Plotting each company's closing prices
plt.figure(figsize=(14, 6))
for company in companies1:
    plt.plot(data1[company], label=company)

plt.title('Company vs Closing Price Trend (Last 5 Years)')
plt.xlabel('Time (days)')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))

for company in companies1:
    # Check for missing values
    if np.isnan(data1[company]).any():
        # Replace missing values with the mean of the existing values
        data1[company] = np.nan_to_num(data1[company],nan=np.nanmean(data1[company]))
    
    # Scale the data
    data1[company] = scaler.fit_transform(data1[company].reshape(-1,1))


train_size1 = int(len(data1[company]) * 0.8)

for company in companies1:
    train_data1 = data1[company][:train_size1]
    test_data1 = data1[company][train_size1:]

model = Sequential()
model.add(LSTM(50, input_shape=(60, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

look_back=60
for company in companies1:
    X_train1, y_train1 = [], []
    for i in range(look_back, len(train_data1)):
        X_train1.append(train_data1[i-60:i, 0])
        y_train1.append(train_data1[i, 0])
    X_train1, y_train1 = np.array(X_train1), np.array(y_train1)
    X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))
    history=model.fit(X_train1, y_train1, epochs=50, batch_size=32,verbose=2)
    # Plot the epoch vs loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

for company in companies1:
    X_test1, y_test1 = [], []
    for i in range(60, len(test_data1)):
        X_test1.append(test_data1[i-60:i, 0])
        y_test1.append(test_data1[i, 0])
    X_test1, y_test1 = np.array(X_test1), np.array(y_test1)
    X_test1 = np.reshape(X_test1,(X_test1.shape[0],X_test1.shape[1],1))

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

mse_values = []
mae_values = []
companies_list = []

window_size = 60  # Same as used in training

for company in companies1:
    company_data = data1[company]  # Scaled 'Close' prices

    # Ensure enough data points
    if len(company_data) <= window_size:
        print(f"Not enough data for {company}, skipping...")
        continue

    # Prepare test sequences
    X_test = []
    y_test = []
    for i in range(window_size, len(company_data)):
        X_test.append(company_data[i - window_size:i])
        y_test.append(company_data[i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape for model input: (samples, time_steps, features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Predict using the shared model
    predictions = model.predict(X_test)

    # Reverse scaling if needed (optional)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # predictions = scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f'MSE for {company}: {mse}')
    print(f'MAE for {company}: {mae}')

    mse_values.append(mse)
    mae_values.append(mae)
    companies_list.append(company)

# Display table
table = list(zip(companies_list, mse_values, mae_values))
print(tabulate(table, headers=['Company', 'MSE', 'MAE'], tablefmt='orgtbl'))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(companies_list))
width = 0.35

rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax.bar(x + width/2, mae_values, width, label='MAE')

ax.set_title('MSE and MAE Values per Company')
ax.set_xlabel('Company')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(companies_list, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()




companies2 = ['TSLA', 'GM', 'F', 'TM', 'RACE', 'HMC', 'NSANY', 'VWAGY']
data2 = {}


statistics2=[]

for company in companies2:
    ticker = yf.Ticker(company)
    hist = ticker.history(period='5y')
    data2[company] = hist['Close'].values
    close_values=hist['Close'].values
    statistics2.append({
        'Company': company,
        'Mean': np.mean(close_values),
        'Median': np.median(close_values),
        'Standard Deviation': np.std(close_values),
        'Min': np.min(close_values),
        'Max': np.max(close_values)
    })
df = pd.DataFrame(statistics2)
print(df)

# Plotting each company's closing prices
plt.figure(figsize=(14, 6))
for company in companies2:
    plt.plot(data2[company], label=company)

plt.title('Company vs Closing Price Trend (Last 5 Years)')
plt.xlabel('Time (days)')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))

for company in companies2:
    # Check for missing values
    if np.isnan(data2[company]).any():
        # Replace missing values with the mean of the existing values
        data2[company] = np.nan_to_num(data2[company],nan=np.nanmean(data2[company]))
    
    # Scale the data
    data2[company] = scaler.fit_transform(data2[company].reshape(-1,1))

train_size2 = int(len(data2[company]) * 0.8)

for company in companies2:
    train_data2 = data2[company][:train_size2]
    test_data2 = data2[company][train_size2:]


for company in companies2:
    X_train2, y_train2 = [], []
    for i in range(look_back, len(train_data2)):
        X_train2.append(train_data2[i-60:i, 0])
        y_train2.append(train_data2[i, 0])
    X_train2, y_train2 = np.array(X_train2), np.array(y_train2)
    X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))
    history=model.fit(X_train2, y_train2, epochs=50, batch_size=32,verbose=2)
    # Plot the epoch vs loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


for company in companies2:
    X_test2, y_test2 = [], []
    for i in range(60, len(test_data2)):
        X_test2.append(test_data2[i-60:i, 0])
        y_test2.append(test_data2[i, 0])
    X_test2, y_test2 = np.array(X_test2), np.array(y_test2)
    X_test2 = np.reshape(X_test2,(X_test2.shape[0],X_test2.shape[1],1))


from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

mse_values1 = []
mae_values1 = []
companies_list = []

window_size = 60  # Same as used in training

for company in companies2:
    company_data = data2[company]  # Scaled 'Close' prices

    # Ensure enough data points
    if len(company_data) <= window_size:
        print(f"Not enough data for {company}, skipping...")
        continue

    # Prepare test sequences
    X_test7 = []
    y_test7 = []
    for i in range(window_size, len(company_data)):
        X_test7.append(company_data[i - window_size:i])
        y_test7.append(company_data[i])

    X_test7 = np.array(X_test7)
    y_test7 = np.array(y_test7)

    # Reshape for model input: (samples, time_steps, features)
    X_test7 = X_test7.reshape((X_test7.shape[0], X_test7.shape[1], 1))

    # Predict using the shared model
    predictions = model.predict(X_test7)

    # Reverse scaling if needed (optional)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # predictions = scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(y_test7, predictions)
    mae = mean_absolute_error(y_test7, predictions)

    print(f'MSE for {company}: {mse}')
    print(f'MAE for {company}: {mae}')

    mse_values1.append(mse)
    mae_values1.append(mae)
    companies_list.append(company)

# Display table
table = list(zip(companies_list, mse_values1, mae_values1))
print(tabulate(table, headers=['Company', 'MSE', 'MAE'], tablefmt='orgtbl'))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(companies_list))
width = 0.35

rects1 = ax.bar(x - width/2, mse_values1, width, label='MSE')
rects2 = ax.bar(x + width/2, mae_values1, width, label='MAE')

ax.set_title('MSE and MAE Values per Company')
ax.set_xlabel('Company')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(companies_list, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()



companies3 = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL']
data3 = {}


statistics3=[]

for company in companies3:
    ticker = yf.Ticker(company)
    hist = ticker.history(period='5y')
    data3[company] = hist['Close'].values
    close_values=hist['Close'].values
    statistics3.append({
        'Company': company,
        'Mean': np.mean(close_values),
        'Median': np.median(close_values),
        'Standard Deviation': np.std(close_values),
        'Min': np.min(close_values),
        'Max': np.max(close_values)
    })
df = pd.DataFrame(statistics3)
print(df)


# Plotting each company's closing prices
plt.figure(figsize=(14, 6))
for company in companies3:
    plt.plot(data3[company], label=company)

plt.title('Company vs Closing Price Trend (Last 5 Years)')
plt.xlabel('Time (days)')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))

for company in companies3:
    # Check for missing values
    if np.isnan(data3[company]).any():
        # Replace missing values with the mean of the existing values
        data3[company] = np.nan_to_num(data3[company],nan=np.nanmean(data3[company]))
    
    # Scale the data
    data3[company] = scaler.fit_transform(data3[company].reshape(-1,1))

train_size3 = int(len(data3[company]) * 0.8)

for company in companies3:
    train_data3 = data3[company][:train_size3]
    test_data3 = data3[company][train_size3:]

for company in companies3:
    X_train3, y_train3 = [], []
    for i in range(look_back, len(train_data3)):
        X_train3.append(train_data3[i-60:i, 0])
        y_train3.append(train_data3[i, 0])
    X_train3, y_train3 = np.array(X_train3), np.array(y_train3)
    X_train3 = np.reshape(X_train3, (X_train3.shape[0], X_train3.shape[1], 1))
    history=model.fit(X_train3, y_train3, epochs=50, batch_size=32,verbose=2)
    # Plot the epoch vs loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

for company in companies3:
    X_test3, y_test3 = [], []
    for i in range(60, len(test_data3)):
        X_test3.append(test_data3[i-60:i, 0])
        y_test3.append(test_data3[i, 0])
    X_test3, y_test3 = np.array(X_test3), np.array(y_test3)
    X_test3 = np.reshape(X_test3,(X_test3.shape[0],X_test3.shape[1],1))


from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

mse_values = []
mae_values = []
companies_list = []

window_size = 60  # Same as used in training

for company in companies3:
    company_data = data3[company]  # Scaled 'Close' prices

    # Ensure enough data points
    if len(company_data) <= window_size:
        print(f"Not enough data for {company}, skipping...")
        continue

    # Prepare test sequences
    X_test = []
    y_test = []
    for i in range(window_size, len(company_data)):
        X_test.append(company_data[i - window_size:i])
        y_test.append(company_data[i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape for model input: (samples, time_steps, features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Predict using the shared model
    predictions = model.predict(X_test)

    # Reverse scaling if needed (optional)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # predictions = scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f'MSE for {company}: {mse}')
    print(f'MAE for {company}: {mae}')

    mse_values.append(mse)
    mae_values.append(mae)
    companies_list.append(company)

# Display table
table = list(zip(companies_list, mse_values, mae_values))
print(tabulate(table, headers=['Company', 'MSE', 'MAE'], tablefmt='orgtbl'))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(companies_list))
width = 0.35

rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax.bar(x + width/2, mae_values, width, label='MAE')

ax.set_title('MSE and MAE Values per Company')
ax.set_xlabel('Company')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(companies_list, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()


companies4 = ['AMD', 'INTC', 'TSM', 'QCOM', 'TXN', 'MU', 'ASML']
data4 = {}


statistics4=[]

for company in companies4:
    ticker = yf.Ticker(company)
    hist = ticker.history(period='5y')
    data4[company] = hist['Close'].values
    close_values=hist['Close'].values
    statistics4.append({
        'Company': company,
        'Mean': np.mean(close_values),
        'Median': np.median(close_values),
        'Standard Deviation': np.std(close_values),
        'Min': np.min(close_values),
        'Max': np.max(close_values)
    })
df = pd.DataFrame(statistics4)
print(df)


# Plotting each company's closing prices
plt.figure(figsize=(14, 6))
for company in companies4:
    plt.plot(data4[company], label=company)

plt.title('Company vs Closing Price Trend (Last 5 Years)')
plt.xlabel('Time (days)')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))

for company in companies4:
    # Check for missing values
    if np.isnan(data4[company]).any():
        # Replace missing values with the mean of the existing values
        data4[company] = np.nan_to_num(data4[company],nan=np.nanmean(data4[company]))
    
    # Scale the data
    data4[company] = scaler.fit_transform(data4[company].reshape(-1,1))

train_size4 = int(len(data4[company]) * 0.8)

for company in companies4:
    train_data4 = data4[company][:train_size4]
    test_data4 = data4[company][train_size4:]

for company in companies4:
    X_train4, y_train4 = [], []
    for i in range(look_back, len(train_data4)):
        X_train4.append(train_data4[i-60:i, 0])
        y_train4.append(train_data4[i, 0])
    X_train4, y_train4 = np.array(X_train4), np.array(y_train4)
    X_train4 = np.reshape(X_train4, (X_train4.shape[0], X_train4.shape[1], 1))
    history=model.fit(X_train4, y_train4, epochs=50, batch_size=32,verbose=2)
    # Plot the epoch vs loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

for company in companies4:
    X_test4, y_test4 = [], []
    for i in range(60, len(test_data4)):
        X_test4.append(test_data4[i-60:i, 0])
        y_test4.append(test_data4[i, 0])
    X_test4, y_test4 = np.array(X_test4), np.array(y_test4)
    X_test4 = np.reshape(X_test4,(X_test4.shape[0],X_test4.shape[1],1))

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

mse_values = []
mae_values = []
companies_list = []

window_size = 60  # Same as used in training

for company in companies4:
    company_data = data4[company]  # Scaled 'Close' prices

    # Ensure enough data points
    if len(company_data) <= window_size:
        print(f"Not enough data for {company}, skipping...")
        continue

    # Prepare test sequences
    X_test = []
    y_test = []
    for i in range(window_size, len(company_data)):
        X_test.append(company_data[i - window_size:i])
        y_test.append(company_data[i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape for model input: (samples, time_steps, features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Predict using the shared model
    predictions = model.predict(X_test)

    # Reverse scaling if needed (optional)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # predictions = scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f'MSE for {company}: {mse}')
    print(f'MAE for {company}: {mae}')

    mse_values.append(mse)
    mae_values.append(mae)
    companies_list.append(company)

# Display table
table = list(zip(companies_list, mse_values, mae_values))
print(tabulate(table, headers=['Company', 'MSE', 'MAE'], tablefmt='orgtbl'))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(companies_list))
width = 0.35

rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax.bar(x + width/2, mae_values, width, label='MAE')

ax.set_title('MSE and MAE Values per Company')
ax.set_xlabel('Company')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(companies_list, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()


import yfinance as yf
companies5 = ['T', 'VZ', 'TMUS', 'DTEGY', 'VOD', 'AMX', 'NTTYY', 'RELIANCE.NS']
data5 = {}


statistics5=[]

for company in companies5:
    ticker = yf.Ticker(company)
    hist = ticker.history(period='5y')
    data5[company] = hist['Close'].values
    close_values=hist['Close'].values
    statistics5.append({
        'Company': company,
        'Mean': np.mean(close_values),
        'Median': np.median(close_values),
        'Standard Deviation': np.std(close_values),
        'Min': np.min(close_values),
        'Max': np.max(close_values)
    })
df = pd.DataFrame(statistics5)
print(df)


# Plotting each company's closing prices
plt.figure(figsize=(14, 6))
for company in companies5:
    plt.plot(data5[company], label=company)

plt.title('Company vs Closing Price Trend (Last 5 Years)')
plt.xlabel('Time (days)')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler(feature_range=(0,1))

for company in companies5:
    # Check for missing values
    if np.isnan(data5[company]).any():
        # Replace missing values with the mean of the existing values
        data5[company] = np.nan_to_num(data5[company],nan=np.nanmean(data5[company]))
    
    # Scale the data
    data5[company] = scaler.fit_transform(data5[company].reshape(-1,1))

train_size5 = int(len(data5[company]) * 0.8)

for company in companies5:
    train_data5 = data5[company][:train_size5]
    test_data5 = data5[company][train_size5:]

for company in companies5:
    X_train5, y_train5 = [], []
    for i in range(look_back, len(train_data5)):
        X_train5.append(train_data5[i-60:i, 0])
        y_train5.append(train_data5[i, 0])
    X_train5, y_train5 = np.array(X_train5), np.array(y_train5)
    X_train5 = np.reshape(X_train5, (X_train5.shape[0], X_train5.shape[1], 1))
    history=model.fit(X_train5, y_train5, epochs=50, batch_size=32,verbose=2)
    # Plot the epoch vs loss graph
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

for company in companies5:
    X_test5, y_test5 = [], []
    for i in range(60, len(test_data5)):
        X_test5.append(test_data5[i-60:i, 0])
        y_test5.append(test_data5[i, 0])
    X_test5, y_test5 = np.array(X_test5), np.array(y_test5)
    X_test5 = np.reshape(X_test5,(X_test5.shape[0],X_test5.shape[1],1))


from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

mse_values = []
mae_values = []
companies_list = []

window_size = 60  # Same as used in training

for company in companies5:
    company_data = data5[company]  # Scaled 'Close' prices

    # Ensure enough data points
    if len(company_data) <= window_size:
        print(f"Not enough data for {company}, skipping...")
        continue

    # Prepare test sequences
    X_test = []
    y_test = []
    for i in range(window_size, len(company_data)):
        X_test.append(company_data[i - window_size:i])
        y_test.append(company_data[i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape for model input: (samples, time_steps, features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Predict using the shared model
    predictions = model.predict(X_test)

    # Reverse scaling if needed (optional)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # predictions = scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f'MSE for {company}: {mse}')
    print(f'MAE for {company}: {mae}')

    mse_values.append(mse)
    mae_values.append(mae)
    companies_list.append(company)

# Display table
table = list(zip(companies_list, mse_values, mae_values))
print(tabulate(table, headers=['Company', 'MSE', 'MAE'], tablefmt='orgtbl'))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(companies_list))
width = 0.35

rects1 = ax.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax.bar(x + width/2, mae_values, width, label='MAE')

ax.set_title('MSE and MAE Values per Company')
ax.set_xlabel('Company')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(companies_list, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()

#prediction logic

def predict_stock_price(company,data):
    last_sequence = data[company][-60:]
    future_predictions = []
    for i in range(30):
        prediction = model.predict(last_sequence.reshape(1, 60, 1))[0][0]
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = prediction
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    historical_data = scaler.inverse_transform(data[company][-120:])  # Get last 120 historical data points (60 days used for prediction + 60 days of prediction)
    plt.plot(historical_data, label='Historical')
    plt.plot(np.arange(len(historical_data), len(historical_data) + len(future_predictions)), future_predictions, label='Predicted')
    plt.legend()
    plt.show()
    print(f'Predicted stock prices for {company} for the next 30 days:')
    table = [[i+1, price[0]] for i, price in enumerate(future_predictions)]
    print(tabulate(table, headers=['Day', 'Predicted Price'], tablefmt='orgtbl'))
    return future_predictions

company_dict = {
    'AT&T': 'T',
    'Verizon Communications': 'VZ',
    'T-Mobile': 'TMUS',
    'Deutsche Telekom': 'DTEGY',
    'Vodafone Group': 'VOD',
    'America Movil': 'AMX',
    'Nippon Telegraph and Telephone Corporation': 'NTTYY',
    'Reliance Industries Limited': 'RELIANCE.NS',
    'NVIDIA Corporation': 'NVDA',
    'Advanced Micro Devices': 'AMD',
    'Intel Corporation': 'INTC',
    'Taiwan Semiconductor Manufacturing Company Limited': 'TSM',
    'QUALCOMM Incorporated': 'QCOM',
    'Texas Instruments Incorporated': 'TXN',
    'Micron Technology': 'MU',
    'ASML Holding': 'ASML',
    'Tesla': 'TSLA',
    'General Motors Company': 'GM',
    'Ford Motor Company': 'F',
    'Toyota Motor Corporation': 'TM',
    'Ferrari': 'RACE',
    'Honda Motor': 'HMC',
    'Nissan Motor': 'NSANY',
    'Volkswagen': 'VWAGY',
    'JPMorgan Chase': 'JPM',
    'Bank of America Corporation': 'BAC',
    'Wells Fargo & Company': 'WFC',
    'Citigroup': 'C',
    'The Goldman Sachs': 'GS',
    'Morgan Stanley': 'MS',
    'American Express Company': 'AXP',
    'Visa': 'V',
    'Mastercard Incorporated': 'MA',
    'PayPal Holdings': 'PYPL',
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google': 'GOOGL',
    'Amazon': 'AMZN',
    'Meta Platforms': 'META',
    'Cisco Systems': 'CSCO',
    'Oracle Corporation': 'ORCL' ,
    'Bancorp': 'USB',
    'PNC':'PNC' ,
    'State Street Corporation': 'STT' ,
    'Zions Bancorporation': 'ZION' ,
    'Regions Financial Corporation': 'RF' ,
    'Citizens Financial Group': 'CFG'
}

for key in company_dict:
    print(key)

company_name=input('Choose a company from the above companies:')
company=company_dict[company_name]
if(company in companies1):
    predict_stock_price(company,data1)
elif(company in companies2):
    predict_stock_price(company,data2)
elif(company in companies3):
    predict_stock_price(company,data3)
elif(company in companies4):
    predict_stock_price(company,data4)
elif(company in companies5):
    predict_stock_price(company,data5)
elif(company in companies6):
    predict_stock_price(company,data6)
else:
    print("company is not present in dictionary")