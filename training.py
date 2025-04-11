import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

def create_sequences(data, time_window, target_col):
    X, y = [], []
    for i in range(time_window, len(data)):
        X.append(data[i-time_window:i])
        y.append(data[i, target_col])
    return np.array(X), np.array(y)

def create_scaled_sequences(data, time_window, target_col):
    X_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(time_window, len(data)):
        mu_list.append(np.mean(data[i-time_window:i, target_col]))
        std_list.append(np.std(data[i-time_window:i, target_col]))
        X_scaled.append((data[i-time_window:i]-mu_list[i-time_window])/std_list[i-time_window])
        y.append(data[i, target_col])
    return np.array(X_scaled), np.array(y), mu_list, std_list

# Load stock data
filename = 'aapl'
data = pd.read_csv(f'{filename}_Stock_Data_Full.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.dropna(inplace=True)
data_basic = data[['Date', 'Adj Close', 'Volume']]
data_basic.rename(columns={'Adj Close': 'Close'}, inplace=True)
data_sentiment = data[['Date', 'Adj Close', 'Volume', 'compoundNYTimes']]
data_sentiment.rename(columns={'Adj Close': 'Close', 'compoundNYTimes': 'sentiment'}, inplace=True)

# Split into training and testing sets
train_basic, test_basic = train_test_split(data_basic, train_size=0.8, test_size=0.2, shuffle=False)
train_sentiment, test_sentiment = train_test_split(data_sentiment, train_size=0.8, test_size=0.2, shuffle=False)

# Define model parameters
time_window = 5
batch_size = 32
epochs = 50
def build_basic_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def build_fine_tuned_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    return model
def train_evaluate_model(model, X_train, y_train, X_test, y_test, mu_test, std_test, actual_values):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              shuffle=False,
              callbacks=[early_stopping],
              verbose=0)
    
    predictions = (model.predict(X_test) * np.array(std_test).reshape(-1,1)) + np.array(mu_test).reshape(-1,1)
    
    rmse = math.sqrt(mean_squared_error(actual_values, predictions))
    mape = mean_absolute_percentage_error(actual_values, predictions)
    
    return predictions, rmse, mape
scaler_basic = StandardScaler()
scaled_train_basic = scaler_basic.fit_transform(np.array(train_basic[['Close', 'Volume']]).reshape(-1, 2))

X_train_basic, y_train_basic = create_sequences(scaled_train_basic, time_window, 0)
X_test_basic, y_test_basic, mu_test_basic, std_test_basic = create_scaled_sequences(
    np.array(test_basic[['Close', 'Volume']]).reshape(-1, 2), time_window, 0)
scaler_sentiment = StandardScaler()
scaled_train_sentiment = scaler_sentiment.fit_transform(np.array(train_sentiment[['Close', 'Volume', 'sentiment']]).reshape(-1, 3))

X_train_sentiment, y_train_sentiment = create_sequences(scaled_train_sentiment, time_window, 0)
X_test_sentiment, y_test_sentiment, mu_test_sentiment, std_test_sentiment = create_scaled_sequences(
    np.array(test_sentiment[['Close', 'Volume', 'sentiment']]).reshape(-1, 3), time_window, 0)

# Train and evaluate basic LSTM
basic_lstm = build_basic_lstm((X_train_basic.shape[1], X_train_basic.shape[2]))
basic_preds, basic_rmse, basic_mape = train_evaluate_model(
    basic_lstm, X_train_basic, y_train_basic, X_test_basic, y_test_basic, 
    mu_test_basic, std_test_basic, test_basic['Close'][time_window:])

# Train and evaluate fine-tuned LSTM
fine_tuned_lstm = build_fine_tuned_lstm((X_train_basic.shape[1], X_train_basic.shape[2]))
fine_tuned_preds, fine_tuned_rmse, fine_tuned_mape = train_evaluate_model(
    fine_tuned_lstm, X_train_basic, y_train_basic, X_test_basic, y_test_basic, 
    mu_test_basic, std_test_basic, test_basic['Close'][time_window:])

# Train and evaluate basic LSTM with sentiment
basic_sentiment_lstm = build_basic_lstm((X_train_sentiment.shape[1], X_train_sentiment.shape[2]))
basic_sentiment_preds, basic_sentiment_rmse, basic_sentiment_mape = train_evaluate_model(
    basic_sentiment_lstm, X_train_sentiment, y_train_sentiment, X_test_sentiment, y_test_sentiment, 
    mu_test_sentiment, std_test_sentiment, test_sentiment['Close'][time_window:])

# Train and evaluate fine-tuned LSTM with sentiment
fine_tuned_sentiment_lstm = build_fine_tuned_lstm((X_train_sentiment.shape[1], X_train_sentiment.shape[2]))
fine_tuned_sentiment_preds, fine_tuned_sentiment_rmse, fine_tuned_sentiment_mape = train_evaluate_model(
    fine_tuned_sentiment_lstm, X_train_sentiment, y_train_sentiment, X_test_sentiment, y_test_sentiment, 
    mu_test_sentiment, std_test_sentiment, test_sentiment['Close'][time_window:])

results_df = pd.DataFrame({
    'Date': test_basic['Date'][time_window:],
    'Actual': test_basic['Close'][time_window:],
    'Basic LSTM': basic_preds.reshape(-1),
    'Fine-tuned LSTM': fine_tuned_preds.reshape(-1),
    'Basic LSTM w/Sentiment': basic_sentiment_preds.reshape(-1),
    'Fine-tuned LSTM w/Sentiment': fine_tuned_sentiment_preds.reshape(-1)
})

# Print model performance metrics
print("Model Performance Metrics:")
print(f"Basic LSTM - RMSE: {basic_rmse:.4f}, MAPE: {basic_mape:.4f}%")
print(f"Fine-tuned LSTM - RMSE: {fine_tuned_rmse:.4f}, MAPE: {fine_tuned_mape:.4f}%")
print(f"Basic LSTM w/Sentiment - RMSE: {basic_sentiment_rmse:.4f}, MAPE: {basic_sentiment_mape:.4f}%")
print(f"Fine-tuned LSTM w/Sentiment - RMSE: {fine_tuned_sentiment_rmse:.4f}, MAPE: {fine_tuned_sentiment_mape:.4f}%")

plt.figure(figsize=(14, 7))
plt.plot(results_df['Date'], results_df['Actual'], 'k-', label='Actual Price')
plt.plot(results_df['Date'], results_df['Basic LSTM'], 'b--', label='Basic LSTM')
plt.plot(results_df['Date'], results_df['Fine-tuned LSTM'], 'r--', label='Fine-tuned LSTM')
plt.plot(results_df['Date'], results_df['Basic LSTM w/Sentiment'], 'g--', label='Basic LSTM w/Sentiment')
plt.plot(results_df['Date'], results_df['Fine-tuned LSTM w/Sentiment'], 'm--', label='Fine-tuned LSTM w/Sentiment')
plt.title('Stock Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create bar chart of model performance
models = ['Basic LSTM', 'Fine-tuned LSTM', 'Basic LSTM w/Sentiment', 'Fine-tuned LSTM w/Sentiment']
rmse_values = [basic_rmse, fine_tuned_rmse, basic_sentiment_rmse, fine_tuned_sentiment_rmse]
mape_values = [basic_mape, fine_tuned_mape, basic_sentiment_mape, fine_tuned_sentiment_mape]

plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(models))

plt.bar(index, rmse_values, bar_width, label='RMSE')
plt.bar(index + bar_width, mape_values, bar_width, label='MAPE (%)')

plt.xlabel('Model')
plt.ylabel('Error')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width/2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()