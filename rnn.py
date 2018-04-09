# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#model creation
def compile_model(model_type,X_train,y_train):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(model_type(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(model_type(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(model_type(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(model_type(units = 50))
    regressor.add(Dropout(0.2))
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 700, batch_size = 32)
    return regressor


# Importing the training set
data= pd.read_csv('Suicides in India 2001-2012.csv')
data=(data.where(data.Type_code=='Causes').groupby(['Year'], )).Total.sum()/12
c=data
for i in range(11):
  data=data.append(c)
c=(data).astype(np.int64)
data=c.to_frame(name=None)
data['Year'] = (c.index).astype(np.int64)
data=data.reset_index(drop=True)
data.Total
data=data.sort_values(['Year']) 
dates = pd.date_range(start='2001-01-01', freq='MS', periods=len(data))
import calendar
data['Month'] = dates.month
data['Month'] = data['Month'].apply(lambda x: calendar.month_abbr[x])
data['Year'] = dates.year
data = data[['Month', 'Year', 'Total']]
data.set_index(dates, inplace=True)
sales_ts = data['Total']
plt.figure(figsize=(10, 5))
plt.plot(sales_ts)
plt.xlabel('Years')
plt.ylabel('Total')
plt.clf()    
training_set = data.iloc[:, 2:3].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(12, len(training_set)-36):
    X_train.append(training_set_scaled[i-12:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
lstm_regressor=compile_model(LSTM,X_train,y_train)
#gru_regressor = compile_model(GRU,X_train,y_train)
#simpleRNNCell_regressor = compile_model(SimpleRNN,X_train,y_train)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
real_ = training_set[:,:]
predicted_ = training_set[:-36,:]
# Getting the predicted stock price of 2017
inputs = data.iloc[:, 2:3].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(len(training_set)-36, len(training_set)):
    X_test.append(inputs[i-12:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_lstm = lstm_regressor.predict(X_test)
#predicted_gru = gru_regressor.predict(X_test)
#predicted_simple = simpleRNNCell_regressor.predict(X_test)
predicted_lstm = sc.inverse_transform(predicted_lstm)
#predicted_gru = sc.inverse_transform(predicted_gru)
#predicted_simple = sc.inverse_transform(predicted_simple)
predicted_lstm = predicted_lstm.reshape(len(predicted_lstm))
#predicted_gru = predicted_gru.reshape(len(predicted_gru))
#predicted_simple = predicted_simple.reshape(len(predicted_simple))
result_lstm = (pd.Series(predicted_lstm))
#result_gru = pd.Series(predicted_gru)
#result_simple =(pd.Series(predicted_simple))
dates = pd.Series.to_frame(pd.Series((i) for i in range(len(result_lstm))))
dates[0]
result_lstm = pd.Series.to_frame(result_lstm)
result_lstm.set_index(dates[0], inplace=True)
#result_gru = pd.Series.to_frame(result_gru)
#result_gru.set_index(dates[0], inplace=True)
#result_simple = pd.Series.to_frame(result_simple)
#result_simple.set_index(dates[0], inplace=True)
# Visualising the results
plt.plot(predicted_[-36:], color = 'Red', label = 'Total')
plt.plot(result_lstm, color = 'green', label = 'lstm')
#plt.plot(result_simple, color = 'orange', label = 'simple')
#plt.plot(result_gru, color = 'yellow', label = 'gru')
plt.title('next year forecast')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
#plt.show()
plt.savefig("saw.png")




