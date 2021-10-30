import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, GRU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.backend import square, mean
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt


# constants
SEQ_LEN = 124 # change in future
FUTURE_PERIOD_PREDICT = 30
TO_PREDICT = 'Total Cloud Cover [%]'


# note to self: future and target would be the same in my case

df = pd.read_csv('./data/train/train/train.csv')
# df.set_index('Time [Mins]', inplace=True) #setting index to time series THIS might make the machine go wonky so i will test it later
main_df = pd.DataFrame()
# df = df[[TO_PREDICT]] # might change in future
df['future'] = df[TO_PREDICT].shift(-FUTURE_PERIOD_PREDICT)

# getting input signals and output signals
x_data = df[TO_PREDICT].values[0:-FUTURE_PERIOD_PREDICT]
y_data = df['future'].values[:-FUTURE_PERIOD_PREDICT]
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)
print(type(x_data))
print("Shape:", x_data.shape)
print(type(y_data))
print("Shape:", y_data.shape)

#splitting the data into train and test
num_data = len(x_data)
train_split = 0.9 
num_train = int(train_split * num_data)
num_test = num_data - num_train 
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
y_train = y_data[0:num_train]
y_test = y_data[num_train:]

#scaling the data
x_scaler = MinMaxScaler()
# x_train = x_train.reshape(-1, 1) # might change in future
# x_test = x_test.reshape(-1, 1) # might change in future
# # x_data = x_test.reshape(-1, 1) # might change
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
# print(y_train)
# print(y_train.shape)
# y_train = y_train.reshape(-1, 1) # might change in future
# y_test = y_test.reshape(-1, 1) # might change in future
# # y_data = y_test.reshape(-1, 1) # might change
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# generating batches
batch_size = 128    
sequence_length = 30

# declaring shapes of input and output signals
num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]
print('.............................')
print(num_x_signals)
print(num_y_signals)

# copied code from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

x_batch, y_batch = next(generator)
print(x_batch.shape)
print(y_batch.shape)
# x_batch = x_batch.reshape(-1, 1)
# y_batch = y_batch.reshape(-1, 1)
# num_x_signals = x_batch.shape[1]
# num_y_signals = y_batch.shape[1]

print(x_batch.shape)
print(y_batch.shape)
print(num_x_signals)
# print(num_x_signals.shape)
print(x_data.shape[1])

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


# model
model = Sequential()


model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='sigmoid'))

# from tensorflow.python.keras.initializers import RandomUniform

# # Maybe use lower init-ranges.
# init = RandomUniform(minval=-0.01, maxval=0.01)

# model.add(Dense(num_y_signals,
#                 activation='linear',
#                 kernel_initializer=init))

warmup_steps = 50
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculat the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))
    
    return mse

optimizer = RMSprop(lr=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
model.fit(x=generator,
          epochs=20,
          steps_per_epoch=100,
          validation_data=validation_data)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(TO_PREDICT)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(TO_PREDICT[signal])
        plt.legend()
        plt.show()

plot_comparison(start_idx=500, length=130, train=True)