import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import RMSprop
import joblib


# constants
SEQ_LEN = 30
FUTURE_PERIOD_PREDICT = 30
TO_PREDICT = 'Total Cloud Cover [%]'
INPUT_SIGNALS = ['Global CMP22 (vent/cor) [W/m^2]','Direct sNIP [W/m^2]','Azimuth Angle [degrees]','Tower Dry Bulb Temp [deg C]','Tower Wet Bulb Temp [deg C]','Tower Dew Point Temp [deg C]','Tower RH [%]','Total Cloud Cover [%]','Peak Wind Speed @ 6ft [m/s]','Avg Wind Direction @ 6ft [deg from N]','Station Pressure [mBar]','Precipitation (Accumulated) [mm]','Snow Depth [cm]','Moisture','Albedo (CMP11)']

df = pd.read_csv('./train/train.csv')
main_df = pd.DataFrame()
df['future'] = df[TO_PREDICT].shift(-FUTURE_PERIOD_PREDICT)

# getting input signals and output signals
x_data = df[INPUT_SIGNALS].values[0:-FUTURE_PERIOD_PREDICT]
y_data = df['future'].values[:-FUTURE_PERIOD_PREDICT]
y_data = y_data.reshape(-1, 1)

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
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# exporting the scalers
joblib.dump(x_scaler, './scaler/x_scaler.sav')
joblib.dump(y_scaler, './scaler/y_scaler.sav')

# generating batches
batch_size = 64
sequence_length = SEQ_LEN

# declaring shapes of input and output signals
num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

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

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


# model
model = Sequential()


model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='sigmoid'))

learning_rate = 0.001
optimizer = RMSprop(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
model.fit(x=generator,
          epochs=20,
          steps_per_epoch=100,
          validation_data=validation_data)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

model.save(f'model/NEW-model-batch_size-{batch_size}-learning_rate-{learning_rate}.h5')