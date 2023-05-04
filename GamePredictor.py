# Main File
from ProcessDataset import getDataset
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

df, x, y = getDataset()

# print(df.shape)
# print(x.shape)
# print(y.shape)
# print(df.head())

# y are features, x are outputs
y = df['y']
x = df['x']
x = np.asarray(x).astype('float32')
y = y.tolist()
# print(y.shape)
# print(y.head())

y_train, y_test, x_train, x_test = train_test_split(y, x, test_size=0.2, random_state=42)


# define the neural network model
model = keras.Sequential([
    keras.layers.Dense(500, activation='relu', input_shape=(500,)),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(y_train, x_train, epochs=50, batch_size=32, validation_data=(y_test, x_test))

# evaluate the model on the testing set
loss = model.evaluate(y_test, x_test)
print('Test loss:', loss)
