# Main File
from ProcessDataset import getDataset
from keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import *
import os.path
import tensorflow_probability as tfp
import math



def customLoss(x, pair):
    tfd = tfp.distributions
    dist = tfd.Normal(loc=pair[0], scale=pair[1])
    loss = tf.reduce_mean(-dist.log_prob(x))
    return loss

    # loss = -tf.math.log(pair[1])
    # loss -= 0.5 * tf.math.log(2*math.pi)
    # loss -= 0.5 * tf.pow(((x - pair[0])/(pair[1])),2)
    # return tf.math.reduce_mean(tf.math.log(loss))

    # if pair[1] == 0:
    #     return 0.0

    # loss = -tf.math.log(pair[1])
    # loss -= 0.5 * tf.math.log(2*math.pi)
    # loss -= 0.5 * tf.pow(((x - pair[0])/(pair[1])),2)
    # return tf.math.reduce_mean(-tf.math.log(loss))

    # loss = (1/tf.math.sqrt(2*math.pi*pair[1]*pair[1]))
    # loss += (x - pair[0])*(x - pair[0])/(2*pair[1]*pair[1])
    # return tf.math.reduce_mean(tf.math.log(loss))


df, x, y = getDataset()
x = df['x']
x = np.asarray(x).astype('float32')

y = []
for idx, row in enumerate(df['y']):
    # temp = np.zeros(500)
    temp = []
    for tag in row:
        temp.append(tag)
    y.append(temp)

y = np.array(y)
# print(y.shape)

y_train, y_test, x_train, x_test = train_test_split(y, x, test_size=0.2, random_state=42)
# print(y_test[-1])

# define the neural network model

if os.path.isfile("./Test_Model/saved_model.pb"):
    model=tf.keras.models.load_model("Test_Model")
else:
    model = keras.Sequential(
    [
        keras.layers.Dense(500, activation='relu', input_shape=(500,)),
        keras.layers.Dense(250, activation='relu'),
        #keras.layers.Dense(1, activation='linear')
        keras.layers.Dense(1,activation=lambda x: tf.nn.elu(x) + 1)
    ])
    #mu = tf.layers.dense(inputs=layer, units=1)
    #sigma = tf.layers.dense(inputs=layer, units=1,activation=lambda x: tf.nn.elu(x) + 1)

    inp = Input((500,))
    x = Dense(500, activation='relu')(inp)
    x = Dense(250, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    mu = Dense(1, activation='linear')(x)
    sigma = Dense(1,activation=lambda x: tf.nn.elu(x) + 1)(x)

    model = Model(inp, [mu, sigma])


    # compile the model
    model.compile(optimizer='adam', loss=customLoss)

    # train the model
    model.fit(y_train, x_train, epochs=10, batch_size=32, validation_data=(y_test, x_test))

    model.save("Test_Model")



# evaluate the model on the testing set
results = model.evaluate(y_test, x_test)

predictresutl,predictresut2 = model.predict(y_test)

print('Test loss:', results)
print('Test predict:', np.any(predictresutl < 0))
print('Test predict:', np.asarray(predictresutl))
print('Test predict:', np.asarray(predictresut2))


# accuracy = keras.metrics.accuracy_score(x_test, predictresutl)
