# Main File
from ProcessDataset import getDataset
from keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import *
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df, x, y = getDataset()

# print(df.shape)
# print(x.shape)
# print(y.shape)
# print(df.head())

# y are features, x are outputs
# y = df['y']
x = df['x']
x = np.asarray(x).astype('int')
# y = y.tolist()


y = []
for idx, row in enumerate(df['y']):
    # temp = np.zeros(500)
    temp = []
    for tag in row:
        temp.append(tag)
    y.append(temp)

y = np.array(y).astype('int')
# print(y.shape)

y_train, y_test, x_train, x_test = train_test_split(y, x, test_size=0.2, random_state=42)
print(y_train)

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
    model.compile(optimizer='adam', loss='mean_squared_error')

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

(x_test, predictresutl)




#  bayes

mnb = MultinomialNB(fit_prior=True)

mnb.fit(y_train, x_train)

predictY = mnb.predict(y_test)
print("bayes: ")
accuracy = metrics.accuracy_score(x_test,predictY)
print(accuracy)