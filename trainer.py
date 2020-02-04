import tensorflow as tf
import numpy as np
import pickle

X = pickle.load(open("X.pickle", "rb"))
y_train = pickle.load(open("y.pickle", "rb"))

x_train=X/255.0
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train, y_train,epochs=150, verbose=2)

model.save('num_reader.model')

new_model = tf.keras.models.load_model('num_reader.model')


predictions = model.predict(x_train[:20])

print(np.argmax(predictions[6]))

predictions = new_model.predict(x_train[:20])

print(np.argmax(predictions[6]))
