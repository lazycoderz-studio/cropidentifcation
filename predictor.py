import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import os

#X = pickle.load(open("X.pickle", "rb"))
#y_train = pickle.load(open("y.pickle", "rb"))

#x_train=X/255.0
categories = ["mustard","pea","wheat","paddy","potato","tomato","maize"]
image="G:/dataset/test"
ar=[]
img_size=50
for img in os.listdir(image):
    print(img)
    img_array=cv2.imread(os.path.join(image,img))

    im = cv2.resize(img_array,(50,50))
    ar.append(im)
ar=np.array(ar).reshape(-1,img_size, img_size,3)
ar=ar/255.0
#print(ar.shape)
new_model = tf.keras.models.load_model('num_reader.model')

'''print(y_train[:20])

predictions = new_model.predict(x_train[:20])
for i in range(20):
    
    print(np.argmax(predictions[i]),end=" ")

print()'''
p=new_model.predict(ar)
print(p)

for a in p:
    print(categories[np.argmax(a)])
'''print(categories[np.argmax(p[1])])
print(categories[np.argmax(p[2])])'''

