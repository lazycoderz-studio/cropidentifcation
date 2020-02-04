import cv2
import numpy as np
import matplotlib .pyplot as plt
import os
import cv2
import random
import pickle

directory="G:/dataset"
categories = ["mustard","pea","wheat","paddy","potato","tomato","maize"]
img_size = 50
training_data=[]
for category in categories:
    path = os.path.join(directory, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img))
            new_array= cv2.resize(img_array,(img_size,img_size))
            training_data.append([new_array,class_num])
        except Exception as e:
            pass
        
     
        
random.shuffle(training_data)

X=[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1,img_size, img_size,3)
y=np.array(y)
print(X.shape)
print(y.shape)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
