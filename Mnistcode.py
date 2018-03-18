import pandas as pd
k=pd.read_csv('train.csv')
k1=pd.read_csv('test.csv')
import numpy as np
import matplotlib.pyplot as plt

k=cv2.imread('Untitled.jpg')
cv2.imshow("Images",k)
cv2.waitKey(0)

gray_image=cv2.cvtColor(k,cv2.COLOR_BGR2GRAY)
gray_image=gray_image.reshape(28,28,1)
gray_image=gray_image/255.0
gray_image=np.array([gray_image])



X_train=k.drop(labels=["label"],axis=1)
y_train=k["label"]

X_test=k1


X_train=X_train/255.0
X_test=X_test/255.0

X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
y_train=to_categorical(y_train, num_classes=10)

from sklearn.cross_validation import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=0)


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier=Sequential()

classifier.add(Convolution2D(32,4,4,input_shape=(28,28,1),activation='relu'))
classifier.add(Convolution2D(32,4,4,activation='relu'))
classifier.add(MaxPooling2D(2,2))


classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(Convolution2D(64,4,4,activation='relu'))
classifier.add(MaxPooling2D(2,2))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(256,activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(10,activation='softmax'))


classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True) 
train_datagen.fit(X_train)
 

history = classifier.fit_generator(train_datagen.flow(X_train,y_train, batch_size=86),
                                   epochs = 1, validation_data = (X_val,y_val),
                                   verbose = 2, steps_per_epoch=X_train.shape[0] // 86)

# Predict the values from the validation dataset
y_pred = classifier.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
accuracy_score(y_true,y_pred_classes)

z_pred=classifier.predict(gray_image)
z=np.argmax(z_pred)
