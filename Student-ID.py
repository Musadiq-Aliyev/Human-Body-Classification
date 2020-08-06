import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
%matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
import pickle, datetime

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import LSTM, Input,Activation

from keras.optimizers import RMSprop, SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Import the backend
from keras import backend as K

import os

#Loading Images
img_dir='DS'
train_body_images = []
train_body_labels = []

for img_path in glob.glob(os.path.join(img_dir,'*.jpeg')):

    body_label = img_path.split("\\")[-1]

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    train_body_images.append(img)
    train_body_labels.append(body_label[0:body_label.find('#')])
train_body_images = np.array(train_body_images)
train_body_labels = np.array(train_body_labels)


#Preprocessing

label_to_num = {v:i for i,v in enumerate(np.unique(train_body_labels))}
num_to_label = {v: k for k, v in label_to_num.items()}


train_label_num = np.array([label_to_num[x] for x in train_body_labels])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_body_images, train_label_num, test_size = 0.2, shuffle=True, random_state=2)

n_class=len(label_to_num)

#Normalization of the images and one-hot encoding of the labels

x_train_normalized = np.array(x_train / 255.0 - 0.5 )
x_test_normalized = np.array(x_test / 255.0 - 0.5 )

label_binarizer = LabelBinarizer()
y_train_hot = label_binarizer.fit_transform(y_train)
y_test_hot = label_binarizer.fit_transform(y_test)

#CNN Model

def cnn_model(input_shape,nb_classes):

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('tanh'))

    return model

cnn_model = cnn_model((32,32,3),n_class)
cnn_model.summary()

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#Training the CNN model with the normalized image data and labels
cnn_model.fit(x_train_normalized, y_train_hot, batch_size=15, epochs=5,verbose=1, validation_data=[x_test_normalized,y_test_hot])

layer_name = 'dense_1'
FC_layer_model = Model(inputs=cnn_model.input,outputs=cnn_model.get_layer(layer_name).output)

#Feature Extraction
i=0
features=np.zeros(shape=(1500,1024))

for img_path in glob.glob(os.path.join(img_dir,'*.jpeg')):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)
    FC_output = FC_layer_model.predict(img)
    features[i]=FC_output
    i+=1
    if i==1500:
        break

feature_names=[]
for i in range(1024):
    feature_names.append("col_"+str(i))
    i+=1

#Create DataFrame with features and coloumn name
train_features=pd.DataFrame(data=features,columns=feature_names)
feature_names = np.array(feature_names)

from sklearn.model_selection import train_test_split
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(train_features, train_label_num, test_size = 0.2, shuffle=True, random_state=2)

#KNN Model

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_1, y_train_1)

y_pred = classifier.predict(x_test_1)

from sklearn.externals import joblib
# Save the model as a pickle in a file
joblib.dump(classifier, 'student-id-Knn.pkl')

#Evaluating Model
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_1, y_pred))
print(classification_report(y_test_1, y_pred))

accuracy=accuracy_score(y_test_1,y_pred)
print('KNN- Accuracy:', accuracy*100, '%.')

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_1, y_train_1)
    pred_i = knn.predict(x_test_1)
    error.append(np.mean(pred_i != y_test_1))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#Random Forest Model

model_params={
# Number of trees in random forest
'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
# Number of features to consider at every split
'max_features' : ['auto', 'sqrt'],
# Minimum number of samples required to split a node
'min_samples_split' : [2, 5, 10],
# Minimum number of samples required at each leaf node
'min_samples_leaf' : [1, 2, 4] }

# creating random forest model
rf_model = RandomForestClassifier()

# consturct random search
clf = RandomizedSearchCV(rf_model, model_params, n_iter=2, cv=3, random_state=1)

model = clf.fit(x_train_1, y_train_1)

# printing best set of hyperparameters
from pprint import pprint
pprint(model.best_estimator_.get_params())

from sklearn.externals import joblib
# Save the model as a pickle in a file
joblib.dump(model, 'student-id-Rf.pkl')

# predictions using the best-performing model
predictions = model.predict(x_test_1)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_1, predictions))
print(classification_report(y_test_1, predictions))

accuracy=accuracy_score(y_test_1,predictions)
print('RF- Accuracy:', accuracy*100, '%.')
