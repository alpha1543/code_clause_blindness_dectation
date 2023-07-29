! kaggle compitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip
import cv2;
import matplotlib.pyplot as plt import numpy as np;
import pandas as pd
from random import shuffle; import cv2;
from random import shuffle; from tqdm import tqdm; import tensorflow;
from keras import layers; from keras import Model;
from keras.optimizers import SGD;
from keras.callbacks import TensorBoard;
IMAGE_SIZE=300;

pip install keras
from keres.optimizers import Adam, SGD, RMSprop

﻿from pydrive.auth import GoogleAuth 
from pydrive.drive import GoogleDrive 
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth= GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default() 
drive GoogleDrive(gauth)
train_Data_X_File = drive.CreateFile({'id':'113Rvy-hV3sgEzjJVMnAtieeaAsnjc1Z1'});
train_Data_Y_File = drive.CreateFile({'id':'1_flYVBAJCp-VP0lYWKAEH3Ngk560q9w'});

﻿
train_Data_X_File.GetContentFile('train_Data_x.npy'); 
train_Data_X = np.load('train_Data_x.npy', allow pickle=True)

train_Data_Y_File.GetContentFile('train_Data_Y.npy');
train_Data_Y=np.load('train_Data_Y.npy',allow_pick=True)

train_Data_X.shape

count = 0;
Num_of_Images= 20; 
plt.figure(figsize=(20,20))
label = ['No DR', 'Mild', 'Moderate', 'Severe", "Proliferative DR'];
for i in np.random.randint(1000, size Num_of_Images):
count
count+1;
plt.subplot (Num_of_Images/4,4, count);
plt.imshow(np.reshape(train_Data_x[i], (IMAGE_SIZE, IMAGE_SIZE,3)));
plt.title(label[int (train_Data_v[i])]);

﻿

num_of_Rows = train_Data_x.shape[0];
num_of_columns =train_Data_x.shape[1];
training_X=train_Data_X[:int(np.round(num_of_Rows* 0.8))]
testing X train_Data_X[int (np.round(num_of_Rows *0.8)):] 
training Y train_Data_Y[:int(np.round(num_of_Rows*0.8))] 
testing Y train_Data_Y[int (np.round(num_of_Rows *0.8)):]

img_input layers. Input (shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x= layers.Conv2D(16, 3, activation='relu") (img_input) 
x = layers.MaxPooling20(2)(x)

x = layers.Conv2D(32, 3, activation='relu")(x) 
x = layers.MaxPooling20(2)(x)

x= layers.Conv2D(64, 3, activation='relu")(x) 
x = layers.MaxPooling2D(2)(x)

x= layers.Conv2D(128, 3, activation='relu")(x) 
x = layers.MaxPooling2D(2)(x)

x=layer.Flatten()(x)
x=layer.Dense(512,activation'relu')(x)

output=layers.Dense(5,activation='softmax')(x)

Model=Model(img_input,output)
Model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adas(lr=0.00005),matrices=['acc']);

x_train = np.array([i[e] for i in tqdm(training_X)]); 
Y_train = np.array([i[e] for i in training Y]);
X_test = np.array([i[e] for i in tqdm(testing_X)]); 
Y_test = np.array([i[e] for i in testing Y]);
x_train.shape

Model_fit = model.fit(x_train,Y_train, batch_size =64, epochs = 10, verbose=1, validation_data=(x_test, Y_test))
﻿
Marks model.evaluate(x_test,Y_test, verbose=0) 
print('Test Accuracy percentage:',100 Marks[1],"%") 
print('Test Loss percentage:,100 Marks[0],"%")

import matplotlib.pyplot as plt
count=0;
Num_of_Images=20;
plt.figure(figsize=(20,20))
label=["No DR, Mild', 'Moderate', 'Severe', 'Proliferative DR']; 
for i in np.random.randint(500, size=Num_of_Images):
count=count+1;
plt.subplot (Num_of_Images/4,4, count);
plt.imshow(np.reshape(x_test[i], (IMAGE_SIZE, IMAGE_SIZE,3)));
P = model.predict(x_test[i].reshape(1, IMAGE SIZE, IMAGE_SIZE,3)) # Prediction of testing images
P=np.array(P);
plt.title(label[int(Y_test[i])]);

acc-Model fit.history['acc"] 
val_acc Model fit.history['val_acc"] 
loss Model fit.history['loss'] 
val_loss Model fit.history['val_loss'] 
epochs range(1, len(acc)+1)

plt.figure()
plt.title('Training and Validation Loss")
plt.plot(epochs, loss, 'red', label='Training loss") 
plt.plot(epochs, val_loss, 'blue', label='validation loss")
plt.legend()

plt.figure()
plt.title('Training and Validation accuracy")
plt.plot(epochs, loss, 'green', label='Training acc') 
plt.plot(epochs, val loss, 'yellow', label='validation acc')

plt.legend()
plt.show()
