# coding: utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import json
import os
from IPython.display import display, clear_output
import copy
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('~/GitData/EmotionalRecChallenge/train.csv', index_col = 'id')
df_test = pd.read_csv('~/GitData/EmotionalRecChallenge/test.csv', index_col = 'id')
df_train.columns, df_test.columns
df_train.isnull().sum()
from tqdm import tqdm_notebook as tqdm
import matplotlib
import matplotlib.cm as cm
from keras.utils import np_utils
df_pixels = df_train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
X_train = np.vstack(df_pixels.values)
y_train = np.array(df_train["target"])
X_train.shape, y_train.shape
X_train, X_test, y_train_c, y_test_c = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
X_train.shape, X_test.shape
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
X_train.shape, X_test.shape
y_train = np_utils.to_categorical(y_train_c)
y_test = np_utils.to_categorical(y_test_c)
y_train.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
df_pixels_final = df_test['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
X_final = np.vstack(df_pixels_final.values)
X_final.shape
X_final = X_final.reshape(-1, 48, 48, 1)
X_final.shape
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.regularizers import l2

from keras.models import load_model
from keras.models import model_from_json

from IPython.display import clear_output
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
# Model 2 plot and write history
hist2 = TrainingPlot()
num_features = 256
num_labels = 7
width, height = 48, 48

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
json_file = open('newCNN_14.06.2019/newCNN_14.06.2019_model2.json', 'r')
# load model
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights
model2.load_weights("newCNN_14.06.2019/newCNN_14.06.2019_model2.h5")
# load history
json_hist = open('newCNN_14.06.2019/newCNN_14.06.2019_model2_hist.json', 'r')
loaded_hist_json = json_hist.read()
json_hist.close()
hist2.history = json.loads(loaded_hist_json)
        
print("Loaded model, weight and history from disk")
def plot_history(hist):
   plt.figure(figsize=(28,6))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.show()
plot_history(hist2)    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_features = 256
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Flatten())

model2.add(Dense(2*2*2*num_features, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(2*2*num_features, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(2*num_features, activation='relu'))
model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
def plot_history(hist):
   plt.figure(figsize=(28,6))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.show()
plot_history(hist2)    
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 64
epochs = 10
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist):
   plt.figure(figsize=(28,6))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.show()
plot_history(hist2)    
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_14.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights('SmallerCNN_14.06.2019/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open('SmallerCNN_14.06.2019/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open('SmallerCNN_14.06.2019/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'newCNN_14.06.2019/'model2_file_name 1-999999")
clear_output()
print('All files saved')
len(hist2.history['val_loss'])
def plot_history(hist):
   plt.figure(figsize=(28,6))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.show()
plot_history(hist2)    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.4))
model2.add(Dense(14, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 10
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
batch_size = 128
epochs = 50
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.4))
model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist):
   plt.figure(figsize=(28,6))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.show()
plot_history(hist2)    
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min'+ str(min_value_val_loss)+' at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max'+ str(max_value_val_acc) + ' at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
model2_file_name = 'SmallerCNN_0.53(test)_18.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f model2_file_name'/'model2_file_name 1-999999")
clear_output()
print('All files saved')
model2_file_name = 'SmallerCNN_18.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f str(model2_file_name)+'/'model2_file_name 1-999999")
clear_output()
print('All files saved')
model2_file_name = 'SmallerCNN_0.53(test)_18.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f model2_file_name'/'model2_file_name 1-999999")
clear_output()
print('All files saved')
model2_file_name = 'SmallerCNN_0.53(test)_18.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_0.53(test)_18.06.2019/'model2_file_name 1-999999")
clear_output()
print('All files saved')
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
"""
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 32
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min'+ str(round(min_value_val_loss,2))+' at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max'+ str(round(max_value_val_acc, 2)) + ' at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
# Model 1 plot and write history
hist1 = TrainingPlot()
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min '+ str(round(min_value_val_loss,2))+' at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max '+ str(round(max_value_val_acc, 2)) + ' at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 32
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 16
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist):
   plt.figure(figsize=(28,6))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.show()
plot_history(hist2)    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
model2.add(Dense(14, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
"""
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.show()
hist = copy.deepcopy(hist2)
plot_history(hist2, hist)    
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.show()

plot_history(hist2, hist)    
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.show()

plot_history(hist2, hist)    
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.show()

plot_history(hist2, hist)    
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_0.50(test)_19.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_0.50(test)_19.06.2019/'model2_file_name 1-999999")
clear_output()
print('All files saved')
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_0.50(test)_19.06.2019'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_0.50(test)_19.06.2019/'model2_file_name 1-999999")
clear_output()
print('All files saved')
batch_size = 64
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
data_final = np.argmax(model2.predict(X_final), axis=1)
data_final
final_dataframe = pd.DataFrame(data_final.T, index = df_test.index)
final_dataframe.columns = ['target']
final_dataframe.head(1)
final_dataframe.to_csv(model2_file_name + '/' +'EmotionalRecognition_19.06.2019_02:15.csv', index='False')
final_dataframe.head()
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.53(public)'
#create folder
newpath = model2_file_name 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_0.50(test)_19.06.2019/0.53(public)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.53(public)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_0.50(test)_19.06.2019/0.53(public)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.53(public)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_19.06.2019/0.53(public)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
batch_size = 64
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.53(public)(overfiting)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_19.06.2019/0.53(public)(overfitting)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.53(public)(overfitting)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_19.06.2019/0.53(public)(overfitting)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
hist = copy.deepcopy(hist2)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())

model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 32
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 16
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 1024
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
batch_size = 512
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
batch_size = 512
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
model.summary()
batch_size = 512
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.54(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_19.06.2019/0.54(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
batch_size = 512
epochs = 300
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 300
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
data_final = np.argmax(model2.predict(X_final), axis=1)
data_final
final_dataframe = pd.DataFrame(data_final.T, index = df_test.index)
final_dataframe.columns = ['target']
final_dataframe.head(1)
final_dataframe.to_csv(model2_file_name + '/' +'EmotionalRecognition_20.06.2019_02:15.csv', index='False')
final_dataframe.head()
model2_file_name = 'SmallerCNN_19.06.2019'
model2_score = '0.562(public)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_19.06.2019/0.562(public)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
batch_size = 512
epochs = 300
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
data_final = np.argmax(model2.predict(X_final), axis=1)
data_final
final_dataframe = pd.DataFrame(data_final.T, index = df_test.index)
final_dataframe.columns = ['target']
final_dataframe.head(1)
final_dataframe.to_csv(newpath + '/' +'EmotionalRecognition_21.06.2019_10:47.csv', index='False')
final_dataframe.head()
model2_file_name = 'SmallerCNN_21.06.2019'
model2_score = '0.582(public)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_21.06.2019/0.582(public)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
batch_size = 512
epochs = 200
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_21.06.2019'
model2_score = '0.576(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_21.06.2019/0.576(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
batch_size = 512
epochs = 300
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min '+ str(round(min_value_val_loss,2))+' at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max '+ str(round(max_value_val_acc, 2)) + ' at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(N))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
model2_file_name = 'SmallerCNN_21.06.2019'
model2_score = '0.576(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_21.06.2019/0.576(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
data_final = np.argmax(model2.predict(X_final), axis=1)
data_final
final_dataframe = pd.DataFrame(data_final.T, index = df_test.index)
final_dataframe.columns = ['target']
final_dataframe.head(1)
final_dataframe.to_csv(newpath + '/' +'EmotionalRecognition_21.06.2019_22:47.csv', index='False')
final_dataframe.head()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))N

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))N

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
# Model 2 plot and write history
hist2 = TrainingPlot()
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 32
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 16
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
class TrainingPlot(Callback):
    
    def __init__(self, hist_old=None):
        # Initialize the lists for holding the logs, losses and accuracies
        if hist_old is None:
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.logs = []
            self.hist_old = None
            self.history = {}
        else:
            self.hist_old = []
            self.losses = []
            self.acc = []
            self.val_losses = []
            self.val_acc = []
            self.history = {}
            # Append old and new history 
            self.hist_old.append(hist_old)
            self.losses = hist_old.losses[:]
            self.acc = hist_old.acc[:]
            self.val_losses = hist_old.val_losses[:]
            self.val_acc = hist_old.val_acc[:]
        
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        #self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            min_eph_val_loss = np.argmin(self.val_losses)
            max_eph_val_loss = np.argmax(self.val_losses)
            min_value_val_loss = min(self.val_losses)
            max_value_val_loss = max(self.val_losses)
            
            min_eph_val_acc = np.argmin(self.val_acc)
            max_eph_val_acc = np.argmax(self.val_acc)
            min_value_val_acc = min(self.val_acc)
            max_value_val_acc = max(self.val_acc)
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            
            plt.figure(figsize=(20,6))
            plt.style.use("seaborn-dark-palette")
            # Return default style
            #plt.style.use('classic') 
            plt.subplot(1, 2, 1)            
            plt.plot(N, self.losses, label = "train_loss")            
            plt.plot(N, self.val_losses, label = "val_loss")            
            plt.scatter(min_eph_val_loss, min_value_val_loss, color='red', s=40, marker='o',
                        label= 'Min '+ str(round(min_value_val_loss,2))+' at '+str(min_eph_val_loss)+' epoch')
            
            plt.title("Training Loss [Epoch {}]".format(N[-1]))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.grid(linewidth=0.2)
            plt.legend(loc='best')
            
            plt.subplot(1, 2, 2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")                    
            plt.scatter(max_eph_val_acc, max_value_val_acc, color='red', s=40, marker='o',
                        label= 'Max '+ str(round(max_value_val_acc, 2)) + ' at '+str(max_eph_val_acc)+' epoch')
            
            plt.title("Training Accuracy [Epoch {}]".format(N[-1]))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.grid(linewidth=0.2)
            plt.show()   
           
    def on_train_end(self, logs={}):
        self.history = {'acc': self.acc, 'loss': self.losses,
                        'val_acc' : self.val_acc, 'val_loss': self.val_losses}        
batch_size = 512
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
data_final = np.argmax(model2.predict(X_final), axis=1)
data_final
final_dataframe = pd.DataFrame(data_final.T, index = df_test.index)
final_dataframe.columns = ['target']
final_dataframe.head(1)
model2_file_name = 'SmallerCNN_22.06.2019'
model2_score = '0.619(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_22.06.2019/0.619(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
final_dataframe.to_csv(newpath + '/' +'EmotionalRecognition_22.06.2019_01:07.csv', index='False')
final_dataframe.head()
hist = copy.deepcopy(hist2)
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_22.06.2019'
model2_score = '0.606(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_22.06.2019/0.606(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

#model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

#model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 1024
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 32
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 1024
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
# Model 2 plot and write history
hist2 = TrainingPlot()
json_file = open('SmallerCNN_22.06.2019/0.619(test)/SmallerCNN_22.06.2019_model2.json', 'r')
# load model
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights
model2.load_weights("SmallerCNN_22.06.2019/0.619(test)/SmallerCNN_22.06.2019_model2.h5")
# load history
json_hist = open('SmallerCNN_22.06.2019/0.619(test)/SmallerCNN_22.06.2019_model2_hist.json', 'r')
loaded_hist_json = json_hist.read()
json_hist.close()
hist2.history = json.loads(loaded_hist_json)
        
print("Loaded model, weight and history from disk")
def plot_history(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history(hist2, hist)    
def plot_history_val(hist):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()
plot_history(hist) 
def plot_history(hist):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()
plot_history(hist) 
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
model.summary()
model2.summary()
model2.summary()
#model.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.summary()
#model.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model2.summary()
model2.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 32
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 16
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 1024
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Training Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 1024
epochs = 5
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 1024
epochs = 64
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 1024
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 1024
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
def plot_history(hist):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()
plot_history(hist) 
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
batch_size = 1024
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 512
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
X_train[1,1,1]
X_train[1,255,1]
X_train[1,48,48]
X_train[1,24,1]
X_train[1,24,6]
X_train.astype('float32') /255
plt.figure(0, figsize=(12,6))
for i in range(1, 13):
    plt.subplot(3,4,i)
    plt.imshow(X_train[i, :, :, 0], cmap="gray")

plt.tight_layout()
plt.show()
X_train[1,24,6]
X_train[2,24,6]
X_train[2,24,7]
df_pixels = df_train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
X_train = np.vstack(df_pixels.values)
y_train = np.array(df_train["target"])
X_train.shape, y_train.shape
X_train.astype('float32') /255
X_train, X_test, y_train_c, y_test_c = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
X_train.shape, X_test.shape
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
X_train.shape, X_test.shape
y_train = np_utils.to_categorical(y_train_c)
y_test = np_utils.to_categorical(y_test_c)
y_train.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
df_pixels_final = df_test['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
X_final = np.vstack(df_pixels_final.values)
X_final.shape
X_final = X_final.reshape(-1, 48, 48, 1)
X_final.shape
plt.figure(0, figsize=(12,6))
for i in range(1, 13):
    plt.subplot(3,4,i)
    plt.imshow(X_train[i, :, :, 0], cmap="gray")

plt.tight_layout()
plt.show()
batch_size = 512
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
model2_file_name = 'SmallerCNN_23.06.2019'
model2_score = '0.598(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'SmallerCNN_23.06.2019/0.598(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
json_file = open('SmallerCNN_22.06.2019/0.619(public)/SmallerCNN_22.06.2019_model2.json', 'r')
# load model
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights
model2.load_weights("SmallerCNN_22.06.2019/0.619(public)/SmallerCNN_22.06.2019_model2.h5")
# load history
json_hist = open('SmallerCNN_22.06.2019/0.619(public)/SmallerCNN_22.06.2019_model2_hist.json', 'r')
loaded_hist_json = json_hist.read()
json_hist.close()
hist2.history = json.loads(loaded_hist_json)
        
print("Loaded model, weight and history from disk")
json_file = open('SmallerCNN_22.06.2019/0.619(public)/SmallerCNN_22.06.2019_model2.json', 'r')
# load model
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights
model2.load_weights("SmallerCNN_22.06.2019/0.619(public)/SmallerCNN_22.06.2019_model2.h5")
# load history
json_hist = open('SmallerCNN_22.06.2019/0.619(public)/SmallerCNN_22.06.2019_model2_hist.json', 'r')
loaded_hist_json = json_hist.read()
json_hist.close()
hist2.history = json.loads(loaded_hist_json)
        
print("Loaded model, weight and history from disk")
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history(hist):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist.history['loss'], color='b', label='Training Loss')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist.history['acc'], color='b', label='Training Accuracy')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()
plot_history(hist) 
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.summary()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
#model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 512
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 512
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.summary()
# Model 2 plot and write history
hist2 = TrainingPlot()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
fromas.preprocessing.image import ImageDatagenerator
froma keras.preprocessing.image import ImageDatagenerator
from keras.preprocessing.image import ImageDatagenerator
from keras.preprocessing.image import ImageDataGenerator
1.
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 512
epochs = 10
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data=(X_valid, y_valid),
                  shuffle=True)
train_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
train_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=20)
                                
                                    
df_pixels = df_train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
X_train = np.vstack(df_pixels.values)
y_train = np.array(df_train["target"])
X_train.shape, y_train.shape
X_train, X_test, y_train_c, y_test_c = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
X_train.shape, X_test.shape
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
X_train.shape, X_test.shape
y_train = np_utils.to_categorical(y_train_c)
y_test = np_utils.to_categorical(y_test_c)
y_train.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
train_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255,
                                    fill_mode='nearest')

val_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255,
                                    fill_mode='nearest')
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=20)

val_datagen.fit(X_valid)
val_generator = val_datagen.flow(X_val, y_val, batch_size=20)
                                
                                    
train_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255,
                                    fill_mode='nearest')

valid_datagen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255,
                                    fill_mode='nearest')
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=20)

valid_datagen.fit(X_valid)
valid_generator = valid_datagen.flow(X_valid, y_valid, batch_size=20)
                                
                                    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 512
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  shuffle=True)
len(X_train)
batch_size = 512
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  shuffle=True)
len(X_valid) / 64
len(X_valid)
len(X_valid), len(X_train)
len(X_train) / 32
batch_size = 512
epochs = 2
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 64,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 64,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 128,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 16,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 128,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
len(X_train) / 128
len(X_train) / 32
len(X_train) / 61
len(X_train) / 64
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 64,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
len(X_train) / 100
batch_size = 64
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 512
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 256
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
len(X_valid) / 32
128/2
len(X_train)/len(X_valid)
batch_size/4
batch_size = 128
epochs = 1
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  #validation_steps= len(X_valid) / 32,
                  validation_steps= batch_size/4,
                  shuffle=True)
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 300
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  #steps_per_epoch=len(X_train) / 32,
                  steps_per_epoch= batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  #validation_steps= len(X_valid) / 32,
                  validation_steps= batch_size/4,
                  shuffle=True)
# Model 2 plot and write history
hist2 = TrainingPlot()
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 100
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
# Model 2 plot and write history
hist2 = TrainingPlot()
num_labels = 7
width, height = 48, 48

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), 
                  data_format='channels_last', kernel_regularizer=l2(0.01)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
#model2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))

#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model2.add(Dropout(0.5))
"""
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Dropout(0.5))
"""
model2.add(Flatten())
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.4))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.4))
#model2.add(Dense(16, activation='relu'))
#model2.add(Dropout(0.5))

model2.add(Dense(num_labels, activation='softmax'))
#model.summary()
train_datagen = ImageDataGenerator(rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)

valid_datagen = ImageDataGenerator(rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=200)

valid_datagen.fit(X_valid)
valid_generator = valid_datagen.flow(X_valid, y_valid, batch_size=200)
                                
                                    
model2.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
batch_size = 128
epochs = 20
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
batch_size = 128
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 64
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 256
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 512
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
batch_size = 512
epochs = 30
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 32,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 32,
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
batch_size = 512
epochs = 50
#early stop
#early_stopping=EarlyStopping(monitor='val_loss', mode='min', verbose=1)

model2.fit_generator(train_generator,
                  steps_per_epoch=len(X_train) / 64,
                  epochs=epochs,
                  verbose=1,
                  callbacks=[hist2],
                  validation_data= valid_generator,
                  validation_steps= len(X_valid) / 64,
                  shuffle=True)
def plot_history_val(hist, hist2):
   plt.figure(figsize=(28,8))
   plt.subplot(1, 2, 1)
   plt.suptitle('Optimizer : Adam', fontsize=10)
   plt.ylabel('Loss', fontsize=16)
   plt.plot(hist2.history['val_loss'], color='b', label='Validation Loss 2')
   plt.plot(hist.history['val_loss'], color='r', label='Validation Loss 1')
   plt.grid()
   plt.legend(loc='upper right')

   plt.subplot(1, 2, 2)
   plt.ylabel('Accuracy', fontsize=16)
   plt.plot(hist2.history['val_acc'], color='b', label='Validation Accuracy 2')
   plt.plot(hist.history['val_acc'], color='r', label='Validation Accuracy 1')
   plt.legend(loc='lower right')
   plt.grid()
   plt.show()

plot_history_val(hist2, hist)    
X_test = ImageDataGenerator(rescale=1./255)
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
df_pixels = df_train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
X_train = np.vstack(df_pixels.values)
y_train = np.array(df_train["target"])
X_train.shape, y_train.shape
X_train, X_test, y_train_c, y_test_c = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
X_train.shape, X_test.shape
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
X_train.shape, X_test.shape
y_train = np_utils.to_categorical(y_train_c)
y_test = np_utils.to_categorical(y_test_c)
y_train.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                   test_size = 0.2,
                                                   random_state=1)
train_datagen = ImageDataGenerator(rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)

valid_datagen = ImageDataGenerator(rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen.fit(X_test)
test_generator = test_datagen.flow(X_test, y_test, batch_size=200)
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size=200)

valid_datagen.fit(X_valid)
valid_generator = valid_datagen.flow(X_valid, y_valid, batch_size=200)
                                
                                    
accuracy_score(y_test_c, np.argmax(model2.predict(X_test), axis=1))
score2 = model2.evaluate(X_test, y_test, verbose=0)
print(model2.metrics_names)
print(score2)
test_loss, test_acc = model2.evaluate_generator(X_test, y_test, step=50)
test_loss, test_acc = model2.evaluate_generator(X_test, y_test, steps=50)
test_loss, test_acc = model2.evaluate_generator(X_test, y_test)
test_loss, test_acc = model2.evaluate_generator(test_generator, steps=50)
print('test_loss, test_acc')
print(test_loss, test_acc)
print('test_loss, test_acc')
print(round(test_loss, 2), round(test_acc, 2))
print('test_loss, test_acc')
print(round(test_loss, 2),'  ', round(test_acc, 2))
print('test_loss, test_acc')
print(round(test_loss, 2),'      ', round(test_acc, 2))
print('test_loss: ', round(test_loss, 2))
print('test_acc: ', round(test_acc, 2))
model2_file_name = 'GenCNN_25.06.2019'
model2_score = '0.62(test)'
#create folder
newpath = model2_file_name + '/' + model2_score 
if not os.path.exists(newpath):
    os.makedirs(newpath)
# save best weights
model2.save_weights(model2_file_name + '/'+ model2_score + '/' + model2_file_name + "_model2.h5")

# save model to json
model2_json = model2.to_json()
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2.json", "w") as json_file:
    json_file.write(model2_json)
# save history
hist_json = json.dumps(hist2.history)
with open(model2_file_name + '/' + model2_score + '/' + model2_file_name + "_model2_hist.json", "w") as json_file:
    json_file.write(hist_json)    
# save description
get_ipython().run_line_magic('save', "-f 'GenCNN_25.06.2019/0.62(test)/'model2_file_name 1-999999")
clear_output()
print('All files saved')
