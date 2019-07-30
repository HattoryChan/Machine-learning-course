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
