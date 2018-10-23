## Needed imports
import csv
import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization, Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
import tarfile

LOCAL = True

## Prepare data (extract data and make data generator for later use)
if LOCAL:
    # Read data on local computer
    to_unzip = glob.glob(os.path.join('training_data','*.tar.gz'))
    if len(to_unzip) > 0 :
        print('Unzipping the data ... ')
        for u in to_unzip:
            tar = tarfile.open(u, "r:gz")
            tar.extractall(os.path.join('.','training_data'))
            tar.close()
            os.remove(u)
            print('- '+ u + ' unzipped!')

if LOCAL:
    folders = ['data', 'own_data', 'reverse_tour'] # List of folders containing training data
    source = os.path.join('.','training_data')
else :
    # Read data on remote Udacity Linux computer
    folders = ['data', 'reverse_tour', 'own_data'] # List of folders containing training data
    source = '/opt/carnd_p3/' # Common path

samples = []
for folder in folders :
    print('Reading .. ' + folder)
    with open(os.path.join(source,folder,'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if 'steering' not in line :
                new_line = line.copy()
                for k in range(3):
                    new_line[k] = os.path.join(source,folder,'IMG',line[k].split('/')[-1]) # To have the correct absolute path of each training sample
                samples.append(new_line)

#  - Generator  (On the fly data augmentation + Right/Center/Left images)
def generator(samples, batch_size=32):
    '''
        This function is a generator that enables, instead of storing the whole dataset in memory all at once,
        yields pieces of data as we need them.
        It also takes care on the fly of the data augmentation (image flipping + left and right image with correction)
    '''

    correction = 0.2
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for k in range(3):
                    name = batch_sample[k]
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3])

                    if 'right' in name:
                        angle-=correction
                    if 'left' in name:
                        angle+=correction
                    if 'center' in name:
                        images.append(cv2.flip(image,1))
                        angles.append(-1.0*angle)

                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

batch_size = 32
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
print('Generator succesfully created')



## Driving Agent
# - Network Architecture (NVIDIA Self-driving architecture) + Batch normalization for regularization

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160,320,3))) # Normalization
model.add(Cropping2D(cropping=((70,25),(0,0)))) # Image cropping to keep the region of interest

model.add(Conv2D(24, (5, 5), strides = (2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(36, (5, 5), strides = (2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), strides = (2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# - Model Training
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size,
                                     validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, epochs=3, verbose = 1)

model.save('model.h5')

# - Visualization
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss']) # plot the training and validation loss for each epoch
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.pdf')
