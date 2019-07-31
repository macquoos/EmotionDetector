

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

#    # Read training and test data files
#    train = pd.read_csv("dataset/train.csv").values
#    test  = pd.read_csv("dataset/test.csv").values
#    
#    trainX = train[:, 1]
#    
#    x_train = []
#    
#    # Process data to convert into 48x48 array and normalize
#    for idx in range(0, trainX.size):
#         tmp = np.asarray([int(s) for s in trainX[idx].split(' ')])
#         x_train.append(np.array(tmp,dtype='uint8').reshape(48,48,1) / 255)
#        
#    x_train = np.asarray(x_train)
#    
#    # Process labels
#    y_train = train[:, 0]
#    tmp = []
#    for idx in range(0, y_train.size):
#        tmp.append(y_train[idx])
#    
#    # Binarize outputs
#    #from sklearn import preprocessing
#    #lb = preprocessing.LabelBinarizer()
#    #y_train = lb.fit_transform(tmp)
#    from keras.utils import to_categorical
#    y_train = to_categorical(tmp)
    
    
def setupClassifier():
    # Setup CNN
    model = Sequential()
    #K.set_image_dim_ordering('th')
    model.add(Convolution2D(32, (3,3), input_shape=(128,128,1),activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3,3), activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' ))
    model.add(Dense(50, activation= 'relu' ))
    model.add(Dense(4, activation= 'softmax' ))
      # Compile model
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    
    return model



def trainClassifier(training_set, classifierModel, epochs = 25):
    
    # Train the classifier
    
    classifierModel.fit_generator(training_set,
                                  steps_per_epoch = training_set.samples / training_set.batch_size,
                                  epochs = 25)
    
    classifierModel.save('classifier')
    
    return classifierModel
    

import cv2


def detectEmotion(img,target_size,classification_model):
    
    # Process the image
    cropped_img = cv2.resize(img,target_size)
    
    cropped_img = np.expand_dims(cropped_img, axis = 0)
    cropped_img = np.expand_dims(cropped_img, axis = 3)
    # scale the image
    scaled_img = cropped_img / 255
    
    result = classification_model.predict(scaled_img)
    
    # get the index of result with max probability
    max_iter = np.argmax(result[0])
    
    emo_dict = {0: 'angry',
                1: 'happy',
                2: 'neutral',
                3: 'sad' }
    
    # return emotion and confidence
    return (emo_dict[max_iter], result[0][max_iter])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




