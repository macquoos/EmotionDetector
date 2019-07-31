
#import shutil
#import os
#
#legend = pd.read_csv("dataset/data/legend.csv").values
#
## drop the user.id
#legend = legend[:,1:]
#
#dictImg = { name : label for name, label in legend } 
#
#
#for key in dictImg:
#    if dictImg[key].lower() == 'anger':
#        shutil.move("dataset/images/"+key, "dataset/anger/"+key)

path = 'dataset/training'


from keras.preprocessing.image import ImageDataGenerator

def getTrainingSet(trainingDir):
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    training_set = train_datagen.flow_from_directory(trainingDir,
                                                     target_size = (128, 128),
                                                     batch_size = 32,
                                                     color_mode = 'grayscale',
                                                     class_mode = 'categorical')
    
    return training_set
