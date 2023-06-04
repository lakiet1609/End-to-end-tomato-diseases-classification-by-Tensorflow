import numpy as np
from imutils import paths
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def data_generate(train_dir: str = None,
                  test_dir: str = None,
                  image_size: int = None,
                  batch_size: int = None):
    
    #Get the train generator
    train_datagen = ImageDataGenerator(validation_split=0.2,
                                       rescale=1./255,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(image_size,image_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        color_mode='rgb',
                                                        subset='training',
                                                        shuffle=True)
    
    #Get the validation generator
    valid_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)
    
    val_generator = valid_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(image_size,image_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        color_mode='rgb',
                                                        subset='validation',
                                                        shuffle=False)
    
    #Get the test generator
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                                        target_size=(image_size,image_size),
                                                        batch_size=1,
                                                        class_mode='categorical',
                                                        color_mode='rgb',
                                                        shuffle=False)
    
    return train_generator, val_generator, test_generator


