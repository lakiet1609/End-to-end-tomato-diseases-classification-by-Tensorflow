from tensorflow import keras
from keras import regularizers

def custom_model(image_size: int = None,
                 output_features: int = None):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (image_size,image_size,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
              
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_features, activation='softmax')
    ])
    return model