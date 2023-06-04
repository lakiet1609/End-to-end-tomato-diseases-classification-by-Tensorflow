import matplotlib.pyplot as plt
import numpy as np
from data_preprocess import data_generate
from keras.preprocessing import image
from prediction import predict_image

def visualize_image(x_batch, 
                    y_batch, 
                    batch_size, 
                    class_idx):
    
    fig = plt.figure(figsize=(20,8))
    columns = 4
    rows = 4
    fig.subplots_adjust(wspace=1.2, hspace=0.3)
    for i in range(1, columns*rows):
        num = np.random.randint(batch_size)
        image = x_batch[num].astype(np.int32)
        fig.add_subplot(rows, columns, i)
        label = int(np.argmax(y_batch[num]))
        plt.title(f'{label}: {[k for k, v in class_idx.items() if v == label]}')
        plt.imshow(image)
        plt.axis(False)
    # plt.show()
    
def visualize_pred_img(x_batch, 
                       y_batch, 
                       model, 
                       it,
                       batch_size
                       ):
    
    fig = plt.figure(figsize=(20,8))
    columns = 4
    rows = 4
    fig.subplots_adjust(wspace=1.2, hspace=0.8)
    for i in range(1, columns*rows-1):
        num = np.random.randint(batch_size)
        image = x_batch[num].astype(np.int32)
        image = np.expand_dims(image, axis=0)
        name = model.predict(image)
        name = np.argmax(name, axis=-1)
        true_name = y_batch[num]
        true_name = int(np.argmax(true_name, axis=-1))
        fig.add_subplot(rows, columns, i)
        plt.title(f'Actual value {true_name}: {[k for k, v in it.items() if v == true_name]}\nPrediction {name}: {[k for k, v in it.items() if v == name]}')
        plt.imshow(image.squeeze(0))     
    plt.show()


def plot_evaluation(history):
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['validation','train'], loc = 'upper right')
    plt.show()
    
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['validation','train'], loc = 'upper right')
    plt.show()


