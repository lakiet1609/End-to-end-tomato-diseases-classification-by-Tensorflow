import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import numpy as np
import split_custom_folder,plot_graph, data_preprocess, CNN_builder, prediction
import os

def main():
    data_path = 'tomato'
    data_path_train = 'tomato/train'
    data_path_test = 'tomato/test'

    #Set parameters for the model
    image_size = 128
    batch_size = 32
    epochs = 12
    worker = os.cpu_count()
    

    ## Split the dataset in the right form
    split_custom_folder.custom_folder(data_path)

    # Split the dataset into training set and testing set
    train_generator, val_generator, test_generator = data_preprocess.data_generate(train_dir=data_path_train,
                                                                                   test_dir=data_path_test,
                                                                                   batch_size=batch_size,
                                                                                   image_size=image_size)

    # Take the images and labels of 1 batch on the training set
    x_batch, y_batch = train_generator.next() 
    it = train_generator.class_indices
    print(it)
    num_classes = len(np.unique(train_generator.classes))
    class_name = list(it.keys())
    print(f'Number of classes: {num_classes}')

    ## Take the images and labels of 1 batch on the testing set
    x_batch_test, y_batch_test = test_generator.next()


    # Visualize image inside 1 batch of the training data
    plot_graph.visualize_image(x_batch=x_batch,
                               y_batch=y_batch,
                               batch_size=batch_size,
                               class_idx=it)
    
    # CNN model
    model_0 = CNN_builder.custom_model(image_size=image_size,
                                       output_features=num_classes)
    
    # Compile model 
    model_0.compile(optimizer='adam',
                    loss= tf.losses.CategoricalCrossentropy,
                    metrics= ['accuracy'])
    
    # Define the learning rate schedule
    def lr_schedule(epoch):
        lr = 0.01
        if epoch > 4:
            lr *= 0.1
        elif epoch > 8:
            lr *= 0.05
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # Train model
    history = model_0.fit(x=train_generator,
                          steps_per_epoch=train_generator.n // batch_size,
                          epochs=epochs,
                          validation_data=val_generator,
                          validation_steps=val_generator.n // batch_size,
                          workers=worker,
                          callbacks=[lr_scheduler])
    
    # Save model
    model_0.save('Model_0.h5')
    
    # Load model
    model_0 = load_model('Model_0.h5')
    print(model_0.summary())
    
    #Plot the evaluation graph 
    plot_graph.plot_evaluation(history=history)
    
    # Plot the prediction in 1 batch of the test set and compared to actual value 
    plot_graph.visualize_pred_img(x_batch=x_batch_test,
                                  y_batch=y_batch_test,
                                  it=it,
                                  batch_size=batch_size,
                                  model=model_0)

    #Predict a single chosen image
    image_path = r'tomato\test\Tomato_Leaf_Mold\0eda4dc5-7f7b-4e27-9e53-1e0326eef88e___Crnl_L.Mold 8850.JPG'
    print(prediction.predict_image(test_image=image_path,
                             model=model_0,
                             class_name=class_name,
                             image_size=image_size))
if __name__ == '__main__':
    main()


