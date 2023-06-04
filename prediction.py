import numpy as np
from keras_preprocessing import image
from keras.utils import img_to_array

def predict_image(test_image,
                  model,
                  class_name,
                  image_size: int = None,
                 ):
    
    test_image = image.load_img(test_image, target_size=(image_size,image_size))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    results = model.predict(test_image)
    i = 0
    prediction = 'Unknown'
    for name in class_name:
        if results[0][i] > 0.4:
            prediction = name
            break
        i+=1
    
    return prediction
    