from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import cv2
from napari.types import ImageData
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from skimage.transform import resize
from napari.utils.notifications import show_info

def do_image_segmentation(
    layer: ImageData
    ) -> ImageData:
    
    def redimension(image):
        X = np.zeros((1,256,256,3),dtype=np.uint8)   
        size_ = image.shape
        img = image[:,:,:3]
        X[0] = resize(img, (256, 256), mode='constant', preserve_range=True)
        return X,size_

    def dice_coefficient(y_true, y_pred):
        eps = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 
    
    image_reshaped,size_ = redimension(layer)
    v = os.path.abspath(__file__)
    model_new = tf.keras.models.load_model("src/napari_blossom/best_model_FL_BCE_0_5.h5",custom_objects={'dice_coefficient': dice_coefficient})
    prediction = model_new.predict(image_reshaped)
    preds_test_t = (prediction > 0.2).astype(np.uint8)
    temp=np.squeeze(preds_test_t[0,:,:,0])*255
    
    return cv2.resize(temp, dsize=(size_[1],size_[0]))

@magic_factory(call_button="Run")
def do_model_segmentation(
    layer: ImageData) -> ImageData:
    show_info('Succes !')
    return do_image_segmentation(layer)
