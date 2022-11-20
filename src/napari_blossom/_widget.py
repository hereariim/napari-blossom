from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import cv2
from napari.types import ImageData, LabelsData
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from skimage.transform import resize
from napari.utils.notifications import show_info
from focal_loss import BinaryFocalLoss
import pathlib
import napari_blossom.path as paths
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QGridLayout, QPushButton, QFileDialog, QWidget, QListWidget
import shutil
import tempfile

zip_dir = tempfile.TemporaryDirectory()

def get_mosaic(img):
  A = []
  h,l,z = img.shape
  #longueur
  L1 = [ i for i in range(0,l-255,255)]+[l-255]
  L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

  #hauteur
  R1 = [ i for i in range(0,h-255,255)]+[h-255]
  R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]

  for h1,h2 in zip(R1,R2):
    for l1,l2 in zip(L1,L2):
      A.append(img[h1:h2,l1:l2])
  return A

def reconstruire(img1,K):
  ex,ey,ez=img1.shape
  A = np.zeros((ex,ey,1), dtype=np.bool)
  h=ex
  l=ey
  z=1
  
  #longueur
  L1 = [ i for i in range(0,l-255,255)]+[l-255]
  L2 = [ 256+i for i in range(0,l,255) if 256+i < l]+[l]

  #hauteur
  R1 = [ i for i in range(0,h-255,255)]+[h-255]
  R2 = [ 256+i for i in range(0,h,255) if 256+i < h]+[h]
  
  n = 0
  for h1,h2 in zip(R1,R2):
    for l1,l2 in zip(L1,L2):
      if A[h1:h2,l1:l2].shape == K[n].shape:
        A[h1:h2,l1:l2] = K[n]
      else:
        A[h1:h2,l1:l2] = np.zeros(A[h1:h2,l1:l2].shape, dtype=np.bool)
      n+=1
  return A

def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 

def do_image_segmentation(
    layer: ImageData
    ) -> ImageData:
    
    img1_list = get_mosaic(layer)

    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),'best_model_W_BCE_chpping.h5'),custom_objects={'dice_coefficient': dice_coefficient})

    taille_p = 256
    X_ensemble = np.zeros((len(img1_list), taille_p, taille_p, 3), dtype=np.uint8)
    for n in range(len(img1_list)):
      sz1_x,sz2_x,sz3_x = img1_list[n].shape
      if (sz1_x,sz2_x)==(256,256):
        X_ensemble[n]=img1_list[n]

    preds_test = model_New.predict(X_ensemble, verbose=1)
    preds_test_opt = (preds_test > 0.2).astype(np.uint8)
    output_image = reconstruire(layer,preds_test_opt)
    imsave(f'{zip_dir.name}\image_output.png',output_image)
    return np.squeeze(output_image[:,:,0])

@magic_factory(call_button="Load",filename={"label": "Pick a file:"})
def get_data(filename=pathlib.Path.cwd()) -> ImageData:
    return imread(filename)[:,:,:3]

@magic_factory(call_button="Run")
def do_model_segmentation(
    layer: ImageData) -> LabelsData:
    show_info('Succes !')
    return do_image_segmentation(layer)

@magic_factory(call_button="save zip",layout="vertical")
def save_as_zip():
    save_button = QPushButton("Save as zip")
    filename, _ = QFileDialog.getSaveFileName(save_button, "Save as zip", ".", "zip")
    shutil.make_archive(filename, 'zip', zip_dir.name)
    show_info('Compressed file done')