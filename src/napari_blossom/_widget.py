from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
import cv2
from tqdm import tqdm
from napari.types import ImageData, LabelsData
import napari
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
from napari import Viewer
from PIL import Image
zip_dir = tempfile.TemporaryDirectory()

A = []
B = []
MASK = []
names = []
path_folder = (zip_dir.name).replace("\\","/")
current_item = []
current_mask_selected_set = []

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

def single_image(layer,image_viewer):
    
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
    output_image_bin = np.where(output_image[:,:,0],255,0)
    output_image_bin = output_image_bin.astype("uint8")
    return image_viewer.add_labels(output_image_bin,name="mask")

def multiple_image(layer,image_viewer):
    nbr_image,h_,l_,channel_ = layer.shape
    
    model_New = tf.keras.models.load_model(os.path.join(paths.get_models_dir(),'best_model_W_BCE_chpping.h5'),custom_objects={'dice_coefficient': dice_coefficient})
    empty_output = np.zeros((nbr_image,h_,l_),np.uint8)
    
    img_array = np.array(layer)
    for ix in tqdm(range(nbr_image),desc="Processing image"):
      current_img = img_array[ix,...]
      img1_list = get_mosaic(current_img)
      
      taille_p = 256
      X_ensemble = np.zeros((len(img1_list), taille_p, taille_p, 3), dtype=np.uint8)
      for n in range(len(img1_list)):
        sz1_x,sz2_x,sz3_x = img1_list[n].shape
        if (sz1_x,sz2_x)==(256,256):
          X_ensemble[n]=img1_list[n]

      preds_test = model_New.predict(X_ensemble, verbose=1)
      preds_test_opt = (preds_test > 0.2).astype(np.uint8)
      output_image = reconstruire(current_img,preds_test_opt)
      output_image_bin = np.where(output_image[:,:,0],255,0)
      output_image_bin = output_image_bin.astype("uint8")
      empty_output[ix,...] = output_image_bin

    return image_viewer.add_labels(empty_output,name="stack mask")

def do_image_segmentation(layer,image_viewer):
    nbr_image = len(layer.shape)
    if nbr_image==3:
        return single_image(layer,image_viewer)
    elif nbr_image==4:
        return multiple_image(layer,image_viewer)
    else:
        show_info("Input isnot RGB image nor stack RGB image")

@magic_factory(call_button="Run")
def do_model_segmentation(
    layer: ImageData,image_viewer: Viewer):
    return do_image_segmentation(layer,image_viewer)

@magic_factory(call_button="save zip",layout="vertical")
def save_as_zip(layer_mask: LabelsData,layer_RGB : ImageData):
    save_button = QPushButton("Save as zip")
    filename, _ = QFileDialog.getSaveFileName(save_button, "Save as zip", ".", "zip")

    nbr_image = layer_RGB.shape
    nbr_mask = layer_mask.shape
    
    if len(nbr_image)==4:
        assert nbr_image[0]==nbr_mask[0], "MASK AND RGB SIZE NOT EQUAL"
    elif len(nbr_image)==3:
        assert len(nbr_mask)==2, "NOT A SINGLE MASK"
    else:
        assert len(nbr_image)==4 or len(nbr_mask)==2, "NOT STACK NOR SINGLE IMAGE"
        

    total_RGB = nbr_image[0]
    img_array = np.array(layer_RGB)
    msk_array = np.array(layer_mask)
    for ix in tqdm(range(total_RGB),"Extracting"):
        data_RGB = img_array[ix,...]
        data_msk = msk_array[ix,...]
        
        im_RGB = Image.fromarray(data_RGB)
        im_msk = Image.fromarray(data_msk)
        
        im_RGB1 = im_RGB.save(os.path.join(zip_dir.name,'RGB_'+str(ix)+".png"))
        im_msk1 = im_msk.save(os.path.join(zip_dir.name,'MSK_'+str(ix)+".png"))


    shutil.make_archive(filename, 'zip', zip_dir.name)
    show_info('Compressed file done')