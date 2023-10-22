# make_napari_viewer is a pytest fixture that returns a napari viewer object
from napari_blossom import *

import numpy as np
import pytest
from napari.types import ImageData, LabelsData
from napari.layers import Image, Labels
import napari

@pytest.fixture
def im_rgb():
    return ImageData(np.random.randint(256,size=(1420,1000,3)))


def get_er(*args, **kwargs):
    er_func = do_model_segmentation()
    return er_func(*args, **kwargs)

def test_segmentation(im_rgb):
    image_viewer = napari.Viewer()
    my_widget_thd = get_er(im_rgb,image_viewer)
    #check if output is numpy array
    print(my_widget_thd)
    assert type(my_widget_thd)==napari.layers.labels.labels.Labels
    #check if output is binary
    output_array = np.array(my_widget_thd.data)
    uniqe = np.unique(output_array.flatten())
    assert len(uniqe)==2