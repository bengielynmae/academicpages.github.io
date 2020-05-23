---
title: "An AI Classifier of Plant Disease"
excerpt: "Convolutional neural networks identify plant diseases from photos of leaves. The project promotes early detection of crop diseases to help improve food security and lives of farmers. <br/><br><img src='/images/plant-disease/plant-disease-cover.png'>"
collection: portfolio
---

<h2>Overview</h2>
<p>This was a final project output for our <b>Deep Learning</b> course under Prof. Chris Monterola in the M.Sc. Data Science program. In this study, the models developed performed well in identifying plant-disease pairs with accuracies around 97%. As was demonstrated, using high-resolution (256x256) images provided marginally superior performance to low-resolution (64x64) images. This was presented to the public in December 2019.</p>


# Take it or Leaf it!
## Plant Disease Classification through Leaf Analysis Using CNNs </b> 


## Summary/Abstract

Plant disease presents a threat to national food security and the livelihood of agricultural workers as diseased plants are less productive. Early, accurate, and rapid detection of these diseases can aid in proper planning and reduce overall costs associated with both the damages incurred and surveillance of agricultural lands.

To provide a tool that can deliver these results, this study developed a deep learning model that can identify diseases present in three species of plants. Through the combnation of multiple CNNs, two models were developed for relatively high resolution coloredimages (256x256 pixles) and lower resolution versions of these images (64x64 pixels). The first model had an f1 score of 97.7% while the second second model using lower resolution images had an f1 score of 97%. Both models had difficulty differentiating between healthy plant leaf specimens making it less useful for identifying plant types and is more accurate in identifying diseases that afflict each. 

## Introduction

In order to meet the demand of more than 7 billion people in the world, human society has been able to harness the power of modern technologies to produce enough food. However, one of the major factors shaking the food demand, viz, food security, remains threatened by a number of factors, which includes decline in pollinators (Report of the Plenary of the Intergovernmental Science-PolicyPlatform on Biodiversity Ecosystem and Services on the work of its fourth session, 2016),, climate change (Tai et al., 2014), plant diseases and others.  

Deep neural networks have recently been successfully applied in many diverse domains as examples of end to end learning. Neural networks provide a mapping between an input—such as an image of a diseased plant—to an output—such as a crop-disease pair. The nodes in a neural network are mathematical functions that take numerical inputs from the incoming edges and provide a numerical output as an outgoing edge. Deep neural networks are simply mapping the input layer to the output layer over a series of stacked layers of nodes. The challenge is to create a deep network in such a way that both the structure of the network as well as the functions (nodes) and edge weights correctly map the input to the output. Deep neural networks are trained by tuning the network parameters in such a way that the mapping improves during the training process. This process is computationally challenging and has in recent times been improved dramatically by several both conceptual and engineering breakthroughs (LeCun et al., 2015; Schmidhuber, 2015). 

  

In order to develop accurate image classifiers for the purposes of plant disease diagnosis, a large, verified dataset of images of diseased and healthy plants. To address this problem, the PlantVillage project has begun collecting tens of thousands of images of healthy and diseased crop plants (Hughes and Salathé, 2015), and has made them openly and freely available. Here, we report the classification of 10 diseases in 3 crop species using 54,306 images with a convolutional neural network approach. The performance of our models was measured by their ability to predict the correct crop-diseases pair, given 38 possible classes. The best performing model achieves a mean F1 score of 0.977 (overall accuracy of 97.70%), hence demonstrating the technical feasibility of our approach. Our results are a first step toward a smartphone-assisted plant disease diagnosis system. 

The goal of this study is to develop a model that can aid the agricultural through the use of deep learning to provide a fast, accurate, an low-cost process that can identify diseases in plants.

## Methods (Data Pre-processing)

The model is to be constructed mainly using the <a href='https://www.tensorflow.org/guide/keras/overview'> `tensorflow` implementation of the `keras` API </a>. Some helper functions are also loaded from <a href='https://opencv.org/'> opencv (cv2) </a> and <a href='https://scikit-learn.org/stable/'> scikit-learn (sklearn) </a>.


```python
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

Setting the intial hyperparameters of model for use later on.

`EPOCHS` : number of epochs for training

`INIT_LR`: the initial learning rate of the model

`BS`     : the batch size per epoch

`width`  : the number of pixels that make up the width of the image

`height` : the number of pixels that make up the height of the image

`depth`  : the number of dimensions for each channel of the image (Red, Green, and Blue for color)

The `EPOCHS`, `INIT_LR`, and `BS` were chosen after tuning the model already.


```python
EPOCHS = 30
INIT_LR = 6e-4
BS = 32
width=64
height=64
depth=3
```


```python
default_image_size = tuple((width, height))
image_size = 0
directory_root = 'input/plantvillage/'
```

We define a helper function to convert images to `numpy` arrays for compatibility in the machine learning models.


```python
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
```

The images are fetched from the directory and organized by their labels.


```python
image_list, label_list = [], []
try:
    print(directory_root)
    root_dir = listdir(directory_root)
    for directory in root_dir :
        root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        for disease_folder in plant_disease_folder_list :
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)
            if disease_folder == ".ipynb_checkpoints" :
                plant_disease_folder_list.remove(disease_folder)
            if "Untitled" in disease_folder :
                plant_disease_folder_list.remove(disease_folder)
                
        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:500]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
```

    [INFO] Loading images ...
    input/plantvillage/
    [INFO] Processing Potato__healthy ...
    [INFO] Processing Tomato__Target_Spot ...
    [INFO] Processing Tomato__Tomato_mosaic_virus ...
    [INFO] Processing Tomato__Bacterial_spot ...
    [INFO] Processing Pepper_bell__healthy ...
    [INFO] Processing Pepper_bell__Bacterial_spot ...
    [INFO] Processing Potato__Early_blight ...
    [INFO] Processing Potato__Late_blight ...
    [INFO] Processing Tomato__Tomato_YellowLeaf__Curl_Virus ...
    [INFO] Processing Tomato__Healthy ...
    [INFO] Image loading completed


The number of images sampled in this study:


```python
image_size = len(image_list)
```


```python
image_size
```




    4522



Transforming the image labels using `scikit-learn's LabelBinarizer function.


```python
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
n_classes = len(label_binarizer.classes_)
```


```python
label_binarizer.classes_
```




    array(['Pepper_bell__Bacterial_spot', 'Pepper_bell__healthy',
           'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy',
           'Tomato__Bacterial_spot', 'Tomato__Healthy', 'Tomato__Target_Spot',
           'Tomato__Tomato_YellowLeaf__Curl_Virus',
           'Tomato__Tomato_mosaic_virus'], dtype='<U37')




```python
pcc = dict(zip(label_binarizer.classes_, image_labels.sum(axis = 0)))
pcc
```




    {'Pepper_bell__Bacterial_spot': 499,
     'Pepper_bell__healthy': 499,
     'Potato__Early_blight': 500,
     'Potato__Late_blight': 500,
     'Potato__healthy': 152,
     'Tomato__Bacterial_spot': 500,
     'Tomato__Healthy': 500,
     'Tomato__Target_Spot': 500,
     'Tomato__Tomato_YellowLeaf__Curl_Virus': 499,
     'Tomato__Tomato_mosaic_virus': 373}



Showing the number of images for each of the 10 classes in the sample dataset.


```python
plt.barh(list(pcc.keys()), list(pcc.values()))
```




    <BarContainer object of 10 artists>




![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_20_1.png)


Examining what the images actually look like in both its 256x256 original version and its 64x64 pixel version.


```python
def convert_image_to_array_spec(image_dir, size = 256):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, (size,size))   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
```


```python
plt.imshow(convert_image_to_array_spec('input/plantvillage/plants/Potato__Early_blight/001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG')/255)
```




    <matplotlib.image.AxesImage at 0x7fe504252a20>




![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_23_1.png)



```python
plt.imshow(convert_image_to_array_spec('input/plantvillage/plants/Potato__Early_blight/001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG', size = 64)/255)
```




    <matplotlib.image.AxesImage at 0x7fe50422b630>




![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_24_1.png)


We see that there is loss of some features as the edges are now poorly defined. However, by eyeball method, it can still be said to be a leaf that has some discoloration in its leaves.

Calculating the Proportional Chance Criterion (PCC) for this problem:


```python
np.sum([(x/8519)**2 for x in image_labels.sum(axis = 0)])
```




    0.029752459721412102



Normalizing the image arrays' values to be between 0 and 1


```python
np_image_list = np.array(image_list) / 255.0
```

## Methods (Deep Learning Models)

Two separate models are generated for the 256x256 images and the 64x64 images. The architecture described is the same for both models with the only difference being in the final max pooling layer which was changed in the interest of simplifying computation time. The CNNs all had ReLU activation functions while the final dense output used a Softmax function since there are many classes which can be predicted. 

  

The entire dataset was split into a training and test set with a 75:25 ratio. To improve the generalization of the model given the number of images obtained, data augmentation was performed on the training set. Images used to train the model were rotated up to 30 degrees, height shifted up to 15%, sheared horizontally up to 15%, zoomed in up to 20% and could be horizontally flipped. The training set per batch was drawn from a uniformly random distribution of these transformed images. 

  

In training, the loss function to be optimized is the binary cross-entropy while using an Adam optimizer. Despite the problem being a multi-class classification problem generally requiring a categorical cross-entropy function to be optimized, a binary cross-entropy function proved to result in significantly better performance. The optimizer used a learning rate of 6e-4 and a decay of 1.2e-5. For each trial, the model was trained over 50 epochs with a batch size of 32.   

### 256x256 model


```python
height = 256
width = 256
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 100)
aug = ImageDataGenerator(
    rotation_range=30, height_shift_range=0.15, shear_range=0.15, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
    model.add(Conv2D(64, (2, 2), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])
```

We define a helper function that trains the model and returns the accuracies an losses for both the training and validation sets. There is also a provision to set the random state to set different realizations. 


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 256, 256, 64)      832       
    _________________________________________________________________
    activation (Activation)      (None, 256, 256, 64)      0         
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256, 256, 64)      256       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 85, 85, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 85, 85, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 85, 85, 64)        36928     
    _________________________________________________________________
    activation_1 (Activation)    (None, 85, 85, 64)        0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 85, 85, 64)        256       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 85, 85, 64)        36928     
    _________________________________________________________________
    activation_2 (Activation)    (None, 85, 85, 64)        0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 85, 85, 64)        256       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 42, 42, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 42, 42, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 42, 42, 128)       73856     
    _________________________________________________________________
    activation_3 (Activation)    (None, 42, 42, 128)       0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 42, 42, 128)       512       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 42, 42, 128)       147584    
    _________________________________________________________________
    activation_4 (Activation)    (None, 42, 42, 128)       0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 42, 42, 128)       512       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 21, 21, 128)       0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 21, 21, 128)       0         
    _________________________________________________________________
    flatten (Flatten)            (None, 56448)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              57803776  
    _________________________________________________________________
    activation_5 (Activation)    (None, 1024)              0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 1024)              4096      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                10250     
    _________________________________________________________________
    activation_6 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 58,116,042
    Trainable params: 58,113,098
    Non-trainable params: 2,944
    _________________________________________________________________



```python
def do_a_run(rstate = 200):
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = rstate)
    aug = ImageDataGenerator(
        rotation_range=30, height_shift_range=0.15, shear_range=0.15, 
        zoom_range=0.2,horizontal_flip=True, 
        fill_mode="nearest")
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(64, (2, 2), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    return acc, val_acc, loss, val_loss, epochs
```


```python
acc1, val_acc1, loss1, val_loss1, epochs1 = do_a_run(rstate=100)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1762 - accuracy: 0.9372
    Epoch 00001: val_accuracy improved from -inf to 0.82228, saving model to BEST-256x256-rstate-100-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 17s 163ms/step - loss: 0.1751 - accuracy: 0.9375 - val_loss: 1.0676 - val_accuracy: 0.8223
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0849 - accuracy: 0.9693
    Epoch 00002: val_accuracy improved from 0.82228 to 0.82405, saving model to BEST-256x256-rstate-100-weights-improvement-02-0.82.hdf5
    105/105 [==============================] - 16s 154ms/step - loss: 0.0850 - accuracy: 0.9693 - val_loss: 0.9059 - val_accuracy: 0.8240
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0757 - accuracy: 0.9722
    Epoch 00003: val_accuracy improved from 0.82405 to 0.89080, saving model to BEST-256x256-rstate-100-weights-improvement-03-0.89.hdf5
    105/105 [==============================] - 15s 146ms/step - loss: 0.0752 - accuracy: 0.9723 - val_loss: 0.7272 - val_accuracy: 0.8908
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0577 - accuracy: 0.9783
    Epoch 00004: val_accuracy did not improve from 0.89080
    105/105 [==============================] - 15s 142ms/step - loss: 0.0574 - accuracy: 0.9784 - val_loss: 0.8505 - val_accuracy: 0.8557
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0573 - accuracy: 0.9792
    Epoch 00005: val_accuracy improved from 0.89080 to 0.89355, saving model to BEST-256x256-rstate-100-weights-improvement-05-0.89.hdf5
    105/105 [==============================] - 16s 151ms/step - loss: 0.0571 - accuracy: 0.9793 - val_loss: 0.5667 - val_accuracy: 0.8935
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0485 - accuracy: 0.9830
    Epoch 00006: val_accuracy did not improve from 0.89355
    105/105 [==============================] - 15s 143ms/step - loss: 0.0486 - accuracy: 0.9829 - val_loss: 0.8793 - val_accuracy: 0.8683
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0537 - accuracy: 0.9801
    Epoch 00007: val_accuracy improved from 0.89355 to 0.97401, saving model to BEST-256x256-rstate-100-weights-improvement-07-0.97.hdf5
    105/105 [==============================] - 16s 151ms/step - loss: 0.0537 - accuracy: 0.9802 - val_loss: 0.0980 - val_accuracy: 0.9740
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0443 - accuracy: 0.9833
    Epoch 00008: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 144ms/step - loss: 0.0447 - accuracy: 0.9832 - val_loss: 0.3550 - val_accuracy: 0.9393
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0420 - accuracy: 0.9850
    Epoch 00009: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 140ms/step - loss: 0.0421 - accuracy: 0.9849 - val_loss: 0.8488 - val_accuracy: 0.8774
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0333 - accuracy: 0.9880
    Epoch 00010: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 142ms/step - loss: 0.0331 - accuracy: 0.9881 - val_loss: 0.0903 - val_accuracy: 0.9724
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0310 - accuracy: 0.9884
    Epoch 00011: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 142ms/step - loss: 0.0311 - accuracy: 0.9885 - val_loss: 0.4356 - val_accuracy: 0.9306
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0290 - accuracy: 0.9890
    Epoch 00012: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 138ms/step - loss: 0.0293 - accuracy: 0.9889 - val_loss: 0.1624 - val_accuracy: 0.9536
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0338 - accuracy: 0.9873
    Epoch 00013: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 140ms/step - loss: 0.0341 - accuracy: 0.9872 - val_loss: 0.1369 - val_accuracy: 0.9627
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0328 - accuracy: 0.9884
    Epoch 00014: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 14s 137ms/step - loss: 0.0326 - accuracy: 0.9885 - val_loss: 0.3925 - val_accuracy: 0.9233
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0240 - accuracy: 0.9903
    Epoch 00015: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 140ms/step - loss: 0.0238 - accuracy: 0.9904 - val_loss: 0.4863 - val_accuracy: 0.9236
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0230 - accuracy: 0.9904
    Epoch 00016: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 14s 138ms/step - loss: 0.0229 - accuracy: 0.9905 - val_loss: 0.2486 - val_accuracy: 0.9474
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0257 - accuracy: 0.9910
    Epoch 00017: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 142ms/step - loss: 0.0256 - accuracy: 0.9910 - val_loss: 0.1942 - val_accuracy: 0.9577
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0204 - accuracy: 0.9926
    Epoch 00018: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 14s 137ms/step - loss: 0.0207 - accuracy: 0.9925 - val_loss: 0.1840 - val_accuracy: 0.9687
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0166 - accuracy: 0.9936
    Epoch 00019: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 138ms/step - loss: 0.0166 - accuracy: 0.9935 - val_loss: 0.1781 - val_accuracy: 0.9639
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0185 - accuracy: 0.9929
    Epoch 00020: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 14s 138ms/step - loss: 0.0184 - accuracy: 0.9930 - val_loss: 0.3029 - val_accuracy: 0.9446
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0184 - accuracy: 0.9928
    Epoch 00021: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 14s 138ms/step - loss: 0.0183 - accuracy: 0.9929 - val_loss: 0.2235 - val_accuracy: 0.9586
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0192 - accuracy: 0.9932
    Epoch 00022: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 140ms/step - loss: 0.0191 - accuracy: 0.9932 - val_loss: 0.3403 - val_accuracy: 0.9434
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0220 - accuracy: 0.9923
    Epoch 00023: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 15s 138ms/step - loss: 0.0219 - accuracy: 0.9923 - val_loss: 0.2723 - val_accuracy: 0.9501
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0186 - accuracy: 0.9933
    Epoch 00024: val_accuracy did not improve from 0.97401
    105/105 [==============================] - 14s 135ms/step - loss: 0.0186 - accuracy: 0.9934 - val_loss: 0.1881 - val_accuracy: 0.9630
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0177 - accuracy: 0.9939
    Epoch 00025: val_accuracy improved from 0.97401 to 0.99045, saving model to BEST-256x256-rstate-100-weights-improvement-25-0.99.hdf5
    105/105 [==============================] - 15s 147ms/step - loss: 0.0179 - accuracy: 0.9939 - val_loss: 0.0294 - val_accuracy: 0.9905
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0155 - accuracy: 0.9946
    Epoch 00026: val_accuracy did not improve from 0.99045
    105/105 [==============================] - 14s 138ms/step - loss: 0.0156 - accuracy: 0.9945 - val_loss: 0.4195 - val_accuracy: 0.9224
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0210 - accuracy: 0.9925
    Epoch 00027: val_accuracy did not improve from 0.99045
    105/105 [==============================] - 14s 137ms/step - loss: 0.0210 - accuracy: 0.9926 - val_loss: 0.3291 - val_accuracy: 0.9385
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0200 - accuracy: 0.9922
    Epoch 00028: val_accuracy did not improve from 0.99045
    105/105 [==============================] - 14s 138ms/step - loss: 0.0201 - accuracy: 0.9921 - val_loss: 0.2061 - val_accuracy: 0.9629
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0162 - accuracy: 0.9942
    Epoch 00029: val_accuracy did not improve from 0.99045
    105/105 [==============================] - 15s 138ms/step - loss: 0.0163 - accuracy: 0.9940 - val_loss: 0.2768 - val_accuracy: 0.9505
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0168 - accuracy: 0.9946
    Epoch 00030: val_accuracy did not improve from 0.99045
    105/105 [==============================] - 14s 136ms/step - loss: 0.0169 - accuracy: 0.9945 - val_loss: 0.2108 - val_accuracy: 0.9565



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_37_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_37_2.png)



```python
acc2, val_acc2, loss2, val_loss2, epochs2 = do_a_run(rstate=200)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1723 - accuracy: 0.9391
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-256x256-rstate-200-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 16s 149ms/step - loss: 0.1718 - accuracy: 0.9394 - val_loss: 1.3689 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0971 - accuracy: 0.9635
    Epoch 00002: val_accuracy did not improve from 0.82334
    105/105 [==============================] - 15s 144ms/step - loss: 0.0971 - accuracy: 0.9634 - val_loss: 1.8149 - val_accuracy: 0.8233
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0746 - accuracy: 0.9729
    Epoch 00003: val_accuracy improved from 0.82334 to 0.82414, saving model to BEST-256x256-rstate-200-weights-improvement-03-0.82.hdf5
    105/105 [==============================] - 16s 153ms/step - loss: 0.0745 - accuracy: 0.9728 - val_loss: 1.3302 - val_accuracy: 0.8241
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0684 - accuracy: 0.9757
    Epoch 00004: val_accuracy improved from 0.82414 to 0.85031, saving model to BEST-256x256-rstate-200-weights-improvement-04-0.85.hdf5
    105/105 [==============================] - 16s 152ms/step - loss: 0.0684 - accuracy: 0.9758 - val_loss: 0.7779 - val_accuracy: 0.8503
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0510 - accuracy: 0.9806
    Epoch 00005: val_accuracy improved from 0.85031 to 0.85897, saving model to BEST-256x256-rstate-200-weights-improvement-05-0.86.hdf5
    105/105 [==============================] - 15s 146ms/step - loss: 0.0512 - accuracy: 0.9806 - val_loss: 0.6138 - val_accuracy: 0.8590
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0529 - accuracy: 0.9804
    Epoch 00006: val_accuracy improved from 0.85897 to 0.88462, saving model to BEST-256x256-rstate-200-weights-improvement-06-0.88.hdf5
    105/105 [==============================] - 16s 149ms/step - loss: 0.0530 - accuracy: 0.9804 - val_loss: 0.5787 - val_accuracy: 0.8846
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0449 - accuracy: 0.9836
    Epoch 00007: val_accuracy improved from 0.88462 to 0.89328, saving model to BEST-256x256-rstate-200-weights-improvement-07-0.89.hdf5
    105/105 [==============================] - 16s 149ms/step - loss: 0.0449 - accuracy: 0.9836 - val_loss: 0.5841 - val_accuracy: 0.8933
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0476 - accuracy: 0.9827
    Epoch 00008: val_accuracy improved from 0.89328 to 0.89576, saving model to BEST-256x256-rstate-200-weights-improvement-08-0.90.hdf5
    105/105 [==============================] - 15s 147ms/step - loss: 0.0473 - accuracy: 0.9828 - val_loss: 0.6060 - val_accuracy: 0.8958
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0427 - accuracy: 0.9839
    Epoch 00009: val_accuracy improved from 0.89576 to 0.90416, saving model to BEST-256x256-rstate-200-weights-improvement-09-0.90.hdf5
    105/105 [==============================] - 16s 151ms/step - loss: 0.0423 - accuracy: 0.9840 - val_loss: 0.6015 - val_accuracy: 0.9042
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0436 - accuracy: 0.9854
    Epoch 00010: val_accuracy improved from 0.90416 to 0.95429, saving model to BEST-256x256-rstate-200-weights-improvement-10-0.95.hdf5
    105/105 [==============================] - 15s 146ms/step - loss: 0.0439 - accuracy: 0.9854 - val_loss: 0.2145 - val_accuracy: 0.9543
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0317 - accuracy: 0.9890
    Epoch 00011: val_accuracy did not improve from 0.95429
    105/105 [==============================] - 15s 142ms/step - loss: 0.0315 - accuracy: 0.9890 - val_loss: 0.6749 - val_accuracy: 0.9073
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0286 - accuracy: 0.9898
    Epoch 00012: val_accuracy improved from 0.95429 to 0.96667, saving model to BEST-256x256-rstate-200-weights-improvement-12-0.97.hdf5
    105/105 [==============================] - 16s 148ms/step - loss: 0.0285 - accuracy: 0.9898 - val_loss: 0.1909 - val_accuracy: 0.9667
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0277 - accuracy: 0.9895
    Epoch 00013: val_accuracy did not improve from 0.96667
    105/105 [==============================] - 15s 139ms/step - loss: 0.0278 - accuracy: 0.9894 - val_loss: 0.5599 - val_accuracy: 0.8988
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0353 - accuracy: 0.9878
    Epoch 00014: val_accuracy did not improve from 0.96667
    105/105 [==============================] - 15s 141ms/step - loss: 0.0351 - accuracy: 0.9879 - val_loss: 0.3776 - val_accuracy: 0.9270
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0254 - accuracy: 0.9910
    Epoch 00015: val_accuracy did not improve from 0.96667
    105/105 [==============================] - 15s 140ms/step - loss: 0.0253 - accuracy: 0.9910 - val_loss: 0.2075 - val_accuracy: 0.9597
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0208 - accuracy: 0.9923
    Epoch 00016: val_accuracy did not improve from 0.96667
    105/105 [==============================] - 15s 140ms/step - loss: 0.0208 - accuracy: 0.9923 - val_loss: 0.1767 - val_accuracy: 0.9610
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0252 - accuracy: 0.9914
    Epoch 00017: val_accuracy improved from 0.96667 to 0.97109, saving model to BEST-256x256-rstate-200-weights-improvement-17-0.97.hdf5
    105/105 [==============================] - 15s 145ms/step - loss: 0.0253 - accuracy: 0.9915 - val_loss: 0.1014 - val_accuracy: 0.9711
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0273 - accuracy: 0.9897
    Epoch 00018: val_accuracy did not improve from 0.97109
    105/105 [==============================] - 14s 138ms/step - loss: 0.0271 - accuracy: 0.9898 - val_loss: 0.4169 - val_accuracy: 0.9284
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0229 - accuracy: 0.9918
    Epoch 00019: val_accuracy improved from 0.97109 to 0.98886, saving model to BEST-256x256-rstate-200-weights-improvement-19-0.99.hdf5
    105/105 [==============================] - 15s 143ms/step - loss: 0.0228 - accuracy: 0.9918 - val_loss: 0.0379 - val_accuracy: 0.9889
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0227 - accuracy: 0.9919
    Epoch 00020: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 15s 139ms/step - loss: 0.0225 - accuracy: 0.9920 - val_loss: 0.1219 - val_accuracy: 0.9692
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0192 - accuracy: 0.9931
    Epoch 00021: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 15s 140ms/step - loss: 0.0193 - accuracy: 0.9930 - val_loss: 0.1671 - val_accuracy: 0.9676
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0151 - accuracy: 0.9942
    Epoch 00022: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 15s 140ms/step - loss: 0.0150 - accuracy: 0.9943 - val_loss: 0.1732 - val_accuracy: 0.9625
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0176 - accuracy: 0.9937
    Epoch 00023: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 135ms/step - loss: 0.0176 - accuracy: 0.9937 - val_loss: 0.0713 - val_accuracy: 0.9817
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0167 - accuracy: 0.9943
    Epoch 00024: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 137ms/step - loss: 0.0166 - accuracy: 0.9943 - val_loss: 0.1763 - val_accuracy: 0.9637
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0192 - accuracy: 0.9934
    Epoch 00025: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 136ms/step - loss: 0.0190 - accuracy: 0.9935 - val_loss: 0.9473 - val_accuracy: 0.8845
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0176 - accuracy: 0.9934
    Epoch 00026: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 15s 140ms/step - loss: 0.0174 - accuracy: 0.9934 - val_loss: 0.0832 - val_accuracy: 0.9773
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0183 - accuracy: 0.9934
    Epoch 00027: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 137ms/step - loss: 0.0182 - accuracy: 0.9935 - val_loss: 0.1475 - val_accuracy: 0.9665
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0125 - accuracy: 0.9956
    Epoch 00028: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 134ms/step - loss: 0.0124 - accuracy: 0.9956 - val_loss: 0.1371 - val_accuracy: 0.9714
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0156 - accuracy: 0.9951
    Epoch 00029: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 134ms/step - loss: 0.0158 - accuracy: 0.9950 - val_loss: 0.1186 - val_accuracy: 0.9706
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0206 - accuracy: 0.9919
    Epoch 00030: val_accuracy did not improve from 0.98886
    105/105 [==============================] - 14s 135ms/step - loss: 0.0205 - accuracy: 0.9919 - val_loss: 0.6513 - val_accuracy: 0.9059



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_38_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_38_2.png)



```python
acc3, val_acc3, loss3, val_loss3, epochs3 = do_a_run(rstate=300)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1788 - accuracy: 0.9375
    Epoch 00001: val_accuracy improved from -inf to 0.83696, saving model to BEST-256x256-rstate-300-weights-improvement-01-0.84.hdf5
    105/105 [==============================] - 16s 150ms/step - loss: 0.1784 - accuracy: 0.9376 - val_loss: 1.1545 - val_accuracy: 0.8370
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1034 - accuracy: 0.9620
    Epoch 00002: val_accuracy improved from 0.83696 to 0.83722, saving model to BEST-256x256-rstate-300-weights-improvement-02-0.84.hdf5
    105/105 [==============================] - 16s 148ms/step - loss: 0.1029 - accuracy: 0.9622 - val_loss: 0.8893 - val_accuracy: 0.8372
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0792 - accuracy: 0.9717
    Epoch 00003: val_accuracy improved from 0.83722 to 0.83767, saving model to BEST-256x256-rstate-300-weights-improvement-03-0.84.hdf5
    105/105 [==============================] - 16s 152ms/step - loss: 0.0790 - accuracy: 0.9718 - val_loss: 1.0556 - val_accuracy: 0.8377
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0625 - accuracy: 0.9771
    Epoch 00004: val_accuracy improved from 0.83767 to 0.85827, saving model to BEST-256x256-rstate-300-weights-improvement-04-0.86.hdf5
    105/105 [==============================] - 16s 149ms/step - loss: 0.0623 - accuracy: 0.9772 - val_loss: 0.7480 - val_accuracy: 0.8583
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0666 - accuracy: 0.9764
    Epoch 00005: val_accuracy improved from 0.85827 to 0.88453, saving model to BEST-256x256-rstate-300-weights-improvement-05-0.88.hdf5
    105/105 [==============================] - 16s 148ms/step - loss: 0.0666 - accuracy: 0.9764 - val_loss: 0.5033 - val_accuracy: 0.8845
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0544 - accuracy: 0.9810
    Epoch 00006: val_accuracy improved from 0.88453 to 0.95376, saving model to BEST-256x256-rstate-300-weights-improvement-06-0.95.hdf5
    105/105 [==============================] - 15s 148ms/step - loss: 0.0550 - accuracy: 0.9809 - val_loss: 0.1328 - val_accuracy: 0.9538
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0522 - accuracy: 0.9815
    Epoch 00007: val_accuracy did not improve from 0.95376
    105/105 [==============================] - 15s 143ms/step - loss: 0.0518 - accuracy: 0.9817 - val_loss: 0.3393 - val_accuracy: 0.9268
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0444 - accuracy: 0.9840
    Epoch 00008: val_accuracy did not improve from 0.95376
    105/105 [==============================] - 15s 140ms/step - loss: 0.0442 - accuracy: 0.9840 - val_loss: 0.2078 - val_accuracy: 0.9419
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0391 - accuracy: 0.9864
    Epoch 00009: val_accuracy did not improve from 0.95376
    105/105 [==============================] - 15s 142ms/step - loss: 0.0393 - accuracy: 0.9862 - val_loss: 0.9821 - val_accuracy: 0.8790
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0369 - accuracy: 0.9868
    Epoch 00010: val_accuracy improved from 0.95376 to 0.96808, saving model to BEST-256x256-rstate-300-weights-improvement-10-0.97.hdf5
    105/105 [==============================] - 15s 147ms/step - loss: 0.0366 - accuracy: 0.9868 - val_loss: 0.1249 - val_accuracy: 0.9681
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0335 - accuracy: 0.9879
    Epoch 00011: val_accuracy did not improve from 0.96808
    105/105 [==============================] - 15s 141ms/step - loss: 0.0333 - accuracy: 0.9880 - val_loss: 0.2548 - val_accuracy: 0.9413
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0348 - accuracy: 0.9877
    Epoch 00012: val_accuracy did not improve from 0.96808
    105/105 [==============================] - 15s 140ms/step - loss: 0.0348 - accuracy: 0.9877 - val_loss: 0.2350 - val_accuracy: 0.9500
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0306 - accuracy: 0.9892
    Epoch 00013: val_accuracy improved from 0.96808 to 0.98294, saving model to BEST-256x256-rstate-300-weights-improvement-13-0.98.hdf5
    105/105 [==============================] - 16s 148ms/step - loss: 0.0305 - accuracy: 0.9893 - val_loss: 0.0583 - val_accuracy: 0.9829
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0296 - accuracy: 0.9888
    Epoch 00014: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 137ms/step - loss: 0.0294 - accuracy: 0.9888 - val_loss: 0.7949 - val_accuracy: 0.8953
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0254 - accuracy: 0.9903
    Epoch 00015: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 15s 139ms/step - loss: 0.0256 - accuracy: 0.9902 - val_loss: 0.1964 - val_accuracy: 0.9514
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0254 - accuracy: 0.9914
    Epoch 00016: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 135ms/step - loss: 0.0254 - accuracy: 0.9914 - val_loss: 0.6409 - val_accuracy: 0.9020
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0241 - accuracy: 0.9913
    Epoch 00017: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 137ms/step - loss: 0.0239 - accuracy: 0.9914 - val_loss: 0.1196 - val_accuracy: 0.9752
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0188 - accuracy: 0.9929
    Epoch 00018: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 136ms/step - loss: 0.0188 - accuracy: 0.9929 - val_loss: 0.2792 - val_accuracy: 0.9360
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0312 - accuracy: 0.9891
    Epoch 00019: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 134ms/step - loss: 0.0310 - accuracy: 0.9892 - val_loss: 0.0778 - val_accuracy: 0.9738
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0231 - accuracy: 0.9908
    Epoch 00020: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 134ms/step - loss: 0.0233 - accuracy: 0.9908 - val_loss: 0.8224 - val_accuracy: 0.8939
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0216 - accuracy: 0.9920
    Epoch 00021: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 138ms/step - loss: 0.0216 - accuracy: 0.9920 - val_loss: 0.1615 - val_accuracy: 0.9570
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0207 - accuracy: 0.9929
    Epoch 00022: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 137ms/step - loss: 0.0206 - accuracy: 0.9929 - val_loss: 0.4378 - val_accuracy: 0.9331
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0176 - accuracy: 0.9938
    Epoch 00023: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 138ms/step - loss: 0.0177 - accuracy: 0.9938 - val_loss: 0.3520 - val_accuracy: 0.9343
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0197 - accuracy: 0.9922
    Epoch 00024: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 134ms/step - loss: 0.0196 - accuracy: 0.9923 - val_loss: 0.2451 - val_accuracy: 0.9523
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0215 - accuracy: 0.9925
    Epoch 00025: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 14s 137ms/step - loss: 0.0213 - accuracy: 0.9926 - val_loss: 0.2451 - val_accuracy: 0.9543
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0169 - accuracy: 0.9941
    Epoch 00026: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 15s 142ms/step - loss: 0.0168 - accuracy: 0.9941 - val_loss: 0.5828 - val_accuracy: 0.9229
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0223 - accuracy: 0.9925
    Epoch 00027: val_accuracy did not improve from 0.98294
    105/105 [==============================] - 15s 140ms/step - loss: 0.0221 - accuracy: 0.9926 - val_loss: 0.4462 - val_accuracy: 0.9355
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0188 - accuracy: 0.9930
    Epoch 00028: val_accuracy improved from 0.98294 to 0.98992, saving model to BEST-256x256-rstate-300-weights-improvement-28-0.99.hdf5
    105/105 [==============================] - 15s 141ms/step - loss: 0.0190 - accuracy: 0.9928 - val_loss: 0.0401 - val_accuracy: 0.9899
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0149 - accuracy: 0.9946
    Epoch 00029: val_accuracy did not improve from 0.98992
    105/105 [==============================] - 14s 133ms/step - loss: 0.0151 - accuracy: 0.9945 - val_loss: 0.4727 - val_accuracy: 0.9284
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0167 - accuracy: 0.9940
    Epoch 00030: val_accuracy did not improve from 0.98992
    105/105 [==============================] - 14s 134ms/step - loss: 0.0166 - accuracy: 0.9940 - val_loss: 0.0980 - val_accuracy: 0.9785



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_39_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_39_2.png)



```python
acc4, val_acc4, loss4, val_loss4, epochs4 = do_a_run(rstate=400)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1762 - accuracy: 0.9373
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-256x256-rstate-400-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 16s 154ms/step - loss: 0.1760 - accuracy: 0.9374 - val_loss: 1.7352 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0995 - accuracy: 0.9635
    Epoch 00002: val_accuracy improved from 0.82334 to 0.82440, saving model to BEST-256x256-rstate-400-weights-improvement-02-0.82.hdf5
    105/105 [==============================] - 15s 147ms/step - loss: 0.0993 - accuracy: 0.9634 - val_loss: 1.1947 - val_accuracy: 0.8244
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0782 - accuracy: 0.9710
    Epoch 00003: val_accuracy improved from 0.82440 to 0.83103, saving model to BEST-256x256-rstate-400-weights-improvement-03-0.83.hdf5
    105/105 [==============================] - 16s 153ms/step - loss: 0.0787 - accuracy: 0.9709 - val_loss: 0.9932 - val_accuracy: 0.8310
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0721 - accuracy: 0.9724
    Epoch 00004: val_accuracy improved from 0.83103 to 0.83696, saving model to BEST-256x256-rstate-400-weights-improvement-04-0.84.hdf5
    105/105 [==============================] - 16s 154ms/step - loss: 0.0719 - accuracy: 0.9725 - val_loss: 1.0745 - val_accuracy: 0.8370
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0644 - accuracy: 0.9770
    Epoch 00005: val_accuracy improved from 0.83696 to 0.86225, saving model to BEST-256x256-rstate-400-weights-improvement-05-0.86.hdf5
    105/105 [==============================] - 18s 168ms/step - loss: 0.0640 - accuracy: 0.9772 - val_loss: 0.6875 - val_accuracy: 0.8622
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0515 - accuracy: 0.9822
    Epoch 00006: val_accuracy improved from 0.86225 to 0.92087, saving model to BEST-256x256-rstate-400-weights-improvement-06-0.92.hdf5
    105/105 [==============================] - 16s 151ms/step - loss: 0.0515 - accuracy: 0.9822 - val_loss: 0.3610 - val_accuracy: 0.9209
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0480 - accuracy: 0.9825
    Epoch 00007: val_accuracy improved from 0.92087 to 0.94624, saving model to BEST-256x256-rstate-400-weights-improvement-07-0.95.hdf5
    105/105 [==============================] - 15s 147ms/step - loss: 0.0478 - accuracy: 0.9826 - val_loss: 0.2289 - val_accuracy: 0.9462
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0428 - accuracy: 0.9851
    Epoch 00008: val_accuracy did not improve from 0.94624
    105/105 [==============================] - 15s 142ms/step - loss: 0.0425 - accuracy: 0.9852 - val_loss: 0.2418 - val_accuracy: 0.9362
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0376 - accuracy: 0.9860
    Epoch 00009: val_accuracy did not improve from 0.94624
    105/105 [==============================] - 15s 141ms/step - loss: 0.0374 - accuracy: 0.9861 - val_loss: 0.3700 - val_accuracy: 0.9393
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0415 - accuracy: 0.9843
    Epoch 00010: val_accuracy did not improve from 0.94624
    105/105 [==============================] - 15s 139ms/step - loss: 0.0413 - accuracy: 0.9843 - val_loss: 0.3996 - val_accuracy: 0.9260
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0386 - accuracy: 0.9871
    Epoch 00011: val_accuracy improved from 0.94624 to 0.95871, saving model to BEST-256x256-rstate-400-weights-improvement-11-0.96.hdf5
    105/105 [==============================] - 16s 148ms/step - loss: 0.0383 - accuracy: 0.9871 - val_loss: 0.2034 - val_accuracy: 0.9587
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0351 - accuracy: 0.9876
    Epoch 00012: val_accuracy did not improve from 0.95871
    105/105 [==============================] - 14s 138ms/step - loss: 0.0354 - accuracy: 0.9875 - val_loss: 0.5345 - val_accuracy: 0.9036
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0318 - accuracy: 0.9885
    Epoch 00013: val_accuracy improved from 0.95871 to 0.97206, saving model to BEST-256x256-rstate-400-weights-improvement-13-0.97.hdf5
    105/105 [==============================] - 15s 146ms/step - loss: 0.0316 - accuracy: 0.9885 - val_loss: 0.1195 - val_accuracy: 0.9721
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0280 - accuracy: 0.9899
    Epoch 00014: val_accuracy did not improve from 0.97206
    105/105 [==============================] - 15s 139ms/step - loss: 0.0280 - accuracy: 0.9898 - val_loss: 0.1617 - val_accuracy: 0.9597
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0300 - accuracy: 0.9897
    Epoch 00015: val_accuracy improved from 0.97206 to 0.97816, saving model to BEST-256x256-rstate-400-weights-improvement-15-0.98.hdf5
    105/105 [==============================] - 15s 144ms/step - loss: 0.0298 - accuracy: 0.9898 - val_loss: 0.1016 - val_accuracy: 0.9782
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0234 - accuracy: 0.9910
    Epoch 00016: val_accuracy did not improve from 0.97816
    105/105 [==============================] - 15s 139ms/step - loss: 0.0234 - accuracy: 0.9910 - val_loss: 0.2826 - val_accuracy: 0.9530
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0283 - accuracy: 0.9900
    Epoch 00017: val_accuracy did not improve from 0.97816
    105/105 [==============================] - 14s 135ms/step - loss: 0.0282 - accuracy: 0.9900 - val_loss: 0.2228 - val_accuracy: 0.9609
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0232 - accuracy: 0.9907
    Epoch 00018: val_accuracy improved from 0.97816 to 0.97905, saving model to BEST-256x256-rstate-400-weights-improvement-18-0.98.hdf5
    105/105 [==============================] - 15s 141ms/step - loss: 0.0231 - accuracy: 0.9908 - val_loss: 0.1075 - val_accuracy: 0.9790
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0212 - accuracy: 0.9921
    Epoch 00019: val_accuracy did not improve from 0.97905
    105/105 [==============================] - 14s 136ms/step - loss: 0.0210 - accuracy: 0.9922 - val_loss: 0.7744 - val_accuracy: 0.9004
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0216 - accuracy: 0.9917
    Epoch 00020: val_accuracy did not improve from 0.97905
    105/105 [==============================] - 14s 133ms/step - loss: 0.0215 - accuracy: 0.9918 - val_loss: 0.2421 - val_accuracy: 0.9608
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0224 - accuracy: 0.9917
    Epoch 00021: val_accuracy did not improve from 0.97905
    105/105 [==============================] - 14s 134ms/step - loss: 0.0223 - accuracy: 0.9917 - val_loss: 0.2919 - val_accuracy: 0.9508
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0209 - accuracy: 0.9920
    Epoch 00022: val_accuracy did not improve from 0.97905
    105/105 [==============================] - 14s 135ms/step - loss: 0.0213 - accuracy: 0.9920 - val_loss: 0.1095 - val_accuracy: 0.9691
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0207 - accuracy: 0.9915
    Epoch 00023: val_accuracy did not improve from 0.97905
    105/105 [==============================] - 15s 139ms/step - loss: 0.0205 - accuracy: 0.9915 - val_loss: 0.2747 - val_accuracy: 0.9519
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0225 - accuracy: 0.9920
    Epoch 00024: val_accuracy improved from 0.97905 to 0.98258, saving model to BEST-256x256-rstate-400-weights-improvement-24-0.98.hdf5
    105/105 [==============================] - 15s 141ms/step - loss: 0.0224 - accuracy: 0.9920 - val_loss: 0.0725 - val_accuracy: 0.9826
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0142 - accuracy: 0.9944
    Epoch 00025: val_accuracy did not improve from 0.98258
    105/105 [==============================] - 14s 137ms/step - loss: 0.0141 - accuracy: 0.9945 - val_loss: 0.1034 - val_accuracy: 0.9739
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0178 - accuracy: 0.9934
    Epoch 00026: val_accuracy did not improve from 0.98258
    105/105 [==============================] - 14s 137ms/step - loss: 0.0177 - accuracy: 0.9935 - val_loss: 0.3206 - val_accuracy: 0.9345
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0175 - accuracy: 0.9932
    Epoch 00027: val_accuracy did not improve from 0.98258
    105/105 [==============================] - 14s 133ms/step - loss: 0.0174 - accuracy: 0.9933 - val_loss: 0.2632 - val_accuracy: 0.9540
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0163 - accuracy: 0.9941
    Epoch 00028: val_accuracy improved from 0.98258 to 0.98285, saving model to BEST-256x256-rstate-400-weights-improvement-28-0.98.hdf5
    105/105 [==============================] - 15s 142ms/step - loss: 0.0165 - accuracy: 0.9940 - val_loss: 0.0750 - val_accuracy: 0.9828
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0166 - accuracy: 0.9941
    Epoch 00029: val_accuracy did not improve from 0.98285
    105/105 [==============================] - 14s 136ms/step - loss: 0.0166 - accuracy: 0.9941 - val_loss: 0.3857 - val_accuracy: 0.9490
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0156 - accuracy: 0.9940
    Epoch 00030: val_accuracy did not improve from 0.98285
    105/105 [==============================] - 15s 139ms/step - loss: 0.0156 - accuracy: 0.9940 - val_loss: 0.1437 - val_accuracy: 0.9698



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_40_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_40_2.png)



```python
val_accs = np.vstack([val_acc1,val_acc2,val_acc3,val_acc4])
```


```python
np.max(np.sum(val_accs, axis = 0)/4)
```




    0.9767463




```python
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.load_weights("BEST-256x256-rstate-100-weights-improvement-25-0.99.hdf5")
model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 0)
y_pred = np.rint(model.predict(x_test))
confmat = confusion_matrix(np.array(y_test).argmax(axis=1), np.array(y_pred).argmax(axis=1))
```


```python
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
```


![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_44_0.png)



```python
pcc
```




    {'Pepper_bell__Bacterial_spot': 499,
     'Pepper_bell__healthy': 499,
     'Potato__Early_blight': 500,
     'Potato__Late_blight': 500,
     'Potato__healthy': 152,
     'Tomato__Bacterial_spot': 500,
     'Tomato__Healthy': 500,
     'Tomato__Target_Spot': 500,
     'Tomato__Tomato_YellowLeaf__Curl_Virus': 499,
     'Tomato__Tomato_mosaic_virus': 373}



Analyzing the results of the classification report, the model had difficulty classifying healthy potato leaves. Potato leaves tagged as healthy were being misclassified as healthy bell pepper leaves or as having late blight. The misclassification of leaves to be of a different healthy species is no real cause for alarm as it can easily be controlled when batch testing leaves. This kind of error could be attributed to leaves being in various states of folding and having irregular shapes which confuses the model into identifying potato leaves as bell pepper leaves. On the other hand, the misclassifcations of the leaves as having late blight could prove to be problematic when making decisions. If plants are though to have late blight, measures taken to mitigate its spread can be wasteful of resources.

Significant misclassifications also occur for tomato leaves afflicted with target spot. Majority of tomatoes with target spot were deemed to be healthy by the model. This could be attributed to target spot occupying small and very irregular locations in the leaf could have made it difficult for the model to pick up on these salient features.

## 64x64 model

We follow the same procedure for the 256x256 model in training and evaluating the model.


```python
height = 64
width = 64
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 100)
aug = ImageDataGenerator(
    rotation_range=30, height_shift_range=0.15, shear_range=0.15, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(64, (2, 2), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
```


```python
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_15 (Conv2D)           (None, 64, 64, 64)        832       
    _________________________________________________________________
    activation_19 (Activation)   (None, 64, 64, 64)        0         
    _________________________________________________________________
    batch_normalization_18 (Batc (None, 64, 64, 64)        256       
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 21, 21, 64)        0         
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 21, 21, 64)        0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 21, 21, 64)        36928     
    _________________________________________________________________
    activation_20 (Activation)   (None, 21, 21, 64)        0         
    _________________________________________________________________
    batch_normalization_19 (Batc (None, 21, 21, 64)        256       
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 21, 21, 64)        36928     
    _________________________________________________________________
    activation_21 (Activation)   (None, 21, 21, 64)        0         
    _________________________________________________________________
    batch_normalization_20 (Batc (None, 21, 21, 64)        256       
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 10, 10, 64)        0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 10, 10, 64)        0         
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, 10, 10, 128)       73856     
    _________________________________________________________________
    activation_22 (Activation)   (None, 10, 10, 128)       0         
    _________________________________________________________________
    batch_normalization_21 (Batc (None, 10, 10, 128)       512       
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, 10, 10, 128)       147584    
    _________________________________________________________________
    activation_23 (Activation)   (None, 10, 10, 128)       0         
    _________________________________________________________________
    batch_normalization_22 (Batc (None, 10, 10, 128)       512       
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 5, 5, 128)         0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 5, 5, 128)         0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 3200)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1024)              3277824   
    _________________________________________________________________
    activation_24 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    batch_normalization_23 (Batc (None, 1024)              4096      
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                10250     
    _________________________________________________________________
    activation_25 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 3,590,090
    Trainable params: 3,587,146
    Non-trainable params: 2,944
    _________________________________________________________________



```python
def do_a_run(rstate = 200):
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = rstate)
    aug = ImageDataGenerator(
        rotation_range=30, height_shift_range=0.15, shear_range=0.15, 
        zoom_range=0.2,horizontal_flip=True, 
        fill_mode="nearest")
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(64, (2, 2), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])
    from tensorflow.keras.callbacks import ModelCheckpoint
    filepath="BEST-64x64-rstate-"+str(rstate) + "-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS, verbose=1, callbacks=callbacks_list)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

    return acc, val_acc, loss, val_loss, epochs
```


```python
acc1, val_acc1, loss1, val_loss1, epochs1 = do_a_run(rstate=100)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.2144 - accuracy: 0.9255
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-128x128-rstate-100-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 8s 80ms/step - loss: 0.2133 - accuracy: 0.9259 - val_loss: 1.1733 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1181 - accuracy: 0.9562
    Epoch 00002: val_accuracy improved from 0.82334 to 0.82812, saving model to BEST-128x128-rstate-100-weights-improvement-02-0.83.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.1180 - accuracy: 0.9561 - val_loss: 0.8086 - val_accuracy: 0.8281
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0929 - accuracy: 0.9655
    Epoch 00003: val_accuracy did not improve from 0.82812
    105/105 [==============================] - 8s 75ms/step - loss: 0.0927 - accuracy: 0.9655 - val_loss: 1.1981 - val_accuracy: 0.8238
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0748 - accuracy: 0.9721
    Epoch 00004: val_accuracy improved from 0.82812 to 0.83731, saving model to BEST-128x128-rstate-100-weights-improvement-04-0.84.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0751 - accuracy: 0.9720 - val_loss: 0.8494 - val_accuracy: 0.8373
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0756 - accuracy: 0.9721
    Epoch 00005: val_accuracy improved from 0.83731 to 0.91839, saving model to BEST-128x128-rstate-100-weights-improvement-05-0.92.hdf5
    105/105 [==============================] - 8s 80ms/step - loss: 0.0755 - accuracy: 0.9722 - val_loss: 0.2892 - val_accuracy: 0.9184
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0657 - accuracy: 0.9752
    Epoch 00006: val_accuracy improved from 0.91839 to 0.95270, saving model to BEST-128x128-rstate-100-weights-improvement-06-0.95.hdf5
    105/105 [==============================] - 9s 83ms/step - loss: 0.0657 - accuracy: 0.9752 - val_loss: 0.1430 - val_accuracy: 0.9527
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0554 - accuracy: 0.9795
    Epoch 00007: val_accuracy did not improve from 0.95270
    105/105 [==============================] - 8s 72ms/step - loss: 0.0552 - accuracy: 0.9795 - val_loss: 0.1719 - val_accuracy: 0.9475
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0517 - accuracy: 0.9822
    Epoch 00008: val_accuracy did not improve from 0.95270
    105/105 [==============================] - 8s 74ms/step - loss: 0.0518 - accuracy: 0.9822 - val_loss: 0.2000 - val_accuracy: 0.9494
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0507 - accuracy: 0.9810
    Epoch 00009: val_accuracy did not improve from 0.95270
    105/105 [==============================] - 8s 78ms/step - loss: 0.0516 - accuracy: 0.9809 - val_loss: 0.9402 - val_accuracy: 0.8864
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0486 - accuracy: 0.9823
    Epoch 00010: val_accuracy improved from 0.95270 to 0.97171, saving model to BEST-128x128-rstate-100-weights-improvement-10-0.97.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.0484 - accuracy: 0.9823 - val_loss: 0.1115 - val_accuracy: 0.9717
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0403 - accuracy: 0.9854
    Epoch 00011: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 76ms/step - loss: 0.0402 - accuracy: 0.9854 - val_loss: 0.1079 - val_accuracy: 0.9660
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0372 - accuracy: 0.9870
    Epoch 00012: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 77ms/step - loss: 0.0372 - accuracy: 0.9870 - val_loss: 0.1477 - val_accuracy: 0.9582
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0370 - accuracy: 0.9866
    Epoch 00013: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 75ms/step - loss: 0.0375 - accuracy: 0.9865 - val_loss: 0.3576 - val_accuracy: 0.9383
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0434 - accuracy: 0.9847
    Epoch 00014: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 76ms/step - loss: 0.0440 - accuracy: 0.9845 - val_loss: 0.6192 - val_accuracy: 0.9176
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0340 - accuracy: 0.9871
    Epoch 00015: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 77ms/step - loss: 0.0343 - accuracy: 0.9870 - val_loss: 0.2377 - val_accuracy: 0.9402
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0368 - accuracy: 0.9861
    Epoch 00016: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 74ms/step - loss: 0.0365 - accuracy: 0.9863 - val_loss: 0.3710 - val_accuracy: 0.9251
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0294 - accuracy: 0.9891
    Epoch 00017: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 76ms/step - loss: 0.0293 - accuracy: 0.9891 - val_loss: 0.2531 - val_accuracy: 0.9497
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0280 - accuracy: 0.9896
    Epoch 00018: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 77ms/step - loss: 0.0281 - accuracy: 0.9896 - val_loss: 0.3450 - val_accuracy: 0.9292
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0269 - accuracy: 0.9907
    Epoch 00019: val_accuracy did not improve from 0.97171
    105/105 [==============================] - 8s 79ms/step - loss: 0.0268 - accuracy: 0.9908 - val_loss: 0.1988 - val_accuracy: 0.9524
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0269 - accuracy: 0.9899
    Epoch 00020: val_accuracy improved from 0.97171 to 0.97728, saving model to BEST-128x128-rstate-100-weights-improvement-20-0.98.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0270 - accuracy: 0.9899 - val_loss: 0.0774 - val_accuracy: 0.9773
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0282 - accuracy: 0.9904
    Epoch 00021: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 74ms/step - loss: 0.0281 - accuracy: 0.9904 - val_loss: 0.3259 - val_accuracy: 0.9386
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0291 - accuracy: 0.9890
    Epoch 00022: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 75ms/step - loss: 0.0291 - accuracy: 0.9890 - val_loss: 0.3384 - val_accuracy: 0.9349
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0259 - accuracy: 0.9905
    Epoch 00023: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 80ms/step - loss: 0.0257 - accuracy: 0.9906 - val_loss: 0.3368 - val_accuracy: 0.9325
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0243 - accuracy: 0.9913
    Epoch 00024: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 77ms/step - loss: 0.0244 - accuracy: 0.9912 - val_loss: 0.2878 - val_accuracy: 0.9360
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0243 - accuracy: 0.9910 ETA: 0s - loss:
    Epoch 00025: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 77ms/step - loss: 0.0244 - accuracy: 0.9910 - val_loss: 0.4148 - val_accuracy: 0.9388
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0208 - accuracy: 0.9923
    Epoch 00026: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 76ms/step - loss: 0.0208 - accuracy: 0.9923 - val_loss: 0.3041 - val_accuracy: 0.9402
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0253 - accuracy: 0.9908
    Epoch 00027: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 75ms/step - loss: 0.0252 - accuracy: 0.9909 - val_loss: 0.1205 - val_accuracy: 0.9719
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0217 - accuracy: 0.9918
    Epoch 00028: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 8s 74ms/step - loss: 0.0217 - accuracy: 0.9918 - val_loss: 0.3548 - val_accuracy: 0.9390
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0198 - accuracy: 0.9929
    Epoch 00029: val_accuracy did not improve from 0.97728
    105/105 [==============================] - 7s 71ms/step - loss: 0.0197 - accuracy: 0.9930 - val_loss: 0.2114 - val_accuracy: 0.9473
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0203 - accuracy: 0.9927
    Epoch 00030: val_accuracy improved from 0.97728 to 0.99160, saving model to BEST-128x128-rstate-100-weights-improvement-30-0.99.hdf5
    105/105 [==============================] - 8s 76ms/step - loss: 0.0202 - accuracy: 0.9927 - val_loss: 0.0216 - val_accuracy: 0.9916



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_52_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_52_2.png)



```python
acc2, val_acc2, loss2, val_loss2, epochs2 = do_a_run(rstate=200)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.2005 - accuracy: 0.9292
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-128x128-rstate-200-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.2001 - accuracy: 0.9295 - val_loss: 0.8780 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1155 - accuracy: 0.9572
    Epoch 00002: val_accuracy improved from 0.82334 to 0.82573, saving model to BEST-128x128-rstate-200-weights-improvement-02-0.83.hdf5
    105/105 [==============================] - 8s 80ms/step - loss: 0.1149 - accuracy: 0.9574 - val_loss: 0.8859 - val_accuracy: 0.8257
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0901 - accuracy: 0.9662
    Epoch 00003: val_accuracy improved from 0.82573 to 0.83254, saving model to BEST-128x128-rstate-200-weights-improvement-03-0.83.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.0897 - accuracy: 0.9664 - val_loss: 1.2932 - val_accuracy: 0.8325
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0714 - accuracy: 0.9742
    Epoch 00004: val_accuracy improved from 0.83254 to 0.83492, saving model to BEST-128x128-rstate-200-weights-improvement-04-0.83.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.0718 - accuracy: 0.9740 - val_loss: 1.2180 - val_accuracy: 0.8349
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0679 - accuracy: 0.9751
    Epoch 00005: val_accuracy improved from 0.83492 to 0.88064, saving model to BEST-128x128-rstate-200-weights-improvement-05-0.88.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0675 - accuracy: 0.9752 - val_loss: 0.5278 - val_accuracy: 0.8806
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0611 - accuracy: 0.9771
    Epoch 00006: val_accuracy improved from 0.88064 to 0.93156, saving model to BEST-128x128-rstate-200-weights-improvement-06-0.93.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0612 - accuracy: 0.9772 - val_loss: 0.2732 - val_accuracy: 0.9316
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0541 - accuracy: 0.9794
    Epoch 00007: val_accuracy improved from 0.93156 to 0.93660, saving model to BEST-128x128-rstate-200-weights-improvement-07-0.94.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0542 - accuracy: 0.9794 - val_loss: 0.2380 - val_accuracy: 0.9366
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0533 - accuracy: 0.9798
    Epoch 00008: val_accuracy improved from 0.93660 to 0.97595, saving model to BEST-128x128-rstate-200-weights-improvement-08-0.98.hdf5
    105/105 [==============================] - 9s 82ms/step - loss: 0.0532 - accuracy: 0.9799 - val_loss: 0.0708 - val_accuracy: 0.9760
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0425 - accuracy: 0.9839
    Epoch 00009: val_accuracy did not improve from 0.97595
    105/105 [==============================] - 8s 74ms/step - loss: 0.0426 - accuracy: 0.9839 - val_loss: 0.4192 - val_accuracy: 0.9160
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0433 - accuracy: 0.9847
    Epoch 00010: val_accuracy did not improve from 0.97595
    105/105 [==============================] - 8s 77ms/step - loss: 0.0431 - accuracy: 0.9846 - val_loss: 0.1144 - val_accuracy: 0.9656
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0399 - accuracy: 0.9853
    Epoch 00011: val_accuracy improved from 0.97595 to 0.97666, saving model to BEST-128x128-rstate-200-weights-improvement-11-0.98.hdf5
    105/105 [==============================] - 8s 76ms/step - loss: 0.0403 - accuracy: 0.9851 - val_loss: 0.0828 - val_accuracy: 0.9767
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0420 - accuracy: 0.9842
    Epoch 00012: val_accuracy did not improve from 0.97666
    105/105 [==============================] - 8s 76ms/step - loss: 0.0418 - accuracy: 0.9842 - val_loss: 0.1219 - val_accuracy: 0.9679
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0377 - accuracy: 0.9865
    Epoch 00013: val_accuracy did not improve from 0.97666
    105/105 [==============================] - 8s 75ms/step - loss: 0.0378 - accuracy: 0.9865 - val_loss: 0.1568 - val_accuracy: 0.9568
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0291 - accuracy: 0.9894
    Epoch 00014: val_accuracy did not improve from 0.97666
    105/105 [==============================] - 8s 76ms/step - loss: 0.0289 - accuracy: 0.9895 - val_loss: 0.0882 - val_accuracy: 0.9728
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0369 - accuracy: 0.9866
    Epoch 00015: val_accuracy did not improve from 0.97666
    105/105 [==============================] - 8s 74ms/step - loss: 0.0371 - accuracy: 0.9865 - val_loss: 0.1560 - val_accuracy: 0.9630
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0347 - accuracy: 0.9873
    Epoch 00016: val_accuracy did not improve from 0.97666
    105/105 [==============================] - 8s 76ms/step - loss: 0.0345 - accuracy: 0.9873 - val_loss: 0.2414 - val_accuracy: 0.9482
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0321 - accuracy: 0.9880
    Epoch 00017: val_accuracy did not improve from 0.97666
    105/105 [==============================] - 8s 74ms/step - loss: 0.0322 - accuracy: 0.9880 - val_loss: 0.0981 - val_accuracy: 0.9744
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0301 - accuracy: 0.9891
    Epoch 00018: val_accuracy improved from 0.97666 to 0.98134, saving model to BEST-128x128-rstate-200-weights-improvement-18-0.98.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.0300 - accuracy: 0.9891 - val_loss: 0.0684 - val_accuracy: 0.9813
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0314 - accuracy: 0.9891
    Epoch 00019: val_accuracy did not improve from 0.98134
    105/105 [==============================] - 8s 76ms/step - loss: 0.0314 - accuracy: 0.9890 - val_loss: 0.1400 - val_accuracy: 0.9610
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0258 - accuracy: 0.9906
    Epoch 00020: val_accuracy did not improve from 0.98134
    105/105 [==============================] - 8s 77ms/step - loss: 0.0259 - accuracy: 0.9906 - val_loss: 0.0779 - val_accuracy: 0.9772
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0225 - accuracy: 0.9914
    Epoch 00021: val_accuracy did not improve from 0.98134
    105/105 [==============================] - 8s 77ms/step - loss: 0.0223 - accuracy: 0.9915 - val_loss: 0.1767 - val_accuracy: 0.9622
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0236 - accuracy: 0.9918
    Epoch 00022: val_accuracy did not improve from 0.98134
    105/105 [==============================] - 8s 75ms/step - loss: 0.0235 - accuracy: 0.9918 - val_loss: 0.3232 - val_accuracy: 0.9372
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0281 - accuracy: 0.9889
    Epoch 00023: val_accuracy did not improve from 0.98134
    105/105 [==============================] - 8s 78ms/step - loss: 0.0279 - accuracy: 0.9890 - val_loss: 0.4319 - val_accuracy: 0.9271
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0237 - accuracy: 0.9919
    Epoch 00024: val_accuracy did not improve from 0.98134
    105/105 [==============================] - 8s 76ms/step - loss: 0.0238 - accuracy: 0.9917 - val_loss: 0.1015 - val_accuracy: 0.9775
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0250 - accuracy: 0.9906
    Epoch 00025: val_accuracy improved from 0.98134 to 0.98585, saving model to BEST-128x128-rstate-200-weights-improvement-25-0.99.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.0251 - accuracy: 0.9906 - val_loss: 0.0398 - val_accuracy: 0.9859
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0233 - accuracy: 0.9913
    Epoch 00026: val_accuracy improved from 0.98585 to 0.99063, saving model to BEST-128x128-rstate-200-weights-improvement-26-0.99.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0232 - accuracy: 0.9913 - val_loss: 0.0280 - val_accuracy: 0.9906
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0230 - accuracy: 0.9914
    Epoch 00027: val_accuracy did not improve from 0.99063
    105/105 [==============================] - 8s 77ms/step - loss: 0.0228 - accuracy: 0.9915 - val_loss: 0.5903 - val_accuracy: 0.9217
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0177 - accuracy: 0.9936
    Epoch 00028: val_accuracy did not improve from 0.99063
    105/105 [==============================] - 8s 77ms/step - loss: 0.0175 - accuracy: 0.9936 - val_loss: 0.1366 - val_accuracy: 0.9619
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0239 - accuracy: 0.9914
    Epoch 00029: val_accuracy did not improve from 0.99063
    105/105 [==============================] - 8s 75ms/step - loss: 0.0242 - accuracy: 0.9913 - val_loss: 0.1020 - val_accuracy: 0.9743
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0177 - accuracy: 0.9933
    Epoch 00030: val_accuracy did not improve from 0.99063
    105/105 [==============================] - 8s 76ms/step - loss: 0.0177 - accuracy: 0.9933 - val_loss: 0.2130 - val_accuracy: 0.9576



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_53_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_53_2.png)



```python
acc3, val_acc3, loss3, val_loss3, epochs3 = do_a_run(rstate=300)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.2108 - accuracy: 0.9270
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-128x128-rstate-300-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 8s 74ms/step - loss: 0.2114 - accuracy: 0.9269 - val_loss: 1.2200 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1252 - accuracy: 0.9524
    Epoch 00002: val_accuracy did not improve from 0.82334
    105/105 [==============================] - 8s 75ms/step - loss: 0.1247 - accuracy: 0.9525 - val_loss: 1.6945 - val_accuracy: 0.8233
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1013 - accuracy: 0.9638
    Epoch 00003: val_accuracy improved from 0.82334 to 0.84350, saving model to BEST-128x128-rstate-300-weights-improvement-03-0.84.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.1014 - accuracy: 0.9637 - val_loss: 1.1477 - val_accuracy: 0.8435
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0808 - accuracy: 0.9693
    Epoch 00004: val_accuracy improved from 0.84350 to 0.84828, saving model to BEST-128x128-rstate-300-weights-improvement-04-0.85.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.0805 - accuracy: 0.9694 - val_loss: 0.9796 - val_accuracy: 0.8483
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0740 - accuracy: 0.9719
    Epoch 00005: val_accuracy improved from 0.84828 to 0.87816, saving model to BEST-128x128-rstate-300-weights-improvement-05-0.88.hdf5
    105/105 [==============================] - 8s 81ms/step - loss: 0.0744 - accuracy: 0.9716 - val_loss: 0.4725 - val_accuracy: 0.8782
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0645 - accuracy: 0.9773
    Epoch 00006: val_accuracy improved from 0.87816 to 0.96021, saving model to BEST-128x128-rstate-300-weights-improvement-06-0.96.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0645 - accuracy: 0.9773 - val_loss: 0.1105 - val_accuracy: 0.9602
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0624 - accuracy: 0.9775
    Epoch 00007: val_accuracy improved from 0.96021 to 0.98099, saving model to BEST-128x128-rstate-300-weights-improvement-07-0.98.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.0626 - accuracy: 0.9775 - val_loss: 0.0539 - val_accuracy: 0.9810
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0492 - accuracy: 0.9818
    Epoch 00008: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 75ms/step - loss: 0.0497 - accuracy: 0.9817 - val_loss: 0.2327 - val_accuracy: 0.9392
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0479 - accuracy: 0.9813
    Epoch 00009: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 76ms/step - loss: 0.0480 - accuracy: 0.9813 - val_loss: 0.1727 - val_accuracy: 0.9504
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0508 - accuracy: 0.9810
    Epoch 00010: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 77ms/step - loss: 0.0506 - accuracy: 0.9810 - val_loss: 0.0970 - val_accuracy: 0.9683
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0445 - accuracy: 0.9841
    Epoch 00011: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 78ms/step - loss: 0.0447 - accuracy: 0.9840 - val_loss: 0.7815 - val_accuracy: 0.8891
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0391 - accuracy: 0.9852
    Epoch 00012: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 76ms/step - loss: 0.0393 - accuracy: 0.9852 - val_loss: 0.1548 - val_accuracy: 0.9632
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0422 - accuracy: 0.9847
    Epoch 00013: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 75ms/step - loss: 0.0419 - accuracy: 0.9848 - val_loss: 0.1799 - val_accuracy: 0.9449
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0392 - accuracy: 0.9859
    Epoch 00014: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 73ms/step - loss: 0.0390 - accuracy: 0.9860 - val_loss: 0.2139 - val_accuracy: 0.9490
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0355 - accuracy: 0.9859
    Epoch 00015: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 73ms/step - loss: 0.0359 - accuracy: 0.9858 - val_loss: 0.2589 - val_accuracy: 0.9424
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0411 - accuracy: 0.9847
    Epoch 00016: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 74ms/step - loss: 0.0408 - accuracy: 0.9848 - val_loss: 0.4071 - val_accuracy: 0.9274
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0335 - accuracy: 0.9874
    Epoch 00017: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 72ms/step - loss: 0.0333 - accuracy: 0.9874 - val_loss: 0.1752 - val_accuracy: 0.9523
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0334 - accuracy: 0.9877
    Epoch 00018: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 74ms/step - loss: 0.0338 - accuracy: 0.9875 - val_loss: 0.1719 - val_accuracy: 0.9532
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0321 - accuracy: 0.9884
    Epoch 00019: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 73ms/step - loss: 0.0323 - accuracy: 0.9884 - val_loss: 0.1880 - val_accuracy: 0.9580
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0268 - accuracy: 0.9901
    Epoch 00020: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 79ms/step - loss: 0.0270 - accuracy: 0.9901 - val_loss: 0.0736 - val_accuracy: 0.9761
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0293 - accuracy: 0.9891
    Epoch 00021: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 72ms/step - loss: 0.0294 - accuracy: 0.9890 - val_loss: 0.3657 - val_accuracy: 0.9424
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0246 - accuracy: 0.9916
    Epoch 00022: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 74ms/step - loss: 0.0244 - accuracy: 0.9917 - val_loss: 0.0752 - val_accuracy: 0.9781
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0307 - accuracy: 0.9890
    Epoch 00023: val_accuracy did not improve from 0.98099
    105/105 [==============================] - 8s 77ms/step - loss: 0.0306 - accuracy: 0.9890 - val_loss: 0.1177 - val_accuracy: 0.9670
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0251 - accuracy: 0.9910
    Epoch 00024: val_accuracy improved from 0.98099 to 0.98559, saving model to BEST-128x128-rstate-300-weights-improvement-24-0.99.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0249 - accuracy: 0.9911 - val_loss: 0.0431 - val_accuracy: 0.9856
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0206 - accuracy: 0.9925
    Epoch 00025: val_accuracy did not improve from 0.98559
    105/105 [==============================] - 8s 75ms/step - loss: 0.0206 - accuracy: 0.9925 - val_loss: 0.3295 - val_accuracy: 0.9347
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0234 - accuracy: 0.9916
    Epoch 00026: val_accuracy did not improve from 0.98559
    105/105 [==============================] - 8s 75ms/step - loss: 0.0233 - accuracy: 0.9916 - val_loss: 0.2307 - val_accuracy: 0.9557
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0244 - accuracy: 0.9916
    Epoch 00027: val_accuracy did not improve from 0.98559
    105/105 [==============================] - 8s 76ms/step - loss: 0.0244 - accuracy: 0.9916 - val_loss: 0.2103 - val_accuracy: 0.9538
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0222 - accuracy: 0.9923
    Epoch 00028: val_accuracy did not improve from 0.98559
    105/105 [==============================] - 8s 75ms/step - loss: 0.0220 - accuracy: 0.9924 - val_loss: 0.1123 - val_accuracy: 0.9696
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0224 - accuracy: 0.9923
    Epoch 00029: val_accuracy did not improve from 0.98559
    105/105 [==============================] - 8s 77ms/step - loss: 0.0224 - accuracy: 0.9923 - val_loss: 0.0523 - val_accuracy: 0.9814
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0197 - accuracy: 0.9929
    Epoch 00030: val_accuracy did not improve from 0.98559
    105/105 [==============================] - 8s 75ms/step - loss: 0.0196 - accuracy: 0.9930 - val_loss: 0.0654 - val_accuracy: 0.9836



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_54_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_54_2.png)



```python
acc4, val_acc4, loss4, val_loss4, epochs4 = do_a_run(rstate=400)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.2114 - accuracy: 0.9258
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-128x128-rstate-400-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 8s 74ms/step - loss: 0.2101 - accuracy: 0.9261 - val_loss: 0.6738 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1207 - accuracy: 0.9552
    Epoch 00002: val_accuracy did not improve from 0.82334
    105/105 [==============================] - 8s 74ms/step - loss: 0.1209 - accuracy: 0.9552 - val_loss: 1.2864 - val_accuracy: 0.8233
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1013 - accuracy: 0.9625
    Epoch 00003: val_accuracy improved from 0.82334 to 0.83581, saving model to BEST-128x128-rstate-400-weights-improvement-03-0.84.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.1010 - accuracy: 0.9625 - val_loss: 0.9347 - val_accuracy: 0.8358
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0781 - accuracy: 0.9706
    Epoch 00004: val_accuracy did not improve from 0.83581
    105/105 [==============================] - 8s 74ms/step - loss: 0.0777 - accuracy: 0.9706 - val_loss: 1.0977 - val_accuracy: 0.8249
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0677 - accuracy: 0.9749
    Epoch 00005: val_accuracy improved from 0.83581 to 0.85447, saving model to BEST-128x128-rstate-400-weights-improvement-05-0.85.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.0681 - accuracy: 0.9748 - val_loss: 0.7339 - val_accuracy: 0.8545
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0608 - accuracy: 0.9778
    Epoch 00006: val_accuracy improved from 0.85447 to 0.93431, saving model to BEST-128x128-rstate-400-weights-improvement-06-0.93.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.0615 - accuracy: 0.9777 - val_loss: 0.2168 - val_accuracy: 0.9343
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0571 - accuracy: 0.9782
    Epoch 00007: val_accuracy improved from 0.93431 to 0.96561, saving model to BEST-128x128-rstate-400-weights-improvement-07-0.97.hdf5
    105/105 [==============================] - 8s 76ms/step - loss: 0.0572 - accuracy: 0.9782 - val_loss: 0.1044 - val_accuracy: 0.9656
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0584 - accuracy: 0.9795
    Epoch 00008: val_accuracy did not improve from 0.96561
    105/105 [==============================] - 8s 74ms/step - loss: 0.0587 - accuracy: 0.9793 - val_loss: 0.1783 - val_accuracy: 0.9518
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0508 - accuracy: 0.9806
    Epoch 00009: val_accuracy did not improve from 0.96561
    105/105 [==============================] - 8s 76ms/step - loss: 0.0509 - accuracy: 0.9806 - val_loss: 0.2935 - val_accuracy: 0.9333
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0452 - accuracy: 0.9841
    Epoch 00010: val_accuracy did not improve from 0.96561
    105/105 [==============================] - 8s 77ms/step - loss: 0.0453 - accuracy: 0.9840 - val_loss: 0.6542 - val_accuracy: 0.8943
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0411 - accuracy: 0.9851
    Epoch 00011: val_accuracy did not improve from 0.96561
    105/105 [==============================] - 8s 77ms/step - loss: 0.0410 - accuracy: 0.9851 - val_loss: 0.7955 - val_accuracy: 0.8749
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0397 - accuracy: 0.9848
    Epoch 00012: val_accuracy improved from 0.96561 to 0.97091, saving model to BEST-128x128-rstate-400-weights-improvement-12-0.97.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.0396 - accuracy: 0.9849 - val_loss: 0.1070 - val_accuracy: 0.9709
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0340 - accuracy: 0.9869
    Epoch 00013: val_accuracy did not improve from 0.97091
    105/105 [==============================] - 8s 78ms/step - loss: 0.0339 - accuracy: 0.9869 - val_loss: 0.1447 - val_accuracy: 0.9614
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0328 - accuracy: 0.9879
    Epoch 00014: val_accuracy did not improve from 0.97091
    105/105 [==============================] - 8s 75ms/step - loss: 0.0327 - accuracy: 0.9879 - val_loss: 0.1207 - val_accuracy: 0.9649
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0364 - accuracy: 0.9868
    Epoch 00015: val_accuracy did not improve from 0.97091
    105/105 [==============================] - 8s 76ms/step - loss: 0.0367 - accuracy: 0.9868 - val_loss: 0.1297 - val_accuracy: 0.9657
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0354 - accuracy: 0.9873
    Epoch 00016: val_accuracy did not improve from 0.97091
    105/105 [==============================] - 8s 74ms/step - loss: 0.0359 - accuracy: 0.9871 - val_loss: 0.5750 - val_accuracy: 0.9108
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0299 - accuracy: 0.9897
    Epoch 00017: val_accuracy did not improve from 0.97091
    105/105 [==============================] - 8s 76ms/step - loss: 0.0297 - accuracy: 0.9898 - val_loss: 0.2040 - val_accuracy: 0.9515
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0302 - accuracy: 0.9892
    Epoch 00018: val_accuracy improved from 0.97091 to 0.98161, saving model to BEST-128x128-rstate-400-weights-improvement-18-0.98.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0302 - accuracy: 0.9891 - val_loss: 0.0587 - val_accuracy: 0.9816
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0287 - accuracy: 0.9890
    Epoch 00019: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 76ms/step - loss: 0.0286 - accuracy: 0.9891 - val_loss: 0.2468 - val_accuracy: 0.9436
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0296 - accuracy: 0.9891
    Epoch 00020: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 77ms/step - loss: 0.0296 - accuracy: 0.9892 - val_loss: 0.0865 - val_accuracy: 0.9752
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0242 - accuracy: 0.9913
    Epoch 00021: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 75ms/step - loss: 0.0245 - accuracy: 0.9911 - val_loss: 0.1449 - val_accuracy: 0.9618
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0258 - accuracy: 0.9906
    Epoch 00022: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 76ms/step - loss: 0.0261 - accuracy: 0.9905 - val_loss: 0.5683 - val_accuracy: 0.9233
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0194 - accuracy: 0.9925
    Epoch 00023: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 74ms/step - loss: 0.0194 - accuracy: 0.9925 - val_loss: 0.1664 - val_accuracy: 0.9540
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0238 - accuracy: 0.9917
    Epoch 00024: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 74ms/step - loss: 0.0237 - accuracy: 0.9918 - val_loss: 0.2047 - val_accuracy: 0.9479
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0235 - accuracy: 0.9914
    Epoch 00025: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 75ms/step - loss: 0.0234 - accuracy: 0.9914 - val_loss: 0.1150 - val_accuracy: 0.9686
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0209 - accuracy: 0.9927
    Epoch 00026: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 74ms/step - loss: 0.0209 - accuracy: 0.9927 - val_loss: 0.3332 - val_accuracy: 0.9450
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0216 - accuracy: 0.9916
    Epoch 00027: val_accuracy did not improve from 0.98161
    105/105 [==============================] - 8s 75ms/step - loss: 0.0220 - accuracy: 0.9915 - val_loss: 0.2184 - val_accuracy: 0.9490
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0253 - accuracy: 0.9907
    Epoch 00028: val_accuracy improved from 0.98161 to 0.98329, saving model to BEST-128x128-rstate-400-weights-improvement-28-0.98.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0254 - accuracy: 0.9906 - val_loss: 0.0553 - val_accuracy: 0.9833
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0205 - accuracy: 0.9925
    Epoch 00029: val_accuracy did not improve from 0.98329
    105/105 [==============================] - 8s 77ms/step - loss: 0.0204 - accuracy: 0.9926 - val_loss: 0.0530 - val_accuracy: 0.9824
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0218 - accuracy: 0.9921
    Epoch 00030: val_accuracy did not improve from 0.98329
    105/105 [==============================] - 8s 77ms/step - loss: 0.0218 - accuracy: 0.9920 - val_loss: 0.1278 - val_accuracy: 0.9660



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_55_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_55_2.png)



```python
acc5, val_acc5, loss5, val_loss5, epochs5 = do_a_run(rstate=500)
```

    Epoch 1/30
    104/105 [============================>.] - ETA: 0s - loss: 0.2071 - accuracy: 0.9265
    Epoch 00001: val_accuracy improved from -inf to 0.82334, saving model to BEST-128x128-rstate-500-weights-improvement-01-0.82.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.2066 - accuracy: 0.9266 - val_loss: 0.8865 - val_accuracy: 0.8233
    Epoch 2/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1182 - accuracy: 0.9564
    Epoch 00002: val_accuracy did not improve from 0.82334
    105/105 [==============================] - 8s 74ms/step - loss: 0.1178 - accuracy: 0.9566 - val_loss: 1.5336 - val_accuracy: 0.8233
    Epoch 3/30
    104/105 [============================>.] - ETA: 0s - loss: 0.1007 - accuracy: 0.9619
    Epoch 00003: val_accuracy improved from 0.82334 to 0.83351, saving model to BEST-128x128-rstate-500-weights-improvement-03-0.83.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.1005 - accuracy: 0.9620 - val_loss: 1.1236 - val_accuracy: 0.8335
    Epoch 4/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0796 - accuracy: 0.9714
    Epoch 00004: val_accuracy improved from 0.83351 to 0.83554, saving model to BEST-128x128-rstate-500-weights-improvement-04-0.84.hdf5
    105/105 [==============================] - 8s 76ms/step - loss: 0.0792 - accuracy: 0.9716 - val_loss: 1.3047 - val_accuracy: 0.8355
    Epoch 5/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0746 - accuracy: 0.9727
    Epoch 00005: val_accuracy improved from 0.83554 to 0.87321, saving model to BEST-128x128-rstate-500-weights-improvement-05-0.87.hdf5
    105/105 [==============================] - 8s 79ms/step - loss: 0.0754 - accuracy: 0.9724 - val_loss: 0.5000 - val_accuracy: 0.8732
    Epoch 6/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0618 - accuracy: 0.9767
    Epoch 00006: val_accuracy improved from 0.87321 to 0.95871, saving model to BEST-128x128-rstate-500-weights-improvement-06-0.96.hdf5
    105/105 [==============================] - 8s 78ms/step - loss: 0.0614 - accuracy: 0.9768 - val_loss: 0.1270 - val_accuracy: 0.9587
    Epoch 7/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0579 - accuracy: 0.9786
    Epoch 00007: val_accuracy improved from 0.95871 to 0.97940, saving model to BEST-128x128-rstate-500-weights-improvement-07-0.98.hdf5
    105/105 [==============================] - 8s 77ms/step - loss: 0.0577 - accuracy: 0.9786 - val_loss: 0.0591 - val_accuracy: 0.9794
    Epoch 8/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0514 - accuracy: 0.9818
    Epoch 00008: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 77ms/step - loss: 0.0514 - accuracy: 0.9819 - val_loss: 0.2102 - val_accuracy: 0.9459
    Epoch 9/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0525 - accuracy: 0.9820
    Epoch 00009: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 76ms/step - loss: 0.0523 - accuracy: 0.9821 - val_loss: 0.2622 - val_accuracy: 0.9399
    Epoch 10/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0443 - accuracy: 0.9842
    Epoch 00010: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0441 - accuracy: 0.9842 - val_loss: 0.1536 - val_accuracy: 0.9595
    Epoch 11/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0378 - accuracy: 0.9860
    Epoch 00011: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0378 - accuracy: 0.9860 - val_loss: 0.2740 - val_accuracy: 0.9347
    Epoch 12/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0367 - accuracy: 0.9868
    Epoch 00012: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0364 - accuracy: 0.9869 - val_loss: 0.1090 - val_accuracy: 0.9620
    Epoch 13/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0371 - accuracy: 0.9860
    Epoch 00013: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0375 - accuracy: 0.9860 - val_loss: 0.6898 - val_accuracy: 0.9020
    Epoch 14/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0358 - accuracy: 0.9868
    Epoch 00014: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 77ms/step - loss: 0.0359 - accuracy: 0.9866 - val_loss: 0.1472 - val_accuracy: 0.9637
    Epoch 15/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0337 - accuracy: 0.9873
    Epoch 00015: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0335 - accuracy: 0.9874 - val_loss: 0.5626 - val_accuracy: 0.9160
    Epoch 16/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0287 - accuracy: 0.9887
    Epoch 00016: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0284 - accuracy: 0.9888 - val_loss: 0.1211 - val_accuracy: 0.9646
    Epoch 17/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0356 - accuracy: 0.9871
    Epoch 00017: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0353 - accuracy: 0.9872 - val_loss: 0.0953 - val_accuracy: 0.9745
    Epoch 18/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0241 - accuracy: 0.9912
    Epoch 00018: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0243 - accuracy: 0.9910 - val_loss: 0.2424 - val_accuracy: 0.9530
    Epoch 19/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0300 - accuracy: 0.9893
    Epoch 00019: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 76ms/step - loss: 0.0297 - accuracy: 0.9894 - val_loss: 0.1607 - val_accuracy: 0.9574
    Epoch 20/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0259 - accuracy: 0.9908
    Epoch 00020: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 73ms/step - loss: 0.0258 - accuracy: 0.9907 - val_loss: 0.2085 - val_accuracy: 0.9463
    Epoch 21/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0302 - accuracy: 0.9899
    Epoch 00021: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 76ms/step - loss: 0.0309 - accuracy: 0.9898 - val_loss: 0.0743 - val_accuracy: 0.9766
    Epoch 22/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0257 - accuracy: 0.9903
    Epoch 00022: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0257 - accuracy: 0.9903 - val_loss: 0.1039 - val_accuracy: 0.9709
    Epoch 23/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0228 - accuracy: 0.9922
    Epoch 00023: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 77ms/step - loss: 0.0227 - accuracy: 0.9923 - val_loss: 0.3321 - val_accuracy: 0.9376
    Epoch 24/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0211 - accuracy: 0.9919
    Epoch 00024: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0211 - accuracy: 0.9919 - val_loss: 0.2066 - val_accuracy: 0.9501
    Epoch 25/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0228 - accuracy: 0.9913
    Epoch 00025: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 75ms/step - loss: 0.0227 - accuracy: 0.9914 - val_loss: 0.1135 - val_accuracy: 0.9746
    Epoch 26/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0214 - accuracy: 0.9918
    Epoch 00026: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 77ms/step - loss: 0.0213 - accuracy: 0.9918 - val_loss: 0.1181 - val_accuracy: 0.9683
    Epoch 27/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0202 - accuracy: 0.9925
    Epoch 00027: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 76ms/step - loss: 0.0206 - accuracy: 0.9924 - val_loss: 0.3925 - val_accuracy: 0.9373
    Epoch 28/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0230 - accuracy: 0.9918
    Epoch 00028: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 74ms/step - loss: 0.0232 - accuracy: 0.9917 - val_loss: 0.0885 - val_accuracy: 0.9755
    Epoch 29/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0265 - accuracy: 0.9901
    Epoch 00029: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 76ms/step - loss: 0.0264 - accuracy: 0.9901 - val_loss: 0.3425 - val_accuracy: 0.9359
    Epoch 30/30
    104/105 [============================>.] - ETA: 0s - loss: 0.0206 - accuracy: 0.9923
    Epoch 00030: val_accuracy did not improve from 0.97940
    105/105 [==============================] - 8s 77ms/step - loss: 0.0206 - accuracy: 0.9923 - val_loss: 0.2286 - val_accuracy: 0.9477



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_56_1.png)



![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_56_2.png)



```python
val_accs = np.vstack([val_acc1,val_acc2,val_acc3,val_acc4,val_acc5])
```

Taking the average validation accuracies of the 5 models trained on different realizations of the train-validation split, we find that the maximum accuracy it can classify across the 10 classes is `97.04%`.


```python
np.max(np.sum(val_accs, axis = 0)/5)
```




    0.9704156



Taking a look at the confusion matrix of one of the realizations, we can further assess the performance of the model.


```python
model.load_weights("BEST-64x64-rstate-100-weights-improvement-30-0.98.hdf5")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])
```


```python
from sklearn.metrics import confusion_matrix

all_y_test = []
all_y_pred = []
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 0)
y_pred = np.rint(model.predict(x_test))

all_y_test = all_y_test + list(y_test)
all_y_pred = all_y_pred + list(y_pred)

confmat = confusion_matrix(np.array(all_y_test).argmax(axis=1), np.array(all_y_pred).argmax(axis=1))
```


```python
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
```


![png](Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_files/Take%20it%20or%20Leaf%20it%21%20%28LT%2013%29_63_0.png)



```python
pcc
```




    {'Pepper_bell__Bacterial_spot': 499,
     'Pepper_bell__healthy': 499,
     'Potato__Early_blight': 500,
     'Potato__Late_blight': 500,
     'Potato__healthy': 152,
     'Tomato__Bacterial_spot': 500,
     'Tomato__Healthy': 500,
     'Tomato__Target_Spot': 500,
     'Tomato__Tomato_YellowLeaf__Curl_Virus': 499,
     'Tomato__Tomato_mosaic_virus': 373}



The misclassifications from the 64x64 model showed a similar trend as the 256x256 model when handling tomato leaves with target spot. This presents similar problems to before but are less pronounced as there are less misclassifications for target spot overall. 

The model failed to differentiate well between early blight and late blight for potatoes. The appearance of the two diseases are very similar with the difference mainly being the shape of their affected area in a leaf. Due to the downsampling of the image, edges became less defined which could have resulted in poorer differentiation between the two closely related classes.

### Conclusion

Deep learning models have shown great predictive power when classifying plants and their associated diseases. In this study, the models developed performed well in identifying plant-disease pairs with accuracies around 97%. As was demonstrated, using high-resolution (256x256) images provided marginally superior performance to low-resolution (64x64) images. When it comes to weighing the benefits and costs of equipment to take high resolution images opposed to lower resolution ones, this minor difference in performance may not be as important. Given the nature of misclassifications, more work is required to fine-tune improve the accuracy of the models.

### Acknowledgements
This project was completed together with my learning teammates and co-authors Gilbert Chua, Roy Roberto, and Jishu Basak. We would like to thank Dr. Christopher Monterola and Dr. Erika Legara in guiding us through this learning experience.
