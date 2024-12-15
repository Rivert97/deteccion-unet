# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.utils import plot_model
import cv2

"""# Model"""

def unet(sz = (256, 256, 3)):
    x = Input(sz)
    inputs = x

    #down sampling
    f = 8
    layers = []

    for i in range(0, 6):
        #x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = Conv2D(f, 3, padding='same') (x)
        x = LeakyReLU()(x)
        #x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = Conv2D(f, 3, padding='same') (x)
        x = LeakyReLU()(x)
        layers.append(x)
        #x = Conv2D(f, 1, strides=2, activation='relu', padding='same') (x)
        x = Conv2D(f, 1, strides=2, padding='same') (x)
        x = LeakyReLU()(x)
        f = f*2
    ff2 = 64

    #bottleneck
    j = len(layers) - 1
    #x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, padding='same') (x)
    x = LeakyReLU()(x)
    #x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1

    #upsampling
    for i in range(0, 5):
        ff2 = ff2//2
        f = f // 2
        #x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = Conv2D(f, 3, padding='same') (x)
        x = LeakyReLU()(x)
        #x = Conv2D(f, 3, activation='relu', padding='same') (x)
        x = Conv2D(f, 3, padding='same') (x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j -1


    #classification
    #x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, padding='same') (x)
    x = LeakyReLU()(x)
    #x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, padding='same') (x)
    x = LeakyReLU()(x)
    outputs = Conv2D(1, 1, activation='sigmoid') (x)

    #model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [mean_iou])

    model.summary()

    return model

def detect_from_image(filename, use_contours=True):
    """# Loading"""
    model = unet()
    model.load_weights('unet.weights.h5')
    #plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=True, rankdir='LR', expand_nested=False, dpi=96)

    raw = Image.open(filename)
    org_shape = raw.size
    raw = np.array(raw.resize((256, 256)))/255.
    raw = raw[:,:,0:3]

    #predict the mask
    pred = model.predict(np.expand_dims(raw, 0), verbose=False)

    #mask post-processing
    msk  = pred.squeeze()
    msk = np.stack((msk,)*3, axis=-1)
    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0

    #show the mask and the segmented image
    combined = np.concatenate([raw, msk, raw* msk], axis = 1)
    plt.axis('off')
    plt.imshow(combined)
    plt.show()

    raw = draw_rectangles(raw, msk[:, :, 0], use_contours)

    raw = cv2.resize(raw, org_shape)
    plt.axis('off')
    plt.imshow(raw)
    plt.show()

def detect_from_camera(use_contours=True):
    """# Loading"""
    model = unet()
    model.load_weights('unet.weights.h5')

    cap = cv2.VideoCapture(0)

    while True:
        prevtime = time.time()
        ret, frame = cap.read()

        if not ret:
            print("No se pudo capturar frame")
            break

        raw = Image.fromarray(frame)
        org_shape = raw.size
        raw = np.array(raw.resize((256, 256)))/255.
        raw = raw[:,:,0:3]

        #predict the mask
        pred = model.predict(np.expand_dims(raw, 0), verbose=False)

        #mask post-processing
        msk  = pred.squeeze()
        msk = np.stack((msk,)*3, axis=-1)
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0

        raw = draw_rectangles(raw, msk[:, :, 0], use_contours)

        raw = cv2.resize(raw, org_shape)

        cv2.imshow("Frame", raw)

        if cv2.waitKey(1) == ord('q'):
            break
        
        #print("FPS:", 1.0/(time.time() - prevtime))
        prevtime = time.time()

def draw_rectangles(raw, mask, use_contours=True):
    if use_contours:
        msk = mask.astype('uint8')
        contours, hierarchy = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Ignore small contours
            if area < 500:
                continue

            x,y,w,h = cv2.boundingRect(cnt)
            raw = cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
    else:
        # Find bounding box
        whiteidx = np.where(mask == 1)
        if len(whiteidx[0]) > 0 and len(whiteidx[1]) > 0:
            box = ((min(whiteidx[1]), min(whiteidx[0])), (max(whiteidx[1]), max(whiteidx[0])))
            cv2.rectangle(raw, box[0], box[1], (0, 255, 0), 2, cv2.LINE_AA)
    
    return raw

if __name__ == '__main__':
    detect_from_image("test.jpg", use_contours=True)
    #detect_from_camera(use_contours=True)