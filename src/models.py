'''
Models
Define the different NN models we will use
Author: Tawn Kramer
'''
from __future__ import print_function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, Lambda, ELU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.optimizers import Adadelta, Adam
# Alexnet


import conf

def show_model_summary(model):
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)

def get_nvidia_model1(num_outputs):
    '''
    this model is inspired by the NVIDIA paper
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Activation is RELU
    '''
    row, col, ch = conf.row, conf.col, conf.ch
    
    drop = 0.1
    
    img_in = Input(shape=(row, col, ch), name='img_in')
    x = img_in
    # x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    # x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Lambda(lambda x: x/255.0)(x)
    x = Conv2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Conv2D(32, (5,5), strides=(2,2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (5,5), strides=(2,2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    #x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    #x = Dropout(drop)(x)

    outputs = []
    outputs.append(Dense(num_outputs, activation='linear', name='steering_throttle')(x))
    
        
    model = Model(inputs=[img_in], outputs=outputs)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=['acc'])
    return model


def get_nvidia_model2(num_outputs):
    '''
    this model is also inspired by the NVIDIA paper
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    but taken from
    https://github.com/naokishibuya/car-behavioral-cloning/blob/master/model.py
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', strides=2))
    model.add(Conv2D(36, 5, 5, activation='elu', strides=2))
    model.add(Conv2D(48, 5, 5, activation='elu', strides=2))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    NB Tawn Kramer's model uses dropout = 0.1 on five layers, Naoki uses
    0.5 on a single layer
    '''
    row, col, ch = conf.row, conf.col, conf.ch

    drop = 0.5

    img_in = Input(shape=(row, col, ch), name='img_in')
    x = img_in
    # x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    # x = Lambda(lambda x: x/127.5 - 1.0)(x) # normalize and re-center
    x = Lambda(lambda x: x / 255.0)(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu', name="conv2d_1")(x)
    # x = Dropout(drop)(x)
    x = Conv2D(32, (5, 5), strides=(2, 2), activation='elu', name="conv2d_2")(x)
    #x = Dropout(drop)(x)
    x = Conv2D(64, (5, 5), strides=(2, 2), activation='elu', name="conv2d_3")(x)
    #x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), activation='elu', name="conv2d_4")(x) # default strides=(1,1)
    #x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), activation='elu', name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)

    x = Dense(100, activation='elu')(x)
    # x = Dropout(drop)(x)
    x = Dense(50, activation='elu')(x)
    # x = Dropout(drop)(x)
    x = Dense(10, activation='elu')(x) # Added in Naoki's model

    outputs = []
    # outputs.append(Dense(num_outputs, activation='linear', name='steering_throttle')(x))
    outputs.append(Dense(num_outputs, name='steering_throttle')(x))

    model = Model(inputs=[img_in], outputs=outputs)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=['acc'])
    return model
"""
def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet
"""