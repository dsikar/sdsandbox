'''
Models
Define the different NN models we will use
Author: Tawn Kramer
'''
from __future__ import print_function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, Lambda, ELU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from tensorflow.keras import initializers, regularizers
# Alexnet


import conf

def show_model_summary(model):
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)

def nvidia_baseline(num_outputs):
    '''
    this model is approximately equal to:
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Although nothing is said about dropout or activation, which is assumed to be RELU

    Hi Daniel,

    We used the following settings (we haven't documented them in any publication):

    loss function: MSE
    optimizer: adadelta
    learning rate: 1e-4 (but not really used in adadelta)
    dropout: 0.25

    Best regards,
    Urs
    '''
    # note row and col values are now from _NET
    # Adjust sizes accordingly in conf.py
    row, col, ch = conf.row, conf.col, conf.ch

    drop = 0.1 # spreading droupout
    # batch_init = initializers.glorot_uniform #  Original AlexNet initializers.RandomNormal(mean=0., stddev=0.01);
    img_in = Input(shape=(row, col, ch), name='img_in')
    x = img_in
    # RGB values assumed to be normalized and not centered i.e. x/127.5 - 1.
    x = Lambda(lambda x: x / 255.0)(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', name="conv2d_2")(x) #2nd
    x = Dropout(drop)(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_4")(x) # default strides=(1,1) # 4th
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_5")(x) #5th
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    # x = Dense(1164, activation='relu', name="dense_1", kernel_initializer=batch_init, bias_initializer='ones')(x)
    #x = Dropout(drop)(x)
    x = Dense(100, activation='relu', name="dense_2")(x)
    #x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name="dense_3")(x) # Added in Naoki's model
    #x = Dropout(drop)(x)
    # x = Dense(10, activation='relu', name="dense_4", kernel_initializer=batch_init, bias_initializer='zeros')(x)
    #x = Dropout(drop)(x)
    outputs = []
    # outputs.append(Dense(num_outputs, activation='linear', name='steering_throttle')(x))
    outputs.append(Dense(num_outputs, activation='linear', name='steering')(x))

    model = Model(inputs=[img_in], outputs=outputs)
    # opt = Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
    opt = Adam(lr=conf.learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=['acc'])

    # add weight decay
    # https://stackoverflow.com/questions/41260042/global-weight-decay-in-keras
    #alpha = 0.0005  # weight decay coefficient
    #for layer in model.layers:
    #    if isinstance(layer, Conv2D) or isinstance(layer, Dense):
    #        layer.add_loss(lambda: regularizers.l2(alpha)(layer.kernel))
    #    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
    #        layer.add_loss(lambda: regularizers.l2(alpha)(layer.bias))
    return model

def nvidia_model1(num_outputs):
    '''
    This model expects image input size 160hx120w
    this model is inspired by the NVIDIA paper
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Activation is RELU
    '''
    # row, col, ch = conf.row, conf.col, conf.ch
    # better albeit less readable
    row, col, ch = conf.nvidia1_img_dims[conf.IMG_HEIGHT_NET_IDX], conf.nvidia1_img_dims[conf.IMG_WIDTH_NET_IDX], conf.nvidia1_img_dims[conf.IMG_DEPTH_IDX]
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
    x = Dense(1064, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    #x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    #x = Dropout(drop)(x)
    x = Dense(10, activation='relu')(x)

    outputs = []
    outputs.append(Dense(num_outputs, activation='linear', name='steering_throttle')(x))
    
        
    model = Model(inputs=[img_in], outputs=outputs)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=['acc'])
    # might want to try metrics=['acc', 'loss']  https://stackoverflow.com/questions/51047676/how-to-get-accuracy-of-model-using-keras
    return model


def nvidia_model2(num_outputs):
    '''
    A.K.A. NaokiNet - https://github.com/naokishibuya/car-behavioral-cloning/blob/master/model.py
    This model expects images of size 66,200,3
    '''
    # row, col, ch = conf.row, conf.col, conf.ch
    row, col, ch = conf.nvidia2_img_dims[conf.IMG_HEIGHT_NET_IDX], conf.nvidia2_img_dims[conf.IMG_WIDTH_NET_IDX], \
                   conf.nvidia2_img_dims[conf.IMG_DEPTH_IDX]

    drop = 0.1

    img_in = Input(shape=(row, col, ch), name='img_in')
    x = img_in
    # x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    # x = Lambda(lambda x: x/127.5 - 1.0)(x) # normalize and re-center
    x = Lambda(lambda x: x / 255.0)(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Conv2D(32, (5, 5), strides=(2, 2), activation='elu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='elu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), activation='elu', name="conv2d_4")(x) # default strides=(1,1)
    x = Dropout(drop)(x)
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

def nvidia_model3(num_outputs):
    '''
    This model expects images of size 66,200,3
    '''
    row, col, ch = conf.row, conf.col, conf.ch

    drop = 0.1

    img_in = Input(shape=(row, col, ch), name='img_in')
    x = img_in
    # x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    x = Lambda(lambda x: x/127.5 - 1.0)(x) # normalize and re-center
    # x = Lambda(lambda x: x / 255.0)(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu', name="conv2d_1")(x)
    # x = Dropout(drop)(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='elu', name="conv2d_2")(x)
    #x = Dropout(drop)(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='elu', name="conv2d_3")(x)
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

def get_alexnet(num_outputs):
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
    #row, col, ch = conf.image_width_alexnet, conf.image_height_alexnet, conf.ch
    row, col, ch = conf.alexnet_img_dims[conf.IMG_HEIGHT_NET_IDX], conf.nvidia2_img_dims[conf.IMG_WIDTH_NET_IDX], \
    conf.nvidia2_img_dims[conf.IMG_DEPTH_IDX]
    drop = 0.5
    # read https://stackoverflow.com/questions/58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte
    # to work out shapes
    img_in = Input(shape=(row, col, ch), name='img_in')
    x = img_in
    # x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    x = Lambda(lambda x: x/127.5 - 1.0)(x) # normalize and re-center
    # x = Lambda(lambda x: x / 255.0)(x)
    x = Conv2D(48, (8, 8), strides=(4, 4), padding='valid', activation='relu', name="conv2d_1")(x)
    # x = Dropout(drop)(x)
    x = MaxPooling2D(48, (1, 1), padding="same", name="maxpool2d_1")(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', activation='relu', name="conv2d_2")(x)
    #x = Dropout(drop)(x)
    x = MaxPooling2D(128, (1, 1), padding="same", name="maxpool2d_2")(x)

    x = Conv2D(192, (3, 3), strides=(2, 2), padding='valid', activation='relu', name="conv2d_3")(x)
    #x = Dropout(drop)(x)
    x = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2d_4")(x) # default strides=(1,1)
    #x = Dropout(drop)(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2d_5")(x)

    x = MaxPooling2D(128, (1, 1), padding="same", name="maxpool2d_3")(x)

    #x = Conv2D(64, (3, 3), activation='relu', name="conv2d_6")(x)
    # error Negative dimension size caused by subtracting 128 from 10 for '{{node max_pooling2d/MaxPool}} = MaxPool[T=DT_FLOAT,
    # data_format="NHWC", ksize=[1, 128, 128, 1], padding="VALID", strides=[1, 3, 3, 1]](conv2d_4/Identity)'
    # with input shapes: [?,10,10,192].
    # x = MaxPooling2D(128, (3, 3), padding="SAME")(x)
    # By commenting out line above, error is:
    #  Input to reshape is a tensor with 1843200 values, but the requested shape requires a multiple of 101568
    # 	 [[node model/flattened/Reshape (defined at /git/sdsandbox/src/train.py:272) ]] [Op:__inference_train_function_1081]

    x = Dropout(drop)(x)

    """
        x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu', name="conv2d_1")(x)
    # x = Dropout(drop)(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='elu', name="conv2d_2")(x)
    #x = Dropout(drop)(x)
    x = Conv2D(64, (5, 5), strides=(2, 2), activation='elu', name="conv2d_3")(x)
    #x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), activation='elu', name="conv2d_4")(x) # default strides=(1,1)
    #x = Dropout(drop)(x)
    x = MaxPooling2D(64, (3, 3), name="maxpool2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)

    x = Dense(2048, activation='elu')(x)
    # x = Dropout(drop)(x)
    x = Dense(2048, activation='elu')(x)
    # x = Dropout(drop)(x)
    # x = Dense(10, activation='elu')(x) # Added in Naoki's model
    """

    x = Flatten(name='flattened')(x) # error when followed by

    x = Dense(2048, name='Dense_1', activation='relu')(x)# 2048, 2048 ~  Input to reshape is a tensor with 442368 values, but the requested shape requires a multiple of 21632
	 # [[node model/flattened/Reshape (defined at /git/sdsandbox/src/train.py:272) ]] [Op:__inference_train_function_1192]

    # x = Dropout(drop)(x)
    x = Dense(50, name='Dense_2', activation='elu')(x)
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