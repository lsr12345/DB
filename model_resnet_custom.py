
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
# from losses_tf import db_loss


# In[2]:


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.filter_num = filter_num
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        config = {"filter_num":self.filter_num}
        base_config = super(BasicBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


# In[2]:


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.filter_num = filter_num
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        config = {"filter_num":self.filter_num}
        base_config = super(BottleNeck, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block


# In[3]:


def DBNet_res50(input_size=640, k=50, training=True):  # 3, 4, 6, 3
    inputs = keras.layers.Input(name='input_image', shape=(None, None, 3))
    gt_input = keras.layers.Input(shape=(input_size, input_size))
    mask_input = keras.layers.Input(shape=(input_size, input_size))
    thresh_input = keras.layers.Input(shape=(input_size, input_size))
    thresh_mask_input = keras.layers.Input(shape=(input_size, input_size))

    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x, training=training)
    x = keras.layers.Activation('relu')(x)
    x  = keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)

    C2 = make_bottleneck_layer(filter_num=64, blocks=3)(x, training=training)
    C3 = make_bottleneck_layer(filter_num=128, blocks=4, stride=2)(C2, training=training)
    C4 = make_bottleneck_layer(filter_num=256, blocks=6, stride=2)(C3, training=training)
    C5 = make_bottleneck_layer(filter_num=512, blocks=3, stride=2)(C4, training=training)
    
    in2 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

    # 1 / 32 * 8 = 1 / 4
    P5 = keras.layers.UpSampling2D(size=(8, 8))(
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
    # 1 / 16 * 4 = 1 / 4
    out4 = keras.layers.Add()([in4, keras.layers.UpSampling2D(size=(2, 2))(in5)])
    P4 = keras.layers.UpSampling2D(size=(4, 4))(
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
    # 1 / 8 * 2 = 1 / 4
    out3 = keras.layers.Add()([in3, keras.layers.UpSampling2D(size=(2, 2))(out4)])
    P3 = keras.layers.UpSampling2D(size=(2, 2))(
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
    # 1 / 4
    P2 = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
        keras.layers.Add()([in2, keras.layers.UpSampling2D(size=(2, 2))(out3)]))
    # (b, /4, /4, 256)
    fuse = keras.layers.Concatenate()([P2, P3, P4, P5])

    # probability map
    p = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = keras.layers.BatchNormalization()(p)
    p = keras.layers.ReLU()(p)
    p = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = keras.layers.BatchNormalization()(p)
    p = keras.layers.ReLU()(p)
    p = keras.layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(p)

    # threshold map
    t = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.ReLU()(t)
    t = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.ReLU()(t)
    t = keras.layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(t)

    # approximate binary map
    b_hat = keras.layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])

#     loss = keras.layers.Lambda(db_loss, name='db_loss')([p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input])
    p_ = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(p)
    b_hat = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(b_hat)
    t = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(t)
    
    db_outputs = keras.layers.Concatenate(name='db_outputs', axis=-1)([p_, b_hat, t])
    train_model = keras.Model(inputs, db_outputs)
    predict_model = keras.Model(inputs, p)
    
    return train_model, predict_model


# In[6]:


def DBNet_res18(input_size=640, k=50, training=True): # 2, 2, 2, 2
    inputs = keras.layers.Input(name='input_image', shape=(None, None, 3))
    gt_input = keras.layers.Input(shape=(input_size, input_size))
    mask_input = keras.layers.Input(shape=(input_size, input_size))
    thresh_input = keras.layers.Input(shape=(input_size, input_size))
    thresh_mask_input = keras.layers.Input(shape=(input_size, input_size))

    x = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x, training=training)
    x = keras.layers.Activation('relu')(x)
    x  = keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(x)

    C2 = make_basic_block_layer(filter_num=64, blocks=2)(x, training=training)
    C3 = make_basic_block_layer(filter_num=128, blocks=2, stride=2)(C2, training=training)
    C4 = make_basic_block_layer(filter_num=256, blocks=2, stride=2)(C3, training=training)
    C5 = make_basic_block_layer(filter_num=512, blocks=2, stride=2)(C4, training=training)
    
    in2 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = keras.layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

    # 1 / 32 * 8 = 1 / 4
    P5 = keras.layers.UpSampling2D(size=(8, 8))(
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
    # 1 / 16 * 4 = 1 / 4
    out4 = keras.layers.Add()([in4, keras.layers.UpSampling2D(size=(2, 2))(in5)])
    P4 = keras.layers.UpSampling2D(size=(4, 4))(
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
    # 1 / 8 * 2 = 1 / 4
    out3 = keras.layers.Add()([in3, keras.layers.UpSampling2D(size=(2, 2))(out4)])
    P3 = keras.layers.UpSampling2D(size=(2, 2))(
        keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
    # 1 / 4
    P2 = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
        keras.layers.Add()([in2, keras.layers.UpSampling2D(size=(2, 2))(out3)]))
    # (b, /4, /4, 256)
    fuse = keras.layers.Concatenate()([P2, P3, P4, P5])

    # probability map
    p = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = keras.layers.BatchNormalization()(p)
    p = keras.layers.ReLU()(p)
    p = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = keras.layers.BatchNormalization()(p)
    p = keras.layers.ReLU()(p)
    p = keras.layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(p)

    # threshold map
    t = keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.ReLU()(t)
    t = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.ReLU()(t)
    t = keras.layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                   activation='sigmoid')(t)

    # approximate binary map
    b_hat = keras.layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])

#     loss = keras.layers.Lambda(db_loss, name='db_loss')([p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input])
    p_ = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(p)
    b_hat_ = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(b_hat)
    t_ = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(t)
    
    db_outputs = keras.layers.Concatenate(name='db_outputs', axis=-1)([p_, b_hat_, t_])
    train_model = keras.Model(inputs, db_outputs)
    predict_model = keras.Model(inputs, p)
    
    return train_model, predict_model

