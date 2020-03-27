
# coding: utf-8

# In[ ]:


import datetime
import os.path as osp
import tensorflow as tf
import numpy as np

from math import ceil
from tensorflow.keras import callbacks, optimizers
import tensorflow.keras.backend as K

import os
import multiprocessing as mt

from utils_tfkeras import DataGenerator
# from data_pipline import py_func_train_process, py_func_test_process

from model_resnet_custom import DBNet_res50, DBNet_res18

from losses_tfkeras import db_loss

print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[ ]:


lr = 0.001
Epochs = 4
finetune = False

checkpoints_dir = f'checkpoints/{datetime.date.today()}'

batch_size = 4
image_size = 640

root_path = '/home/shaoran/Data/ocr/SROIE2019'

list_IDs = []
labels = []

for i in os.listdir(os.path.join(root_path, 'gt_txt')):
    path_l = os.path.join(root_path, 'gt_txt', i)
    j = i.split('.')[0] + '.jpg'
    path_i = os.path.join(root_path, 'img', j)
    if os.path.exists(path_i):
        list_IDs.append(path_i)
        labels.append(path_l)
    else:
        continue

cc = list(zip(list_IDs, labels))
np.random.shuffle(cc)
list_IDs, labels =zip(*cc)
        
assert len(labels) == len(list_IDs), "labels != images"
print('sample nums: ', len(labels))
length_data = len(labels)

train_list_IDs = list_IDs[:int(0.95*length_data)]
train_labels = labels[:int(0.95*length_data)]

test_list_IDs = list_IDs[int(0.95*length_data):]
test_labels = labels[int(0.95*length_data):]


if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir) 


# In[ ]:


train_generator = DataGenerator(train_list_IDs, train_labels, batch_size=batch_size, image_size=image_size,
                                min_text_size=8, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7, is_training=True, shuffle=True)
val_generator = DataGenerator(test_list_IDs, test_labels, batch_size=batch_size, image_size=image_size,
                              min_text_size=8, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7, is_training=False, shuffle=False)

# train_generator = tf.data.Dataset.from_tensor_slices((np.array(train_list_IDs), np.array(train_labels)))
# train_generator = train_generator.map(py_func_train_process, num_parallel_calls=mt.cpu_count())
# train_generator = train_generator.batch(batch_size).repeat()
# val_generator = tf.data.Dataset.from_tensor_slices((np.array(test_list_IDs), np.array(test_labels)))
# val_generator = val_generator.map(py_func_test_process, num_parallel_calls=mt.cpu_count())
# val_generator = val_generator.batch(batch_size).repeat()

# model, prediction_model = DBNet_res50()
model, prediction_model = DBNet_res18()

if finetune:
    
    lr = lr*0.5
#     resnet_filepath = './models/SROIE2019_resnet50_db_36_1.1591_1.2635_weights.h5'
    resnet_filepath = './models/tf_SROIE2019_resnet18_db_weights.h5'
    
    model.load_weights(resnet_filepath, by_name=True)
    
# model.compile(optimizers.Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred) db_loss
model.compile(loss=db_loss, optimizer=optimizers.Adam(learning_rate=lr))

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = Epochs
    step_each_epoch=int(len(train_labels) // batch_size) #根据自己的情况设置
    baseLR = lr
    power = 0.9
    ite = K.get_value(model.optimizer.iterations)
    # compute the new learning rate based on polynomial decay
    alpha = baseLR*((1 - (ite / float(maxEpochs*step_each_epoch)))**power)
    # return the new learning rate
    return alpha

learningRateScheduler = callbacks.LearningRateScheduler(poly_decay)

logs_dir = f'logs/{datetime.date.today()}'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir) 
tensorboard = callbacks.TensorBoard(log_dir=logs_dir)

checkpoint = callbacks.ModelCheckpoint(
    osp.join(checkpoints_dir, 'tf_SROIE2019_resnet18_db_{epoch:02d}_{loss:.4f}_{val_loss:.4f}_weights.h5'),
#   osp.join(checkpoints_dir, 'SROIE2019_1.h5'),
    verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='auto'
)

# In[ ]:
# fit_generator
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=ceil(len(train_labels) // batch_size),
    initial_epoch=0,
    epochs=Epochs,
    callbacks=[learningRateScheduler, checkpoint, tensorboard],
    validation_data=val_generator,
    validation_steps=ceil(len(test_labels) // batch_size),
    use_multiprocessing=True, workers=2,
    verbose=1)

# tf.data.Dataset-pipline
# model.fit(
#     generator=train_generator,
#     initial_epoch=0,
#     epochs=Epochs,
#     callbacks=[learningRateScheduler, checkpoint, tensorboard],
#     validation_data=val_generator,
#     verbose=1)

model_dir = f'models/{datetime.date.today()}'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_weights(os.path.join(model_dir, 'tf_SROIE2019_resnet18_db_weights.h5'))
model.save(os.path.join(model_dir,'tf_SROIE2019_resnet18_db_model.h5'))
tf.saved_model.save(os.path.join(model_dir, 'saved_tf_model'))

