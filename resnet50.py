from keras.layers import MaxPool2D, Conv2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Dropout
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import image
from keras.models import Sequential
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import os

img_height, img_width = (200, 200)
batch_size = 32

train_data_dir = r"Random3\train"
test_data_dir = r"Random3\test"
val_data_dir = r"Random3\val"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.4)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

valid_generator = train_datagen.flow_from_directory(val_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation')

test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                   target_size=(img_height, img_width),
                                                   batch_size=1,
                                                   class_mode='categorical',
                                                   subset='validation')

x, y = test_generator.next()
x.shape

base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25, activation='softmax'))

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_generator,
          epochs=4)

model.save('model\ResNet50_ASL_4.h5')

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest Accuracy:', test_acc)

