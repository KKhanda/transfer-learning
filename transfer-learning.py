# !pip install -q wget
import wget as wg
from keras import applications
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, Dropout, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

width, height = 299, 299
train_data_dir = 'data/train'
test_data_dir = 'data/test'

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory(train_data_dir, target_size=(width, height),
                                                 batch_size=64, class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_data_dir, target_size=(width, height),
                                            batch_size=64, class_mode='categorical')

if K.image_dim_ordering() == 'th':
    input_tensor = Input(shape=(3, 299, 299))
else:
    input_tensor = Input(shape=(299, 299, 3))

model = applications.InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=False)
model.summary()

for layer in model.layers:
    layer.trainable = False

output = model.output
output = AveragePooling2D((8, 8), padding='valid', name='avg_pool')(output)
output = Dropout(0.4)(output)
output = Flatten()(output)
# output = Dense(units=128, activation='relu')(output)
output = Dense(units=2, activation='softmax')(output)

new_model = Model(inputs=model.input, outputs=output)

new_model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

new_model.fit_generator(training_set, steps_per_epoch=10.0, epochs=3,
                        validation_data=test_set, validation_steps=10.0)
