
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import multi_gpu_model


class ModelIntruderNet38:


    def __init__(self):
        ''' Constructor for this class. '''

    def get_epochs(self):
        #epochs = 1
        #epochs = 3
        #epochs = 10
        #epochs = 50
        #epochs = 100#
        epochs = 150
        #epochs = 500
        return epochs

    def get_modelname(self):
        save_name = "mod38_multi_5.i" + str(self.get_epochs())
        return save_name

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(filters=2, kernel_size=2, padding='same', activation='relu', input_shape=(640, 480, 3)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=4, kernel_size=2, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=8, kernel_size=2, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(GlobalAveragePooling2D(data_format=None))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='sigmoid'))

        return model

    def get_model_compiled_single_gpu(self):
        model = self.get_model()
#        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_model_compiled_multi_gpu(self):
        # Replicates `model` on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        model = self.get_model()
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #parallel_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        # parallel_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return parallel_model

