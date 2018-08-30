import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

class VGG:

    def __init__(self, input_shape, epochs, batch_size, model_name, dual):
        if dual:
            self.model = self.initialize_dual_vgg_model(input_shape)
        else:
            self.model = self.initialize_vgg_model(input_shape)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        if os.path.isfile(model_name + '.h5'):
            self.model = load_model(model_name + '.h5')
        self.model_name = model_name

    def initialize_transfer_model(self, input_shape):
        vgg = VGG16(weights='imagenet', include_top=False)

        input_layer = Input(shape=input_shape, name='image_input')

        vgg_out = vgg(input_layer)

        x = Flatten(name='flatten')(vgg_out)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1, name='prediction')(x)

        return Model(input=input_layer, output=x)

    def conv_vgg(self, x, depth):
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(depth, (3, 3), activation='relu')(x)
        x = ZeroPadding2D((1,1))(x)
        x = Conv2D(depth, (3, 3), activation='relu')(x)
        model = MaxPooling2D((2,2), strides=(2,2))(x)
        return model

    def initialize_vgg_model(self, input_shape):
            input = Input(shape=input_shape)


            x = self.conv_vgg(input, 64)
            x = self.conv_vgg(x, 128)
            x = self.conv_vgg(x, 256)
            x = self.conv_vgg(x, 512)
            x = self.conv_vgg(x, 512)

            x = Flatten()(x)
            x = Dense(4096, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation='relu')(x)
            x = Dropout(0.5)(x)
            out = Dense(1, activation=None)(x)
            model = Model(inputs=[input], outputs=[out])
            return model

    def initialize_dual_vgg_model(self, input_shape):
            x1 = Input(shape=input_shape)

            model1 = self.conv_vgg(x1, 64)
            model1 = self.conv_vgg(model1, 128)
            model1 = self.conv_vgg(model1, 256)
            model1 = self.conv_vgg(model1, 512)
            model1 = self.conv_vgg(model1, 512)

            x2 = Input(shape=input_shape)

            model2 = self.conv_vgg(x2, 64)
            model2 = self.conv_vgg(model2, 128)
            model2 = self.conv_vgg(model2, 256)
            model2 = self.conv_vgg(model2, 512)
            model2 = self.conv_vgg(model2, 512)

            main_model = concatenate([model1, model2])

            x = Flatten()(main_model)
            x = Dense(4096, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation='relu')(x)
            x = Dropout(0.5)(x)
            out = Dense(1, activation=None)(x)
            model = Model(inputs=[x1, x2], outputs=[out])
            return model

    def train(self, train_X, train_y, test_X, test_y):
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                     TensorBoard(log_dir='logs/{}'.format(self.model_name), batch_size=self.batch_size, write_images=True),
                     ModelCheckpoint(filepath=self.model_name + '.h5', save_best_only=True)]
        self.model.fit(train_X,
                       train_y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=2,
                       callbacks=callbacks,
                       validation_data=(test_X, test_y))

    def predict(self, test_X):
        return self.model.predict(test_X)
