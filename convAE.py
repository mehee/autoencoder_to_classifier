import tensorflow as tf 

class convAE():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def create_autoencoder_with_stacked_conv(self, filters=[64, 128, 256, 512], kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2)):
        # define encoder architecture
        self.encoder = tf.keras.models.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(self.input_shape))
        for i in range(len(filters)):
            self.encoder.add(tf.keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='relu', padding='same'))
            self.encoder.add(tf.keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='relu', padding='same'))
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.MaxPooling2D(pool_size=pool))


        # # define decoder architecture
        self.decoder = tf.keras.models.Sequential()
        for i in range(1,len(filters)):
            self.decoder.add(tf.keras.layers.UpSampling2D())
            self.decoder.add(tf.keras.layers.Conv2D(filters=filters[len(filters)-i], kernel_size=kernel, strides=stride, activation='relu', padding='same'))
            self.decoder.add(tf.keras.layers.Conv2D(filters=filters[len(filters)-i], kernel_size=kernel, strides=stride, activation='relu', padding='same'))
            self.decoder.add(tf.keras.layers.BatchNormalization())
        self.decoder.add(    tf.keras.layers.Conv2DTranspose(filters=self.input_shape[2], kernel_size=kernel, strides=strideundo, activation='sigmoid',  padding='same'))

        # compile model
        input         = tf.keras.layers.Input(self.input_shape)
        code          = self.encoder(input)
        reconstructed = self.decoder(code)

        self.autoencoder = tf.keras.models.Model(inputs=input, outputs=reconstructed)

    def create_autoencoder(self, filters, kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2)):
        #self.input_shape = input_shape
        # define encoder architecture
        self.encoder = tf.keras.models.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(self.input_shape))
        for i in range(len(filters)):
            self.encoder.add(tf.keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='elu', padding='same'))
            self.encoder.add(tf.keras.layers.MaxPooling2D(pool_size=pool))


        # # define decoder architecture
        self.decoder = tf.keras.models.Sequential()
        for i in range(1,len(filters)):
            self.decoder.add(tf.keras.layers.Conv2DTranspose(filters=filters[len(filters)-i], kernel_size=kernel, strides=strideundo, activation='elu', padding='same'))
        self.decoder.add(    tf.keras.layers.Conv2DTranspose(filters=self.input_shape[2],          kernel_size=kernel, strides=strideundo, activation='sigmoid',  padding='same'))

        # compile model
        input         = tf.keras.layers.Input(self.input_shape)
        code          = self.encoder(input)
        reconstructed = self.decoder(code)

        self.autoencoder = tf.keras.models.Model(inputs=input, outputs=reconstructed)

    def create_autoencoder_with_dense(self, filters, kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2), code_size=256):
        #self.input_shape = input_shape
        # define encoder architecture
        self.encoder = tf.keras.models.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(self.input_shape))
        for i in range(len(filters)):
            self.encoder.add(tf.keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='relu', padding='same'))
            self.encoder.add(tf.keras.layers.MaxPooling2D(pool_size=pool))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(code_size))

        # # define decoder architecture
        self.decoder = tf.keras.models.Sequential()
        self.decoder.add(tf.keras.layers.Dense(filters[len(filters)-1] * int(self.input_shape[0]/(2**(len(filters)))) * int(self.input_shape[1]/(2**(len(filters))))))
        self.decoder.add(tf.keras.layers.Reshape((int(self.input_shape[0]/(2**(len(filters)))),int(self.input_shape[1]/(2**(len(filters)))), filters[len(filters)-1])))
        for i in range(1,len(filters)):
            self.decoder.add(tf.keras.layers.Conv2DTranspose(filters=filters[len(filters)-i], kernel_size=kernel, strides=strideundo, activation='relu', padding='same'))
        self.decoder.add(    tf.keras.layers.Conv2DTranspose(filters=self.input_shape[2],          kernel_size=kernel, strides=strideundo, activation=None,  padding='same'))

        # compile model
        input         = tf.keras.layers.Input(self.input_shape)
        code          = self.encoder(input)
        reconstructed = self.decoder(code)

        self.autoencoder = tf.keras.models.Model(inputs=input, outputs=reconstructed)


