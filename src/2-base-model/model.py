import tensorflow as tf

keras = tf.keras

class ClassfierCNN():
    def create_model(img_shape, num_classes=3):
        model = keras.Sequential()

        model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation = 'relu', padding = 'same')) # convolutional layer
        model.add(keras.layers.MaxPool2D(2,2)) # max pooling layer
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten(input_shape=(img_shape[1], img_shape[2])))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        # model.add(keras.layers.Dense(10, activation='softmax'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compile the model

        return model

# class RegressionCNN():
#     def create_model(img_shape):
#         model = keras.Sequential()

#         model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation = 'relu', padding = 'same')) # convolutional layer
#         model.add(keras.layers.MaxPool2D(2,2))
#         model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')) # convolutional layer
#         model.add(keras.layers.MaxPool2D(2,2))
#         model.add(keras.layers.Dropout(0.25))
#         model.add(keras.layers.Flatten(input_shape=(img_shape[1], img_shape[2])))
#         model.add(keras.layers.Dense(16, activation='relu'))
#         model.add(keras.layers.Dense(32, activation='relu'))
#         model.add(keras.layers.Dense(1))

#         model.compile(optimizer='adam', loss=['mse']) # regression model uses mse for loss

#         return model