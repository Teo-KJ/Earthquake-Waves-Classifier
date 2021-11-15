import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import (Add, AveragePooling2D, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPool2D, ReLU)
from tensorflow.keras.models import Model

keras = tf.keras

class ResNetCNN():
    def __init__(self) -> None:
        pass

    # RELU + Batch Norm
    def bn_relu(self, inputs: Tensor) -> Tensor:
        bn = BatchNormalization()(inputs)
        relu = ReLU()(bn)
        return relu
    
    # ResNet block
    def residual_block(self,
        x: Tensor, 
        filters: int, 
        kernel_size: int=3
    ) -> Tensor:
        y = self.bn_relu(x)
        y = Conv2D(kernel_size=kernel_size,
                strides= 2,
                filters=filters,
                padding="same")(y)

        y = self.bn_relu(y)
        y = Conv2D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)

        residual = Conv2D(kernel_size=1, strides=1, filters=1)(y)

        x = Conv2D(kernel_size=kernel_size,
                strides=2,
                filters=filters,
                padding="same")(x)
        out = Add()([x, residual])
        return out

    # Modified ResNet
    def create_model(self, img_shape, num_classes=3):

        inputs = Input(shape=img_shape)
        num_filters = 32
        
        t = BatchNormalization()(inputs)
        t = Conv2D(kernel_size=(5, 5),
                filters=num_filters,
                padding="same")(t)
        t = self.bn_relu(t)
        
        t = self.residual_block(t, filters=num_filters)
        t = self.residual_block(t, filters=num_filters*2)
        t = self.residual_block(t, filters=num_filters*4)
        
        t = ReLU()(t)
        t = Flatten(input_shape=(img_shape[1], img_shape[2]))(t)
        t = Dense(16, activation='relu')(t)
        t = ReLU()(t)
        outputs = Dense(num_classes, activation='softmax')(t)
        
        model = Model(inputs, outputs)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compile the model

        return model

if __name__ == "__main__":
    img_shape = (100, 150, 3)
    classifier = ResNetCNN()
    model = ResNetCNN().create_model(img_shape=img_shape, num_classes=3)
