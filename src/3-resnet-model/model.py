import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import (Add, AveragePooling2D, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten, Input,
                                     ReLU)
from tensorflow.keras.models import Model

keras = tf.keras

class ResNetCNN():
    def __init__(self) -> None:
        pass

    # RELU + Batch Norm
    def relu_bn(self, inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn
    
    # ResNet block
    def residual_block(self,
        x: Tensor, 
        downsample: bool, 
        filters: int, 
        kernel_size: int=3
    ) -> Tensor:
        y = Conv2D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = self.relu_bn(y)
        y = Conv2D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)

        if downsample:
            x = Conv2D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = self.relu_bn(out)
        return out

    # Modified ResNet
    def create_model(self, img_shape, num_classes=3):

        inputs = Input(shape=img_shape)
        num_filters = 32
        
        t = BatchNormalization()(inputs)
        t = Conv2D(kernel_size=(5, 5),
                filters=num_filters,
                padding="same")(t)
        t = self.relu_bn(t)
        
        num_blocks_list = [2, 5, 5, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = self.residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
            num_filters *= 2
        
        t = AveragePooling2D(4)(t)
        t = Dropout(0.25)(t)
        t = Flatten(input_shape=(img_shape[1], img_shape[2]))(t)
        t = Dense(16, activation='relu')(t)
        t = Dropout(0.5)(t)
        outputs = Dense(num_classes, activation='softmax')(t)
        
        model = Model(inputs, outputs)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # compile the model

        return model

if __name__ == "__main__":
    img_shape = (100, 150, 3)
    classifier = ResNetCNN()
    model = ResNetCNN().create_model(img_shape=img_shape, num_classes=3)
