import keras.layers as layers
import keras.models as models
import keras.optimizers as optimizers


def create_model(width, height, class_number):
    model = models.Sequential()
    dropout = 0.4

    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(width, height, 3)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(class_number, activation='softmax'))

    optimizer = optimizers.SGD(lr=0.02)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

