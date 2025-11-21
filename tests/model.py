from tensorflow import keras

# Load and save pre-trained MNIST model
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.save('mnist_model.h5')