def main():
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras import optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    import matplotlib.pyplot as plt

    basepath = "D:/100% code fake video detection/100% code"

    # Initializing the CNN
    classifier = Sequential()
    classifier.add(Convolution2D(32, (1, 1), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(32, (1, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(64, (1, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2, activation='softmax'))

    classifier.compile(optimizer=optimizers.SGD(learning_rate=0.01),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # Image generators
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        basepath + '/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
        basepath + '/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    steps_per_epoch = int(np.ceil(training_set.samples / 32))
    val_steps = int(np.ceil(test_set.samples / 32))

    model = classifier.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=test_set,
        validation_steps=val_steps
    )

    # Save model
    classifier.save(basepath + '/model.h5')

    # Evaluate
    test_score = classifier.evaluate(test_set, verbose=1)
    train_score = classifier.evaluate(training_set, verbose=1)

    B = "Testing Accuracy: %.2f%%" % (test_score[1] * 100)
    C = "Training Accuracy: %.2f%%" % (train_score[1] * 100)
    print(B)
    print(C)

    # Plot Accuracy
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/accuracy.png", bbox_inches='tight')
    plt.show()

    # Plot Loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/loss.png", bbox_inches='tight')
    plt.show()

    return B + '\n' + C


# ✅ This will run when you execute the file directly
if __name__ == '__main__':
    result = main()
    print("Training completed.\n" + result)
