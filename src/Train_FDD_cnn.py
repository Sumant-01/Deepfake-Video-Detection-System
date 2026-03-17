import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import warnings
warnings.filterwarnings('ignore')

def load_images_from_folder(folder_path, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('L').resize(img_size)
                images.append(np.array(img).flatten())
                labels.append(label)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    return images, labels

def main():
    img_cols, img_rows = 64, 64
    fake_folder = r"D:\100% code fake video detection\100% code\training_set\1"
    unfake_folder = r"D:\100% code fake video detection\100% code\training_set\2"

    print("📥 Loading FAKE images...")
    fake_images, fake_labels = load_images_from_folder(fake_folder, label=1)

    print("📥 Loading UNFAKE images...")
    unfake_images, unfake_labels = load_images_from_folder(unfake_folder, label=0)

    if len(fake_images) == 0 or len(unfake_images) == 0:
        print("❌ Not enough images found! Check folder paths.")
        return "Error: Data loading failed."

    print(f"✅ Number of FAKE images found: {len(fake_images)}")
    print(f"✅ Number of UNFAKE images found: {len(unfake_images)}")

    # Merge both datasets
    X = np.array(fake_images + unfake_images, dtype='f')
    y = np.array(fake_labels + unfake_labels)

    # Shuffle and split
    X, y = shuffle(X, y, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Reshape and normalize
    X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1).astype('float32') / 255

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # Build CNN model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_cols, img_rows, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("🚀 Training the model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    model.save("fake_event.h5")
    print("✅ Model saved as 'fake_event.h5'")

    predictions = model.predict(X_train)
    correct = sum(np.argmax(p) == np.argmax(a) for p, a in zip(predictions, y_train))
    accuracy = (correct / len(y_train)) * 100

    return f"🎯 Training Accuracy: {accuracy:.2f}%"

if __name__ == "__main__":
    result = main()
    print(result)
