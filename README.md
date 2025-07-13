# 🧠 Brain Tumor Classification (Simple ANN)

This project builds a **simple Artificial Neural Network (ANN)** in TensorFlow to classify MRI brain images as **Tumor** or **No Tumor**.

> ⚡ **Note:** This uses a basic ANN with backpropagation — *not a CNN*. For production-grade medical imaging, a Convolutional Neural Network (CNN) is strongly recommended.

---

## 📌 Overview

* **Goal:** Classify MRI scans to detect brain tumors.
* **Framework:** TensorFlow (Keras API)
* **Type:** Binary Classification (Yes / No)
* **Model:** Basic fully connected neural network.

---

## 🗂️ Dataset Structure

Your dataset should be organized like this:

```
brain_tumer_ml/
└── Dataset/
    ├── Train/
    │   ├── yes/    # Images with brain tumors
    │   └── no/     # Images without brain tumors
    └── Test/
        ├── yes/    # Test images with brain tumors
        └── no/     # Test images without brain tumors
```

✅ **Train/**

* Used to train the model.
* `yes/` ➜ Tumor present
* `no/` ➜ Tumor absent

✅ **Test/**

* Used for evaluating accuracy on new data.
* Follows same folder structure.

---

## ⚙️ How It Works

1️⃣ **Mount Google Drive**
Mount your Drive in Colab to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2️⃣ **Load & Preprocess Images**

* Images resized to 64×64
* Loaded as **grayscale**
* Normalized to `[0, 1]`

3️⃣ **Build the ANN**
A simple feedforward network:

* **Input:** Flatten layer
* **Hidden Layers:** Dense(128) + Dense(64) with ReLU
* **Output:** Dense(1) with Sigmoid activation

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

4️⃣ **Compile & Train**

* Optimizer: Adam
* Loss: Binary Crossentropy
* Metric: Accuracy

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, validation_data=test_ds, epochs=20)
```

5️⃣ **Evaluate Model**

* Evaluate using both manual accuracy and `model.evaluate()`.

6️⃣ **Single Image Prediction**
Test prediction on a new image:

```python
prediction = model.predict(img_array)
```

7️⃣ **Save Model**
Export the trained model:

```python
model.save('my_model.h5')
```

---

## ✅ Requirements

* Python 3.x
* TensorFlow
* NumPy
* Matplotlib
* Google Colab (recommended)

---

## 🔍 Limitations

* **ANN only:** This model uses only dense layers. For better image classification, use CNN layers like `Conv2D` and `MaxPooling2D`.
* **Small Input Size:** 64×64 may limit feature learning.
* **Medical Use:** This is a learning project — **do not** use for real medical diagnosis.

---

## 📊 Results

* Prints manual accuracy and built-in `evaluate()` accuracy.
* Example prediction on a single custom image.
* Saves model as `my_model.h5`.

---

## 📖 Notebook

* [📓 Original Colab Notebook](https://colab.research.google.com/drive/1ZRhJnsXa-oE3BOPOREkiLjJ_7NTXnuij)

---

## 🚀 How to Use

1. Upload the dataset to Google Drive.
2. Open the notebook in Colab.
3. Update dataset paths if needed.
4. Run all cells step-by-step.
5. Test predictions and save the model.

---

## ✍️ Author

**Aditya Bhatagar**
🔗 [Git](https://github.com/Aditya04012)

---

## ⭐️ License

This project is open source for educational purposes. Feel free to fork & experiment!

---

**If you find this project helpful, please give it a ⭐ on GitHub!**
