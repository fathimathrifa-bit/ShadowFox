import os
import ssl
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Safety Fix: Bypass potential local SSL download issues for academic datasets
ssl._create_default_https_context = ssl._create_unverified_context

print("--- Step 1: Verified Local Environment Configuration ---")
print("TensorFlow version running locally:", tf.__version__)

# 2. Load and Prepare the Dataset
print("\n--- Step 2: Downloading and Preparing CIFAR-10 Dataset ---")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0.0 and 1.0
train_images, test_images = train_images / 255.0, test_images / 255.0

# Explicit labels for the 10 structural classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Dataset successfully loaded locally.")
print(f"Training array dimensions: {train_images.shape}")
print(f"Testing array dimensions: {test_images.shape}")

# 3. Verify the Data Visually
print("\n--- Step 3: Launching Sample Verification Window ---")
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.tight_layout()
plt.show()

# 4. Construct the Convolutional Neural Network (CNN)
print("\n--- Step 4: Compiling Convolutional Neural Network Architecture ---")
model = models.Sequential([
    # Feature Engineering via Layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flattening and Dense Classification Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Define optimizer, metrics, and loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 5. Local Training Phase
print("\n--- Step 5: Commencing Local Training Iterations (5 Epochs) ---")
# Using 5 epochs to ensure it finishes quickly on local CPU or GPU configurations
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# 6. Evaluate and Save the Trained Architecture Locally
print("\n--- Step 6: Saving Weights and Local Model Preservation ---")
model_filename = 'shadowfox_image_model.h5'
model.save(model_filename)
print(f"Success! Model binary file saved inside your directory as: '{os.path.abspath(model_filename)}'")

# 7. Local Prediction Walkthrough Example
print("\n--- Step 7: Performing Verification Prediction on Sample Image ---")
test_index = 0  # Feel free to change this index to experiment with different images!
sample_img = test_images[test_index]

# Reshape the data pattern to create a batch size of 1 for evaluation
predictions = model.predict(np.expand_dims(sample_img, axis=0))
softmax_probabilities = tf.nn.softmax(predictions[0])
predicted_class_idx = np.argmax(softmax_probabilities)

predicted_label = class_names[predicted_class_idx]
actual_label = class_names[test_labels[test_index][0]]

# Plot prediction result to screen
plt.figure()
plt.imshow(sample_img)
plt.title(f"Predicted Class: {predicted_label.upper()} \n True Ground Label: {actual_label.upper()}")
plt.axis('off')
plt.show()

print("\nTask execution complete! Ready for documentation and submission pipeline steps.")