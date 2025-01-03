import requests
from keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load the model
model = load_model("C:\\Users\\HP\\Desktop\\1st project\\image_classification_model.keras") # download the model and change the path
print("Model loaded successfully.")

# Class names (CIFAR-10 labels)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def predict_image(image_url):
    temp_image_path = "temp_image.jpg"
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(temp_image_path, "wb") as file:
                file.write(response.content)

            # Load and preprocess the image
            img = load_img(temp_image_path, target_size=(32, 32))
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            # Debugging outputs
            print("Raw predictions:", prediction[0])
            print(f"Predicted class: {class_names[predicted_class]}")
            print(f"Confidence: {confidence:.2f}")

            # Display the image and prediction
            plt.imshow(img)
            plt.title(f"Predicted: {class_names[predicted_class]} ({confidence:.2f})")
            plt.axis('off')
            plt.show()

        else:
            print(f"Failed to download the image. HTTP Status Code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Example usage , replace your image url below
predict_image("https://imgs.search.brave.com/qO-0d9c0lU7vhvjA8sPu644-A2tujw9-6M5wcxpOcJI/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzAzLzIwLzczLzQy/LzM2MF9GXzMyMDcz/NDIxNF84cndWR0U2/VUh0TjJXakVrcGJh/dmVkaUpaMnFrZFBX/Yi5qcGc")
