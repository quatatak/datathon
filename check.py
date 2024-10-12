import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model1.h5')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the categories
categories = ['0', '1', '2']

IMG_WIDTH = 154
IMG_HEIGHT = 116

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    
    # Debug: Show resized image
    cv2.imshow('Resized', img)
    
    # Reshape for model input (no normalization needed)
    reshaped = np.reshape(img, (1, IMG_WIDTH, IMG_HEIGHT, 3))
    
    # Make prediction
    prediction = model.predict(reshaped)
    category_index = np.argmax(prediction)
    category = categories[category_index]
    confidence = prediction[0][category_index]
    
    # Print debug information
    print(f"Raw prediction: {prediction}")
    print(f"Predicted category: {category}")
    print(f"Confidence: {confidence:.2f}")
    
    # Put text on the frame
    text = f"{category}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Mask Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()