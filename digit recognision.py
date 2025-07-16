# Run it on camera

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load your trained model (replace with your model if not saved)
# model = load_model('my_model.h5')  # Optional if you've saved it
# # If you already have the model in memory, skip this step

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Define a region of interest (ROI) to draw
#     x, y, w, h = 200, 100, 200, 200
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Extract ROI from frame
#     roi = frame[y:y+h, x:x+w]
#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     roi_resized = cv2.resize(roi_gray, (28, 28))
    
#     # Invert and normalize
#     roi_inverted = 255 - roi_resized
#     roi_normalized = roi_inverted / 255.0

#     # Predict
#     input_img = roi_normalized.reshape(1, 28, 28)
#     # print(input_img)
#     pred = model.predict(input_img)
#     digit = np.argmax(pred)
#     # print(digit)
    

#     # Display prediction
#     cv2.putText(frame, f"Prediction: {digit}", (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

#     # Show the frame
#     cv2.imshow("Digit Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# run it on image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load a trained model (if not in memory)
model = load_model('my_model.h5')

# Load the image
image = cv2.imread('handwriting.jpg')  # replace with your image file

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: Crop or use ROI if the digit is only in part of the image
# gray = gray[y1:y2, x1:x2]

# Resize to 28x28 (MNIST size)
resized = cv2.resize(gray, (28, 28))

# Invert colors if background is white
inverted = 255 - resized

# Normalize to [0, 1]
normalized = inverted / 255.0

# Reshape to fit model input
input_img = normalized.reshape(1, 28, 28)

# Predict
pred = model.predict(input_img)
digit = np.argmax(pred)

print(f"Predicted digit: {digit}")

# Optional: Display the input
cv2.imshow("Input", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
