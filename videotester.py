import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ✅ Use ONLY tensorflow.keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ✅ Load model
model = load_model("best_model.h5", compile=False)

# Face detector
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

# Emotion labels
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while True:
    ret, test_img = cap.read()
    if not ret:
        continue

    # ✅ Convert to GRAY (correct)
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ✅ Correct ROI
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))

        # Convert to array
        img_pixels = img_to_array(roi_gray)

        # ✅ Convert grayscale → 3 channel
        img_pixels = np.stack((img_pixels,) * 3, axis=-1)

        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = img_pixels / 255.0

        # Prediction
        predictions = model.predict(img_pixels, verbose=0)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Show text
        cv2.putText(
            test_img,
            predicted_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial Emotion Analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()