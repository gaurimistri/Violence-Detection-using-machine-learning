from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Function to send an email alert
def send_email_alert():
    sender_email = "gaurimistri557@gmail.com"
    receiver_email = "missgauri70@gmail.com"
    password = "hlge zwlh mhoe kmsj" # you can generate your own password by visit this link https://myaccount.google.com/apppasswords



    subject = "Violence Detected!"
    body = "Violence has been detected in the video. Please check the footage."

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Load the trained model
model = load_model('violence_detection_model.h5')

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Function to predict violence or non-violence
def predict_violence(model, img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    print(prediction);
    if prediction < 0.3:
        return "Non-Violence"
    else:
        send_email_alert()
        return "Violence"

# Example usage
image_path = 'test_dataset/violence.jpg'  # Replace with the path to your test image
result = predict_violence(model, image_path)
print(f"The image is classified as: {result}")
