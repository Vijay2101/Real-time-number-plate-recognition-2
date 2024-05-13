import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import easyocr
import av

# Load the Haar Cascade classifier for license plate detection
harcascade = "models/haarcascade_russian_plate_number.xml"
# model = "models/license_plate_detector.pt"
plate_cascade = cv2.CascadeClassifier(harcascade)
# plate_cascade = cv2.CascadeClassifier(model)
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

class LicensePlateDetector:
    # def __init__(self):
    #     super().__init__()

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        # Convert frame to grayscale
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # Detect license plates
        plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
        min_area = 500
        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                # Draw a rectangle around the license plate
                cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 250, 0), 2)
                cv2.putText(frm, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                # Extract the license plate region
                img_roi = gray[y: y + h, x: x + w]

                # Perform OCR on the license plate
                output = reader.readtext(img_roi)

                # Display the recognized text
                if output:
                    plate_text = output[0][1]
                    cv2.putText(frm, f"Plate: {plate_text}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return av.VideoFrame.from_ndarray(frm,format='bgr24')

def main():
    st.title("License Plate Recognition")

    webrtc_ctx = webrtc_streamer(
        key="key",
        video_processor_factory=LicensePlateDetector,
        mode=WebRtcMode.SENDRECV,
    )

if __name__ == "__main__":
    main()