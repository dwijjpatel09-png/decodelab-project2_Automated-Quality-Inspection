import cv2
import numpy as np

CAMERA_INDEX = 0
MIN_DEFECT_AREA = 200

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defect_found = False

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > MIN_DEFECT_AREA:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                display,
                "DEFECT",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            defect_found = True

    if defect_found:
        status = "FAIL"
        color = (0, 0, 255)
    else:
        status = "PASS"
        color = (0, 255, 0)

    cv2.putText(
        display,
        f"STATUS: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Inspection Result", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
