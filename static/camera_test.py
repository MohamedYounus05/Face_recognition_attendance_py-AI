import cv2

cam = cv2.VideoCapture(0)

print("Camera opened:", cam.isOpened())

if cam.isOpened():
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Frame not received")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) == 13:
            break
else:
    print("Camera not detected")

cam.release()
cv2.destroyAllWindows()
