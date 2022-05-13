import cv2
import pytesseract


#stream = cv2.VideoCapture("video2.mp4")
#stream = cv2.VideoCapture("video1.mp4")

stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print (stream.isOpened())
screenCnt = None
fpsCount = 0
text = ' '
custom_config = r'--oem 3 --psm 6'
#########################################################################
while True:
    _, img = stream.read()
    fpsCount += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    count = 0
    # loop over contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:  # chooses contours with 4 corners
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)  # finds co-ordinates of the plate
            new_img = img[y:y + h, x:x + w]
            break


    if screenCnt is None:
        cv2.imshow("Final image with plate detected", img)
    else:
        if fpsCount == 24:
            fpsCount = 0
            text = pytesseract.image_to_string(new_img, lang='eng',config=custom_config)  # converts image characters to string
            print("Number is:", text)
            
            
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        cv2.putText(img,text, (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
        cv2.imshow("Grayscale Processing", gray)
        cv2.imshow("Edge Detection Processing", edged)
        cv2.imshow("Final image with plate detected", img)
        cv2.imshow('Cropped For OCR', new_img)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
#########################################################################

cv2.destroyAllWindows()
