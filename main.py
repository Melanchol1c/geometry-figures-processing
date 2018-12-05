import cv2
import imutils


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


image = cv2.imread("./ss1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Thresh", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

cX = 0
cY = 0
for c in cnts:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    k = 0 if area == 0 else ((perimeter * perimeter)/area)

    M = cv2.moments(c)
    if (M["m00"] == 0):
        M["m00"] = 1
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    cv2.drawContours(image, [c], -1, (255, 255, 255), 1)
    cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)

    cv2.putText(image, f"P: {toFixed(perimeter, 1)}", (cX - 60, cY - 0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"S: {toFixed(area, 1)}", (cX - 60, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"K: {toFixed(k, 1)}", (cX - 60, cY - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


cv2.imshow("Processed", image)
cv2.waitKey()
