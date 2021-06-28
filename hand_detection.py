import cv2
import numpy as np
import os
from Video import *


def HandAndFace_detection(image, roi1, roi2, count):
    # test_image = cv2.imread('src/hand_gesture/image_000000000.jpg')
    test_image = image

    hsvim = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (1, 1))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    # hand+face까지 할 때 사용

    # dst = thresh.copy()
    # new_image = np.zeros(shape=thresh.shape)

    # only_hands = dst[roi1[0][1]:roi1[1][1], roi1[0][0]:roi1[1][0]]
    # new_image[roi1[0][1]:roi1[1][1], roi1[0][0]:roi1[1][0]] = only_hands
    # only_faces = dst[roi2[0][1]:roi2[1][1], roi2[0][0]:roi2[1][0]]
    # new_image[roi2[0][1]:roi2[1][1], roi2[0][0]:roi2[1][0]] = only_faces

    cv2.imwrite(f"hand_frames/{count:04}.png", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


hand_roi = ((120, 60), (180, 130))
face_roi = ((205, 33), (274, 114))
col_frames = os.listdir('src/hand')

col_images = []

for i in col_frames:
    img = cv2.imread('src/hand/' + i)
    col_images.append(img)
for i in range(len(col_images)):
    print(f'{i}번째 이미지 저장')
    HandAndFace_detection(
        col_images[i],
        hand_roi,
        face_roi,
        i)
frames2video('hand_frames/', 'video/', 'hand.mp4')
