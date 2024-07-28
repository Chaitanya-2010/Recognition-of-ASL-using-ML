# import the opencv library
# import cv as cv
import cv2
import cv2 as cv
import os

top, right, bottom, left = 100, 150, 400, 450
IMG_SIZE = 512

exit_con = '**'

a = ''

dir0 = input('enter the directory name: ')

try:
    os.mkdir(dir0)
except:
    print('contain folder in same name')

# define a video capture object
vid = cv.VideoCapture(2)
# fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
# fgbg = cv.createBackgroundSubtractorKNN()

while True:
    a = input('exit: ** or enter the label name: ')

    if a == exit_con:
        break

    dir1 = str(dir0) + '/' + str(a) + '/'
    print(dir1)

    try:
        os.mkdir(dir1)
    except:
        print('contain folder')

    i = 0

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        roi = frame[top:bottom, right:left]
        fgmask = cv.GaussianBlur(roi, (7, 7), 0)

        # fgmask = fgbg.apply(frame)

        frame = cv.flip(frame, 1)
        fgmask = cv.flip(fgmask, 1)

        fgmask = cv.resize(fgmask, (IMG_SIZE, IMG_SIZE))

        cv.imwrite("%s%s%d.jpg"%(dir1,a,i), fgmask)
        i += 1
        print(i)
        if i > 3000:
            break

        # Display the resulting frame
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv.imshow('frame', frame)
        cv.imshow('Mask Frame', fgmask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()
