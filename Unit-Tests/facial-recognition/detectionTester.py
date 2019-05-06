import csv
from PIL import Image
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

NUMBEROFIMAGES = 5000

def detect(pic,row):
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        height, width, channels = pic.shape
        scale = 500/width
        frame = imutils.resize(pic, width=500)
        height, width, channels = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                minNeighbors=5, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex,ey,ew,eh) in rects:
            dx = min((float(row[1]) + float(row[3])),(ew+ex)/scale)
            dy = min((float(row[2]) + float(row[4])), (eh+ey)/scale) - max(float(row[2]), ey/scale)
            if (dx>=0) and (dy>=0):
                area = dx*dy
                percent = area / ((ex/scale)*(ey/scale))
                if (percent > 0.4):
                    return True
        return False

with open('umdfaces_batch3_ultraface.csv') as csv_file:
    SUCCESSFUL = 0
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        try:
            pic = cv2.imread(row[0])
            if line_count == NUMBEROFIMAGES:
                break
            if detect(pic,row):
                SUCCESSFUL+=1
        except AttributeError:
            continue
        line_count+=1
        print("Tested image " + str(line_count) + " / " + str(NUMBEROFIMAGES) + " ... Success : " + str((SUCCESSFUL / NUMBEROFIMAGES)*100) + "% ...")
    print("Done.")
