import numpy as np
import csv
import math
import argparse
import cv2


def playCompared(base, compare, link, baseoutput, compareoutput):
    LinkData = np.loadtxt(link, delimiter = ',')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    baseCap = cv2.VideoCapture('input/videos/' + base)
    compareCap = cv2.VideoCapture('input/videos/' + compare)

    ret, image = baseCap.read()
    baseOut = cv2.VideoWriter('output/baseOut.mp4', fourcc, 30.0, (image.shape[1], image.shape[0]))
    ret, image = compareCap.read()
    compareOut = cv2.VideoWriter('output/' + compareoutput, fourcc, 30.0, (image.shape[1], image.shape[0]))

    i = 0
    LinkT = LinkData.T
    baseFrames = LinkT[0]

    while(baseCap.isOpened()):
        
        ret, frame = baseCap.read()
        temp = np.where(baseFrames == i)

        if ret:
            if len(temp[0])>0:
                baseOut.write(frame)
        else:
            break
        i = i + 1

    i = 0
    compareFrames = LinkT[1]
    while(compareCap.isOpened()):
        ret, frame = compareCap.read()
        temp = np.where(compareFrames == i)
        if ret:
            if(len(temp[0])>0):
                for j in range(len(temp[0])):
                    compareOut.write(frame)
        else:
            break
        i = i + 1

    baseCap.release()
    compareCap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compare keypoint position distance with demo csv log file''')
    parser.add_argument('--base', type=str, required=True, help='Base mp4 File')
    parser.add_argument('--compare', type=str, required=True, help='Compare mp4 File')
    parser.add_argument('--link', type=str, required=True, help='FrameLink File')
    parser.add_argument('--baseout', type=str, default='baseOut.mp4', help='OutputBase mp4 File')
    parser.add_argument('--compareout', type=str, default='compareOut.mp4', help='Output Compare mp4 File')
    
    
    args = parser.parse_args()

    playCompared(args.base, args.compare, args.link, args.baseout, args.compareout)