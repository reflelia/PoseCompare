import numpy as np
import csv
import math
import argparse
from utils import resize_kpt, drawPoints, drawLines, saveUnitVec, getCosineSimilarity, getMax, getMin, getSumAndLen, getSummation1, getSummation2, getKpt, getDataMin, getDataMax
import cv2
import keyIndex

kpt_names = keyIndex.mpii_names

num_kpt = len(kpt_names)

def run_single(baseData, frame):

    baseNp = getKpt(num_kpt, baseData, frame)

    # get bbox
    xMin, yMin = getDataMin(num_kpt, baseNp)
    xMax, yMax = getDataMax(num_kpt, baseNp)

    width = xMax - xMin
    height = yMax - yMin
    keypoints = resize_kpt(baseNp, width, height, xMin, yMin)

    return keypoints
        
def CompareAllFrame(base, compare, baseStart, baseEnd, compareStart, compareEnd, threshold, pairArray, cosine):
    
    baseData = np.loadtxt(base, delimiter = ',')
    compareData = np.loadtxt(compare, delimiter = ',')
    TotalFrameLink = np.empty((0,3), float)
    previousValue = 0
    if baseEnd == -1:
        baseEnd = baseData.shape[0]
    if compareEnd == -1:
        compareEnd = compareData.shape[0]
    if cosine:
        for i in range(baseStart, baseEnd):
            vector_list = []
            
            goodScoreList = np.empty((0,2), float)
            base_keypoints = run_single(baseData, i)

            # 단위 벡터 저장
            saveUnitVec(base_keypoints, pairArray, vector_list)
            maxScore = -1
            maxFrame = 0
            for j in range(compareStart, compareEnd):
                vector_list2 = []
                compare_keypoints = run_single(compareData, j)

                saveUnitVec(compare_keypoints, pairArray, vector_list2)

                result_list = []
                for k in range(len(vector_list)):
                    if vector_list[k][0] != -99 and vector_list2[k][0] != -99:
                        similarity = getCosineSimilarity(vector_list[k], vector_list2[k])
                        result_list.append(similarity)
                    else:
                        result_list.append(None)

                # Get Max from result_list
                result_max = getMax(result_list)

                # Get Min from result_list
                result_min = getMin(result_list)

                # Get Sum and Len, Avg from result_list
                result_sum, result_len = getSumAndLen(result_list)
                result_avg = result_sum / result_len

                if result_avg > threshold:
                    if maxScore < result_avg:
                        maxFrame =  j
                        maxScore = result_avg
                    goodScoreList = np.append(goodScoreList, np.array([[j, round(result_avg, 5)]]), axis=0)
                
            if previousValue <= maxFrame and len(goodScoreList) > 0:
                previousValue = maxFrame
                TotalFrameLink = np.append(TotalFrameLink, np.array([[i, maxFrame, maxScore]]), axis=0) 
            elif len(goodScoreList) > 0:
                tmpValue = previousValue
                minValue = 0
                for j in range(len(goodScoreList)):
                    if previousValue <= goodScoreList[j][0] and  minValue < goodScoreList[j][1]:
                        minValue = goodScoreList[j][1]
                        tmpValue = goodScoreList[j][0]
                previousValue = tmpValue
                if minValue != 0:
                    TotalFrameLink = np.append(TotalFrameLink, np.array([[i, tmpValue, minValue]]), axis=0) 

            if len(goodScoreList) > 1:
                print("Frame " , i , "\n" , goodScoreList)
            
    else:
        for i in range(baseStart, baseEnd):
            maxScore = 100
            maxFrame = 0

            confidence_list1 = []
            goodScoreList = np.empty((0,2), float)

            base_keypoints = run_single(baseData, i)
            # print(base_keypoints)

            for i2 in range(num_kpt):
                confidence_list1.append(baseData[i][num_kpt*2 + i2])

            for j in range(compareStart, compareEnd):
                confidence_list2 = []
                compare_keypoints = run_single(compareData, j)

                summation1 = getSummation1(base_keypoints, confidence_list1)

                summation2 = getSummation2(base_keypoints, compare_keypoints, confidence_list1)

                summationResult = summation1 * summation2

                if summationResult < threshold:
                    if maxScore > summationResult:
                        maxFrame =  j
                        maxScore = summationResult
                    goodScoreList = np.append(goodScoreList, np.array([[j, round(summationResult, 3)]]), axis=0)
            if previousValue <= maxFrame and len(goodScoreList) > 0:
                previousValue = maxFrame
                TotalFrameLink = np.append(TotalFrameLink, np.array([[i, maxFrame, maxScore]]), axis=0) 
            elif len(goodScoreList) > 0:
                tmpValue = previousValue
                minValue = 100
                for j in range(len(goodScoreList)):
                    if previousValue <= goodScoreList[j][0] and  minValue > goodScoreList[j][1]:
                        minValue = goodScoreList[j][1]
                        tmpValue = goodScoreList[j][0]
                previousValue = tmpValue
                if minValue != 100:
                    TotalFrameLink = np.append(TotalFrameLink, np.array([[i, tmpValue, minValue]]), axis=0) 

            if len(goodScoreList) > 0:
                print("Frame " , i , " : \n" , goodScoreList)

    print(TotalFrameLink)

    makeCompare(baseData, TotalFrameLink, 'output/baseSimilarity.mp4', 60)
    makeCompare(compareData, TotalFrameLink, 'output/Similarity.mp4', 60, 1)

    f = open('output/FrameLink.csv', 'w', newline='')
    wr = csv.writer(f)
    for temp in TotalFrameLink:
        wr.writerow([temp[0], temp[1], temp[2]])

    f.close()


        

def makeCompare(baseData, frameLink, filename, frame, isBase=0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename,fourcc, 60.0, (256,256))
    
    for i in range(len(frameLink)):
        base_kptwindow= np.zeros((256, 256, 3), np.uint8)
        print(int(frameLink[i][isBase]))
        baseNp = getKpt(num_kpt, baseData, int(frameLink[i][isBase]))

        xMin, yMin = getDataMin(num_kpt, baseNp)
        xMax, yMax = getDataMax(num_kpt, baseNp)

        width = xMax - xMin
        height = yMax - yMin
        base_keypoints = resize_kpt(baseNp, width, height, xMin, yMin)
        
        # 점 그리기
        drawPoints(base_keypoints, base_kptwindow)

        # 선 그리기
        drawLines(base_keypoints, base_kptwindow, keyIndex.mpiiPairs)
        out.write(base_kptwindow)

    out.release()

def playCompared():
    LinkData = np.loadtxt('output/FrameLink.csv', delimiter = ',')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    baseCap = cv2.VideoCapture('input/videos/test1333.mp4')
    compareCap = cv2.VideoCapture('input/videos/compareGolf.mp4')
    ret, image = baseCap.read()
    baseOut = cv2.VideoWriter('output/BaseOut.mp4', fourcc, 60.0, (image.shape[1], image.shape[0]))
    ret, image = compareCap.read()
    compareOut = cv2.VideoWriter('output/CompareOut.mp4', fourcc, 60.0, (image.shape[1], image.shape[0]))

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
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compare keypoint position distance with demo csv log file''')
    parser.add_argument('--base', type=str, required=True, help='Base File(Ground Truth) to standard')
    parser.add_argument('--compare', type=str, required=True, help='Compare file with base')
    parser.add_argument('--start', type=int, default=0, help='Set start frame')
    parser.add_argument('--end', type=int, default=-1, help='Set end frame')
    parser.add_argument('--threshold', type=float, default=0.8, help='if you using compare all frame, set threshold. default = 0.8')
    parser.add_argument('--cosine', action='store_true', help='Use cosine similarity, Default = weight distance')
    parser.add_argument('--comparestart', type=int, default=0, help='start of compare video')
    parser.add_argument('--compareend', type=int, default=-1, help='end of compare video')
    
    
    args = parser.parse_args()

    # CompareAllFrame(args.base, args.compare, args.start, args.end, args.comparestart, args.compareend, args.threshold, keyIndex.mpiiPairs, args.cosine)
    playCompared()

    
