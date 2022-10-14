import cv2
import numpy as np
import math

def resize_kpt(keypoints, width, height, x, y, target=256):

    for i in range(len(keypoints)):
        if (keypoints[i][0] != -1):
            keypoints[i][0] = int((keypoints[i][0] - x) / width * target)
            keypoints[i][1] = int((keypoints[i][1] - y) / height * target)

    return keypoints

def getVecScale(point1, point2):
    return math.sqrt(abs(pow((point2[0] - point1[0]), 2) + pow((point2[1] - point1[1]), 2)))

def getVector(point1, point2):
    # point1 to point2
    return [point2[0] - point1[0], point2[1] - point1[1]]

def getUnitVector(point1, point2):
    if getVecScale(point1, point2) != 0:
        return [(point2[0] - point1[0]) / getVecScale(point1, point2), (point2[1] - point1[1]) / getVecScale(point1, point2)]
    else:
        return [-99, -99]

def getCosineSimilarity(vec1, vec2):
    scale1 = math.sqrt(pow(vec1[0], 2) + pow(vec1[1], 2))
    scale2 = math.sqrt(pow(vec2[0], 2) + pow(vec2[1], 2))

    result = (vec1[0] * vec2[0] + vec1[1] * vec2[1]) / (scale1 * scale2)
    return result

def drawPoints(keypoints, kpt_window):
    for i in range(len(keypoints)):
        if keypoints[i][0] != -1:
            cv2.circle(kpt_window, (int(keypoints[i][0]), int(keypoints[i][1])), 3, (0, 255, 0), -1)

def drawLines(keypoints, kpt_window, kpt_id):
    for i in range(len(kpt_id)):
        _id = kpt_id[i]
        if keypoints[_id[0]][0] != -1 and keypoints[_id[1]][0] != -1:
            cv2.line(kpt_window, (int(keypoints[_id[0]][0]), int(keypoints[_id[0]][1])), \
            (int(keypoints[_id[1]][0]), int(keypoints[_id[1]][1])) , (0, 255, 0), 2)

def saveUnitVec(keypoints, kpt_id, vector_list):
    for i in range(len(kpt_id)):
        _id = kpt_id[i]
        if keypoints[_id[0]][0] != -1 and keypoints[_id[1]][0] != -1:
            vector_list.append(getUnitVector(keypoints[_id[0]], keypoints[_id[1]]))
        else:
            vector_list.append([-99, -99])

def getMax(result_list):
    result_max = -1
    for i in range(len(result_list)):
        if result_list[i] is not None:
            if result_list[i] > result_max:
                result_max = result_list[i]
    return result_max

def getMin(result_list):
    result_min = 1
    for i in range(len(result_list)):
        if result_list[i] is not None:
            if result_list[i] < result_min:
                result_min = result_list[i]
    return result_min

def getSumAndLen(result_list):
    result_sum = 0
    result_len = 0
    for i in range(len(result_list)):
        if result_list[i] is not None:
            result_sum = result_sum + result_list[i]
            result_len = result_len + 1
    return result_sum, result_len

def getSummation1(base_keypoints, heatmap_list):
    summation1 = 0
    for i in range(len(base_keypoints)):
        summation1 = summation1 + heatmap_list[i]
    return 1 / summation1

def getSummation2(base_keypoints, compare_keypoints, heatmap_list):
    summation2 = 0
    for i in range(len(base_keypoints)):
        vecScale = heatmap_list[i] * getVecScale(base_keypoints[i], compare_keypoints[i])
        summation2 = summation2 + vecScale
    return summation2

def getKpt(num_kpt, baseData, frame):
    baseNp = np.zeros((num_kpt, 2))
    for i in range(num_kpt):
        baseNp[i][0] = baseData[frame][i*2]
        baseNp[i][1] = baseData[frame][i*2+1]
    return baseNp

def getDataMin(num_kpt, baseNp):
    xMin = 10000
    yMin = 10000
    for i in range(num_kpt):
        if xMin > baseNp[i][0] and baseNp[i][0] != -1:
            xMin = baseNp[i][0]
        if yMin > baseNp[i][1] and baseNp[i][1] != -1:
            yMin = baseNp[i][1]
    return xMin, yMin

def getDataMax(num_kpt, baseNp):
    xMax = 0
    yMax = 0
    for i in range(num_kpt):
        if xMax < baseNp[i][0] and baseNp[i][0] != -1:
            xMax = baseNp[i][0]
        if yMax < baseNp[i][1] and baseNp[i][1] != -1:
            yMax = baseNp[i][1]
    return xMax, yMax
