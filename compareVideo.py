import numpy as np
import csv
import math
import argparse
from utils import resize_kpt, drawPoints, drawLines, saveUnitVec, getCosineSimilarity, getMax, getMin, getSumAndLen, getSummation1, getSummation2, getKpt, getDataMin, getDataMax
import keyIndex
import cv2

kpt_names = keyIndex.mpii_names

num_kpt = len(kpt_names)

def run_single(baseData, frame, pairArray):
    vector_list = []
    confidence_list = []

    kptWindow = np.zeros((256, 256, 3), np.uint8)
    baseNp = getKpt(num_kpt, baseData, frame)

    # get bbox
    xMin, yMin = getDataMin(num_kpt, baseNp)
    xMax, yMax = getDataMax(num_kpt, baseNp)

    width = xMax - xMin
    height = yMax - yMin
    keypoints = resize_kpt(baseNp, width, height, xMin, yMin)

    # 점 그리기
    drawPoints(keypoints, kptWindow)

    # 선 그리기
    drawLines(keypoints, kptWindow, pairArray)

    # 단위 벡터 저장
    saveUnitVec(keypoints, pairArray, vector_list)

    for i in range(num_kpt):
        confidence_list.append(baseData[frame][num_kpt*2 + i])

    return vector_list, confidence_list, kptWindow, keypoints

def run(baseData, compareData, frame, isBaseBig, frameRatio, start, cosine, pairArray):

    targetFrame = int(((frame - start) * frameRatio) + start)

    if isBaseBig == 1:
        vector_list, confidence_list, base_kptwindow , base_keypoints= run_single(baseData, targetFrame, pairArray)
        vector_list2, confidence_list2, compare_kptwindow, compare_keypoints = run_single(compareData, frame, pairArray)
    else:
        vector_list, confidence_list, base_kptwindow, base_keypoints = run_single(baseData, frame, pairArray)
        vector_list2, confidence_list2, compare_kptwindow, compare_keypoints = run_single(compareData, targetFrame, pairArray)


    if cosine:
        # cosine 유사도 구한 후 저장
        result_list = []
        print("<--------------------------------->")
        for i in range(len(vector_list)):
            if vector_list[i][0] != -99 and vector_list2[i][0] != -99:
                similarity = getCosineSimilarity(vector_list[i], vector_list2[i])
                result_list.append(similarity)
                print('Cosine similarity - ' + kpt_names[pairArray[i][0]] + " -> " + kpt_names[pairArray[i][1]] + " : \t", similarity)
            else:
                result_list.append(None)
                print('Cosine similarity - ' + kpt_names[pairArray[i][0]] + " -> " + kpt_names[pairArray[i][1]] + " : \t Can't compare")

        # Get Max from result_list
        result_max = getMax(result_list)
        print("Max : ", result_max)

        # Get Min from result_list
        result_min = getMin(result_list)
        print("Min : ", result_min)

        # Get Sum and Len, Avg from result_list
        result_sum, result_len = getSumAndLen(result_list)
        print("Average (Except can't compared): ", result_sum/result_len)
        

    else:
        confidence_list1 = []
        confidence_list2 = []
        for i in range(num_kpt):
            confidence_list1.append(baseData[frame][num_kpt*2 + i])
            confidence_list2.append(compareData[frame][num_kpt*2 + i])

        print("<--------------------------------->")
        summation1 = getSummation1(base_keypoints, confidence_list1)
        print('summation1 : ', summation1)

        summation2 = getSummation2(base_keypoints, compare_keypoints, confidence_list1)
        print('summation2 : ', summation2)

        print('moveMirror : ', summation1 * summation2)
    

    cv2.imshow('Original keypoints', base_kptwindow)
    cv2.imshow('Compare Keypoints', compare_kptwindow)
    
    key = cv2.waitKey(30)
    
    if cosine:
        returnValue = result_sum/result_len
    else:
        returnValue = summation1 * summation2
    return returnValue, base_kptwindow, compare_kptwindow

def main(base, compare, start, end, baseFrame, compareFrame, cosine):

    baseData = np.loadtxt(base, delimiter = ',')
    compareData = np.loadtxt(compare, delimiter = ',')

    # count of frame
    baseCount = baseData.shape[0]
    compareCount = compareData.shape[0]

    print('Base Frame : ', baseCount)
    print('Compare Frame : ', compareCount)

    frameRatio = 0
    isBaseBig = 0
    frameRatio = baseFrame / compareFrame
    if frameRatio < 1:
        isBaseBig = 0
        frameRatio = 1 / frameRatio

        assert baseCount > end
        assert compareCount > end * frameRatio
    else:
        isBaseBig = 1
        assert baseCount > end * frameRatio
        assert compareCount > end

    assert start < end


    total_list = []
    frame_array = []
    frame_array2 = []

    for i in range(start, end):
        score, FrameBase, FrameCompare = run(baseData, compareData, i, isBaseBig, frameRatio, start, cosine, keyIndex.mpiiPairs)
        total_list.append(score)
        frame_array.append(FrameBase)
        frame_array2.append(FrameCompare)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/base.mp4',fourcc, 30.0, (256,256))
    out2 = cv2.VideoWriter('output/compare.mp4', fourcc, 30.0, (256,256))

    for i in range(len(frame_array)):
        out.write(frame_array[i])
        out2.write(frame_array2[i])

    out.release()
    out2.release()

    print('총 점수 : ', sum(total_list) / len(total_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compare keypoint position distance with demo csv log file''')
    parser.add_argument('--base', type=str, required=True, help='Base File(Ground Truth) to standard')
    parser.add_argument('--compare', type=str, required=True, help='Compare file with base')
    parser.add_argument('--start', type=int, required=True, help='Set start frame')
    parser.add_argument('--end', type=int, required=True, help='Set end frame')
    parser.add_argument('--baserate', type=int, default=30, help='Framerate of base video')
    parser.add_argument('--comparerate', type=int, default=30, help='Framerate of compare video')
    parser.add_argument('--cosine', action='store_true', help='Use cosine similarity, Default = weight distance')
    
    args = parser.parse_args()

    main(args.base, args.compare, args.start, args.end, args.baserate, args.comparerate, args.cosine)

    
