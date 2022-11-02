# PoseCompare

## How to use

### 1. compareVideo.py
#### Comparing each frame while playing the video

```
python compareVideo.py --base input/example.csv --compare input/example3.csv --start 0 --end 100 
```

If two csv files have different FrameRates

```
python compareVideo.py --base input/example.csv --compare input/example3.csv --start 0 --end 100 --comparerate 60
```

If you want to use Cosine similarity (Default : Google Move Mirror Weighted Distance)
```
python compareVideo.py --base input/example.csv --compare input/example3.csv --start 0 --end 100 --comparerate 60 --cosine
```

### 2. compareAll.py
#### Comparing all frames in two videos
```
python compareAll.py --base input/example.csv --compare input/example3.csv --threshold 30
```

If you want to decide the range
```
python compareAll.py --base input/example.csv --compare input/example3.csv --threshold 30 --start 0 --end 100 --comparestart 0 --compareend 100
```

If you want to use Cosine similarity,
```
python compareAll.py --base input/example.csv --compare input/example3.csv --threshold 0.9 --start 0 --end 100 --comparestart 0 --compareend 100 --cosine
```

### 3. playCompared.py
#### Create an original video with FrameLink.csv extracted from CompareAll.py

```
python playCompared.py --base base.mp4 --compare compare.mp4 --link output/FrameLink.csv
```

All results will be stored in output.

## To connect two images using FrameLink
Connect the frames with the highest similarity between the two images.  
However, if the connected image precedes the previous frame, connect the frame with the highest similarity among the frames behind the frame

![image](https://user-images.githubusercontent.com/28883305/196123213-db66ed72-9be6-46b1-aa18-8fae0815e4f8.png)

## How is input csv configured?
The input csv files contain the x,y coordinates and Confidence Score of the keypoint on a single line.  
You can also compare poses by extracting csvs such as the above form from other pose estimation algorithms.  
