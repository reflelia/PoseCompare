# PoseCompare

## How to use

### 1. compareVideo.py

```
python compareVideo.py --base inputs/example.csv --compare inputs/example3.csv --start 0 --end 100 
```

If two csv files have different FrameRates

```
python compareVideo.py --base inputs/example.csv --compare inputs/example3.csv --start 0 --end 100 --comparerate 60
```

If you want to use Cosine similarity (Default : Google Move Mirror Weighted Distance)
```
python compareVideo.py --base inputs/example.csv --compare inputs/example3.csv --start 0 --end 100 --comparerate 60 --cosine
```

### 2. compareAll.py
```
python compareAll.py --base inputs/example.csv --compare inputs/example3.csv --threshold 30
```

If you want to decide the range
```
python compareAll.py --base inputs/example.csv --compare inputs/example3.csv --threshold 30 --start 0 --end 0 --comparestart 0 --compareend 0
```

If you want to use Cosine similarity,
```
python compareAll.py --base inputs/example.csv --compare inputs/example3.csv --threshold 0.9 --start 0 --end 0 --comparestart 0 --compareend 0 --cosine
```

### 3.If you want to create an original video with FrameLink.csv extracted from CompareAll.py

```
python playCompared.py --base base.mp4 --compare compare.mp4 --link output/FrameLink.csv
```

All results will be stored in output.

## How is input csv configured?
The input csv files contain the x,y coordinates and Confidence Score of the keypoint on a single line.  
You can also compare poses by extracting csvs such as the above form from other pose estimation algorithms.  
