import numpy as np

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]

coco_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
mpiiPairs = [[8,9],[7,8], [7,12],[7,13], [11,12],[13,14], [10,11], [6,7],[2,6], [3,6],[1,2], [3,4],[0,1], [4,5], [14, 15]]

mpii_names = [
    'r_ank', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ank', 'main_hip', 'main_sho', 'neck', 'head', 'r_wri', 'r_elb', 'r_sho', 'l_sho', 'l_elb', 'l_wri'
]