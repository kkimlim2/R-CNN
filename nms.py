import numpy as np
from iou import iou

def box_dict(box):
    x, y, w, h = box
    return {'x1': x, 'x2': x+w,'y1': y,'y2':y+h}

def nms(bb, score, predict):
    ## argument
    ## bb : bb[i] = [x,y,w,h] 인 리스트
    ## score: 각 bb의 score이 담긴 리스트

    nms_list = [True] * len(bb)

    assert len(bb) == len(score), "bb의 수와 score의 수가 다름"

    bb = np.array(bb)
    score = np.array(score)
    predict = np.array(predict)

    order = score.argsort()
    bb = bb[order][::-1]
    score = score[order][::-1]
    predict = predict[order][::-1]

    for e, box in enumerate(bb):
        if (nms_list[e]):
            bb1 = box_dict(box)
            predict1 = predict[e]

            for i in range(len(bb)-e-1):
                if predict[i] == predict1:
                    bb2 = box_dict(bb[i])
                    if iou(bb1, bb2) > 0.6:
                        nms_list[i] = False

    return np.asarray(nms_list)[order][::-1]







