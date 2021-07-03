## iou 계산
def iou(bb1, bb2):
    ## 오른쪽 좌표가 왼쪽 좌표보다 커야 하고, 위 좌표가 아래 좌표보다 커야 함 그렇지 않을 경우 asserterror
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    ## 두개의 bounding box가 겹치는 영역의 좌표
    x_left = max(bb1['x1'], bb2['x1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = max(bb1['y1'], bb2['y1'])
    y_top = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_top < y_bottom: return 0

    intersection_area = (x_right - x_left) * (y_top - y_bottom)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)

    assert iou <= 1
    assert iou >= 0
    return iou