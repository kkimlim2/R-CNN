from iou import iou
import xml.etree.ElementTree as ET  ##xml 파일 읽어주는 라이브러리
import numpy as np
import os
import cv2


def region_proposal(mode):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  ## selective search initialize

    train_images = []
    train_labels = []

    if mode == 'finetune':
        threshold = 0.5
    elif mode == 'classify':
        threshold = 0.3
    elif mode == 'test':
        threshold = 0

    ## xml 파싱해서 정보 읽어들이기
    ## xml 파싱을 통해서 이미지 당 ground_truth 상자 만듦

    classes = {'RBC': '1', "WBC": '2', "Platelets": '3'}

    for e, i in enumerate(os.listdir('data/content/BCCD/train/annos')):

        xml = open("data/content/BCCD/train/annos/" + i, "r")
        tree = ET.parse(xml)
        root = tree.getroot()

        ground_truth = []
        size = root.find("size")

        objects = root.findall("object")

        for object_ in objects:
            class_ = object_.find("name").text
            bndbox = object_.find("bndbox")
            xmin = bndbox.find("xmin").text
            xmax = bndbox.find("xmax").text
            ymin = bndbox.find("ymin").text
            ymax = bndbox.find("ymax").text
            ground_truth.append(
                ({"x1": float(xmin), 'x2': float(xmax), 'y1': float(ymin), 'y2': float(ymax)}, classes[class_]))

        path_img = 'data/content/BCCD/train/images/' + i.split('xml')[0] + 'jpg'
        image = cv2.imread(path_img)
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()

        imgs, labels = pos_neg_region(image, ssresults, ground_truth, threshold)

        train_images += imgs
        train_labels += labels

    return train_images, train_labels


def resize(image, x, y, w, h):
    img = image.copy()[max(y - 16, 0):min(image.shape[0], y + h + 16),
          max(x - 16, 0):min(x + w + 16, image.shape[1])]
    np.pad(img, (
        (max(0, 16 - y), max(0, image.shape[1] - y - h)), (max(0, 16 - x), max(0, image.shape[1] - x - w)),
        (0, 0)),
           mode='constant', constant_values=(0, 0))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)  # padding 후 resize

    return img


def pos_neg_region(image, ssresults, ground_truth, threshold):

    train_images = []
    train_labels = []

    p_num = 0
    n_num = 0

    for k, candidates in enumerate(ssresults):
        if p_num == 30 and n_num == 30: break

        for box, title in ground_truth:
            x, y, w, h = candidates

            ## cv.imread의 경우 (y, x, 채널) 순으로 차원 구성됨
            ## 원본 image에서 iou가 0.5이상인 경우 positive로 판단하고 해당 영역을 cnn의 input 사이즈에 맞게 변형하기
            ## Appendix A를 참고하여 image context를 고려하지 않고 16pixel씩 패딩하기 16픽셀시 패딩할 수 없을 경우, zerro 패딩

            img = resize(image, x, y, w, h)

            # 30 개씩만 일단 하기(원래대로라면 제한 없어야 함)

            if iou(box, ({'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h})) > threshold:
                if p_num <= 30:
                    train_images.append(img)
                    train_labels.append(int(title))
                    p_num += 1

            else:
                if n_num <= 30:
                    train_images.append(img)
                    train_labels.append(0)
                    n_num += 1

    return train_images, train_labels
