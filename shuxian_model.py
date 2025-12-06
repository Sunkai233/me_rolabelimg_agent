#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

from paddleocr import PaddleOCR


def correct_ocr_digits(text):
    correction_map = {
        'l': '1', 'L': '1', 'I': '1', 'i': '1', '|': '1', '/': '1',
        'O': '0', 'o': '0', 'Q': '0', 'D': '0', '°': '0',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5', '$': '5',
        'b': '6',
        'B': '8', '&': '8', 'R': '8',
        'q': '9', 'g': '9',
        'T': '7', 't': '7',
        "日": "8", "曰": "8",
        'A': '4', 'a': '4',
        "口": "0",
    }

    # 先移除常见干扰字符
    remove_chars = [' ', ',', ':', ';', '-', '_', '*', '+', '=', '~', '`', '  ']
    cleaned_text = ''.join([c for c in text if c not in remove_chars])

    # 字符替换
    corrected_text = []
    for char in cleaned_text:
        if char in correction_map:
            corrected_text.append(correction_map[char])
        elif char.isdigit():
            corrected_text.append(char)

    return ''.join(corrected_text)


def model_init():
    '''
    predictor = PaddleOCR(lang='en',
                          use_doc_orientation_classify=False,
                          use_doc_unwarping=False,
                          use_textline_orientation=False,
                          text_det_limit_type='max',
                          text_det_limit_side_len=150 )  # 降低阈值适应低对比度
    # predictor = PaddleOCR(use_angle_cls=True)  # 初始化时需启用分类（cls=True）
    '''
    demo_dir = os.path.abspath(os.path.dirname(__file__))
    paddlex_root = os.path.join(demo_dir, 'model', 'official_models')

    det_model_dir = os.path.join(paddlex_root, 'PP-OCRv5_server_det')
    rec_model_dir = os.path.join(paddlex_root, 'en_PP-OCRv5_mobile_rec')
    cls_model_dir = os.path.join(paddlex_root, 'PP-LCNet_x1_0_textline_ori')
    predictor = PaddleOCR(
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        cls_model_dir=cls_model_dir,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    return predictor


def preprocess_meter_image(img):
    """预处理表计图像"""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # 二值化
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def detect_digital_region(img):
    """检测数字区域，提高识别精度"""
    # 边缘检测
    edges = cv2.Canny(img, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选可能包含数字的区域
    digit_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        # 根据长宽比和面积筛选数字区域
        if 0.2 < aspect_ratio < 5.0 and area > 50:
            digit_contours.append(contour)

    # 创建掩码
    mask = np.zeros_like(img)
    cv2.drawContours(mask, digit_contours, -1, (255), thickness=cv2.FILLED)

    # 应用掩码
    result = cv2.bitwise_and(img, img, mask=mask)

    return result


def preprocess_image(ori_temp):
    max_limit = 150

    w = ori_temp.shape[1]
    h = ori_temp.shape[0]

    ratio = float(max_limit) / w
    new_h = int(h * ratio)

    temp_image = cv2.resize(ori_temp, (max_limit, new_h))

    return temp_image


def model_inference(predictor, img):
    ori_temp = img

    img_temp = preprocess_image(ori_temp)

    cv2.imwrite("temp.jpg", img_temp)

    try:

        results = predictor.predict("temp.jpg")

        result = results[0]

        readings = result['rec_texts']

        rec_polys = result['rec_polys']
        size_list = []

        if len(readings) > 0:
            for poly in rec_polys:
                x_list = []
                y_list = []

                for point in poly:
                    x_list.append(point[0])
                    y_list.append(point[1])

                    min_x = min(x_list)
                    max_x = max(x_list)
                    min_y = min(y_list)
                    max_y = max(y_list)

                size = float(max_x - min_x) * float(max_y - min_y)
                size_list.append(size)

                max_size = max(size_list)
                index = size_list.index(max_size)
                reading = readings[index]

                cleaned_number = correct_ocr_digits(reading)
                new_result = float(cleaned_number)

        else:
            new_result = 0.0  # 默认值


    except Exception as e:
        # print(f"OCR 处理失败: {e}")
        new_result = 0.0  # 异常时的默认值

    return new_result
