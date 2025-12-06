#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XML 转 JSON 转换器（单张图片版本）
支持 bndbox 和 robndbox 两种标注格式
"""

import os
import xml.etree.ElementTree as ET
import json
import math

PI = 3.1415926535


def parse_xml(xml_path):
    """解析XML文件，返回图像尺寸和所有标注对象。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_info = root.find('size')
    if size_info is None:
        raise ValueError(f"XML文件 {xml_path} 缺少 <size> 标签。")
    size = (
        int(size_info.find('width').text),
        int(size_info.find('height').text),
        int(size_info.find('depth').text)
    )
    obj_info = root.findall('object')
    return size, obj_info


def get_crop_from_robndbox(robndbox):
    """从旋转框（robndbox）数据计算并返回其最小外接矩形（crop）坐标。"""
    cx = float(robndbox.find("cx").text)
    cy = float(robndbox.find("cy").text)
    w = float(robndbox.find("w").text)
    h = float(robndbox.find("h").text)
    angle = float(robndbox.find("angle").text)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    corners_x, corners_y = [], []
    for i in [-1, 1]:
        for j in [-1, 1]:
            x_corner = cx + (i * w / 2) * cos_a - (j * h / 2) * sin_a
            y_corner = cy + (i * w / 2) * sin_a + (j * h / 2) * cos_a
            corners_x.append(x_corner)
            corners_y.append(y_corner)

    return [{"xmin": min(corners_x), "ymin": min(corners_y), "xmax": max(corners_x), "ymax": max(corners_y)}]


def get_crop_from_bndbox(bndbox):
    """从标准框（bndbox）提取crop坐标。"""
    return [{
        "xmin": float(bndbox.find("xmin").text),
        "ymin": float(bndbox.find("ymin").text),
        "xmax": float(bndbox.find("xmax").text),
        "ymax": float(bndbox.find("ymax").text)
    }]


def calculate_calibrated_angle(cx_cal, cy_cal, angle_rad, bp_center):
    """根据校准后的指针头坐标和原始弧度，计算出标准化的角度值。"""
    angle_deg = angle_rad * 180 / PI

    if (cx_cal < bp_center[0]) and (cy_cal > bp_center[1]): return angle_deg - 180
    if (cx_cal < bp_center[0]) and (cy_cal < bp_center[1]): return angle_deg
    if (cx_cal > bp_center[0]) and (cy_cal < bp_center[1]): return angle_deg
    if (cx_cal > bp_center[0]) and (cy_cal > bp_center[1]): return 180 + angle_deg
    if (cx_cal < bp_center[0]) and (abs(cy_cal - bp_center[1]) < 2): return 0
    if (abs(cx_cal - bp_center[0]) < 2) and (cy_cal < bp_center[1]): return 90
    if (cx_cal > bp_center[0]) and (abs(cy_cal - bp_center[1]) < 2): return 180
    if (abs(cx_cal - bp_center[0]) < 2) and (cy_cal > bp_center[1]): return 270
    return 0


def cal_pointer_head(cx, cy, width, angle, bp_center):
    """计算指针头部的精确坐标，用于判断象限。"""
    half_w = float(width * 0.5)
    delta_x = half_w * math.cos(angle)
    delta_y = half_w * math.sin(angle)
    p1 = (cx - delta_x, cy - delta_y)
    p2 = (cx + delta_x, cy + delta_y)
    dist1_sq = (p1[0] - bp_center[0]) ** 2 + (p1[1] - bp_center[1]) ** 2
    dist2_sq = (p2[0] - bp_center[0]) ** 2 + (p2[1] - bp_center[1]) ** 2
    return p2 if dist2_sq > dist1_sq else p1


def parse_pointer_meter(obj_info, base_info):
    """处理指针式表计的逻辑 (4个及以上标注)"""
    json_info = base_info.copy()
    json_info['digit_number_info'] = {}

    bp_center, dialType = None, None
    bp_info_details = {}
    pointer_objects = []

    for obj in obj_info:
        name_tag = obj.find("name")
        if name_tag is None: continue
        name = name_tag.text

        robndbox = obj.find("robndbox")
        bndbox = obj.find("bndbox")

        # 处理表计类型标注（bj_type 或 tj_type）
        if "bj_type" in name or "tj_type" in name:
            if robndbox is not None:
                json_info["crop"] = get_crop_from_robndbox(robndbox)
            elif bndbox is not None:
                json_info["crop"] = get_crop_from_bndbox(bndbox)

            parts = name.split("-")
            # 处理 tj_type_1 格式（默认参数）
            if "tj_type" in name:
                json_info["bj_type"] = 1  # 默认为指针式
                # 设置默认值
                bp_info_details.update({
                    "unit": "MPa",  # 默认单位
                    "pointer_num": 1,  # 默认单指针
                    "pointer_type": 0  # 默认类型
                })
            else:
                # 原有的 bj_type 格式处理
                json_info["bj_type"] = int(parts[1])
                bp_info_details.update({
                    "unit": parts[2],
                    "pointer_num": int(parts[3]),
                    "pointer_type": int(parts[4])
                })

        # 处理表盘中心标注（支持 bndbox 和 robndbox）
        elif name.startswith(('o', 'O')):
            dialType = int(name.split("-")[-1])

            # 支持两种格式：robndbox 和 bndbox
            if robndbox is not None:
                # 旋转框格式：直接使用中心点
                bp_center = [int(float(robndbox.find("cx").text)), int(float(robndbox.find("cy").text))]
                print(f"   ✓ 表盘中心(robndbox): cx={bp_center[0]}, cy={bp_center[1]}")
            elif bndbox is not None:
                # 标准框格式：计算中心点
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                bp_center = [cx, cy]
                print(f"   ✓ 表盘中心(bndbox): cx={cx}, cy={cy}")
            else:
                print(f"   ⚠️ 警告：表盘中心标注既没有robndbox也没有bndbox")

        # 处理数显行数标注
        elif 'row_num' in name:
            json_info['digit_number_info'] = {'row_num': int(name.split("-")[-1])}

        # 其他标注（刻度点等）
        else:
            pointer_objects.append(obj)

    if not bp_center:
        raise ValueError("XML文件中未找到有效的圆心(o)标注。")

    angleMeasures = []
    add_counter = 1
    start_pointer_data, final_pointer_data = None, None

    for obj in pointer_objects:
        name_tag, robndbox = obj.find("name"), obj.find("robndbox")
        if name_tag is None or robndbox is None:
            continue

        name = name_tag.text
        measure_str = name.split("_")[-1]

        pointer_data = {
            "name": name, "measure": float(measure_str),
            "cx": float(robndbox.find("cx").text), "cy": float(robndbox.find("cy").text),
            "w": float(robndbox.find("w").text), "h": float(robndbox.find("h").text),
            "angle1": float(robndbox.find("angle").text)
        }

        head_cx, head_cy = cal_pointer_head(pointer_data["cx"], pointer_data["cy"], pointer_data["w"],
                                            pointer_data["angle1"], bp_center)
        pointer_data["angle"] = round(calculate_calibrated_angle(head_cx, head_cy, pointer_data["angle1"], bp_center),
                                      2)

        if "start" in name:
            start_pointer_data = pointer_data
        elif "final" in name:
            final_pointer_data = pointer_data
        else:
            pointer_data["name"] = f"add{add_counter}_{measure_str}"
            add_counter += 1
            angleMeasures.append(pointer_data)

    if start_pointer_data and final_pointer_data:
        if abs(final_pointer_data["angle"] - start_pointer_data["angle"]) < 10 or final_pointer_data["angle"] < \
                start_pointer_data["angle"]:
            start_pointer_data["angle"] -= 360
        angleMeasures.insert(0, final_pointer_data)
        angleMeasures.insert(0, start_pointer_data)
    elif start_pointer_data:
        angleMeasures.insert(0, start_pointer_data)
    elif final_pointer_data:
        angleMeasures.insert(0, final_pointer_data)

    bp_info_details.update({"bp_center": bp_center, "dialType": dialType, "angleMeasures": angleMeasures})
    json_info["bp_info"] = [bp_info_details]
    return json_info


def parse_other_meter(obj_info, base_info):
    """处理数显或刻度式表计的逻辑 (1或2个标注)"""
    json_info = base_info.copy()

    for obj in obj_info:
        name_tag = obj.find("name")
        if name_tag is None: continue
        name = name_tag.text

        robndbox, bndbox = obj.find("robndbox"), obj.find("bndbox")

        if "bj_type" in name or "tj_type" in name:
            # 处理 tj_type 格式
            if "tj_type" in name:
                json_info["bj_type"] = int(name.split("_")[-1])  # tj_type_1 → bj_type=1
            else:
                json_info["bj_type"] = int(name.split("-")[1])

            if robndbox is not None:
                json_info["crop"] = get_crop_from_robndbox(robndbox)
            elif bndbox is not None:
                json_info["crop"] = get_crop_from_bndbox(bndbox)
        elif 'row_num' in name:
            json_info.setdefault('digit_number_info', {})['row_num'] = int(name.split("-")[-1])

    return json_info


def convert_single_xml_to_json(xml_path):
    """
    将单个XML文件转换为JSON格式

    Args:
        xml_path: XML文件路径

    Returns:
        dict: JSON格式的数据，如果失败返回None
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"正在转换: {os.path.basename(xml_path)}")
        print(f"{'=' * 60}")

        size, obj_info = parse_xml(xml_path)
        obj_count = len(obj_info)
        print(f"   图片尺寸: {size[0]}x{size[1]}")
        print(f"   标注数量: {obj_count}")

        # 从文件名提取pointID
        xml_name = os.path.basename(xml_path)
        pointID_full = xml_name.split(".")[0]

        base_info = {
            "pointID": pointID_full.split("_")[-1] if "_" in pointID_full else pointID_full,
            "point_name": pointID_full.split("_")[0] if "_" in pointID_full else pointID_full,
            "image_width": size[0],
            "image_height": size[1],
        }

        if obj_count >= 4:
            print("   类型: 指针式表计")
            json_data = parse_pointer_meter(obj_info, base_info)
        elif obj_count in [1, 2]:
            print("   类型: 数显/液位表计")
            json_data = parse_other_meter(obj_info, base_info)
        else:
            print(f"   ⚠️ 警告：标注数量为 {obj_count}，不符合处理规则")
            return None

        print(f"{'=' * 60}\n")
        return json_data

    except Exception as e:
        print(f"❌ 转换XML失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None