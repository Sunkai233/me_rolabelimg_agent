#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import math
import json
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import logging

# ====== 彻底禁用 PyTorch 安全检查 ======
import torch

# 方法1：禁用全局类型检查
try:
    import torch._weights_only_unpickler as _unpickler

    _unpickler.GLOBAL_ALLOWED_TYPES = None
    print("✓ 已禁用 PyTorch 安全检查")
except Exception as e:
    print(f"警告：无法禁用安全检查: {e}")

# 方法2：Monkey patch torch.load
_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False  # 强制禁用 weights_only
    return _original_torch_load(f, *args, **kwargs)


torch.load = _patched_torch_load
print("✓ 已 Patch torch.load")

# 现在安全地导入其他模块
import zhizhen_model
import youwei_model
import shuxian_model
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= 辅助函数 =============

def get_rotated_rect_vertices(cx: float, cy: float, w: float, h: float, angle: float) -> List[Tuple[float, float]]:
    """计算旋转矩形的四个顶点"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    half_w = w / 2
    half_h = h / 2

    dx = [-half_w, half_w, half_w, -half_w]
    dy = [-half_h, -half_h, half_h, half_h]

    vertices = []
    for i in range(4):
        x_rot = dx[i] * cos_a - dy[i] * sin_a
        y_rot = dx[i] * sin_a + dy[i] * cos_a
        x = cx + x_rot
        y = cy + y_rot
        vertices.append((x, y))

    return vertices


def determine_direction_point(center: Tuple[float, float],
                              vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """确定方向点（距离表盘中心最远的窄边中点）"""
    cx_o, cy_o = center

    # 计算四个边的中点
    mid_points = []
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        mid_points.append((mid_x, mid_y))

    # 计算各边长度，找出窄边
    edge_lengths = []
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        edge_lengths.append(length)

    # 找出最短的两条边
    sorted_indices = sorted(range(len(edge_lengths)), key=lambda i: edge_lengths[i])
    narrow_edge_indices = sorted_indices[:2]

    # 获取窄边的中点
    narrow_mid_points = [mid_points[i] for i in narrow_edge_indices]

    # 计算窄边中点到表盘中心的距离，找出距离最远的点
    max_distance = -1
    direction_point = narrow_mid_points[0]

    for point in narrow_mid_points:
        px, py = point
        distance = math.sqrt((px - cx_o) ** 2 + (py - cy_o) ** 2)
        if distance > max_distance:
            max_distance = distance
            direction_point = point

    return direction_point


def calculate_angle(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """计算两点之间的角度（弧度）"""
    dx = cx2 - cx1
    dy = cy1 - cy2
    angle = math.atan2(dy, dx)

    if angle >= 0:
        angle = angle
    else:
        angle = angle + 2 * math.pi

    return angle


def extract_key_points_from_config(meter_config: Dict) -> Dict:
    """从JSON配置中提取关键点信息（使用窄边中点）"""
    key_points = {}

    print("🔍 开始提取关键点...")
    print(f"   meter_config 包含的键: {list(meter_config.keys())}")

    # 检查 bp_info 是否存在
    if 'bp_info' not in meter_config:
        print("❌ meter_config 中没有 'bp_info'")
        return key_points

    bp_info_list = meter_config['bp_info']
    if not bp_info_list or len(bp_info_list) == 0:
        print("❌ bp_info 列表为空")
        return key_points

    bp_info = bp_info_list[0]
    print(f"   bp_info 包含的键: {list(bp_info.keys())}")

    # 表盘中心信息
    if 'bp_center' in bp_info:
        bp_center = bp_info['bp_center']
        key_points['o'] = {
            'cx': bp_center[0],
            'cy': bp_center[1],
            'w': 0,
            'h': 0,
            'angle': 0,
            'value': None
        }
        print(f"   ✓ 提取表盘中心: cx={bp_center[0]}, cy={bp_center[1]}")
    else:
        print("   ❌ bp_info 中没有 'bp_center'")

    # 获取表盘中心坐标
    center_point = (key_points['o']['cx'], key_points['o']['cy']) if 'o' in key_points else None

    # 刻度点信息
    if 'angleMeasures' not in bp_info:
        print("   ❌ bp_info 中没有 'angleMeasures'")
        return key_points

    angle_measures = bp_info['angleMeasures']
    if not angle_measures:
        print("   ❌ angleMeasures 列表为空")
        return key_points

    print(f"   刻度点数量: {len(angle_measures)}")

    key_points['angle_measures'] = []

    for i, measure in enumerate(angle_measures):
        try:
            name = measure.get('name', '')
            cx = measure['cx']
            cy = measure['cy']
            w = measure['w']
            h = measure['h']
            angle = measure['angle1']

            # 计算刻度框的四个顶点
            vertices = get_rotated_rect_vertices(cx, cy, w, h, angle)

            # 确定刻度点方向点（窄边中点）
            if center_point:
                direction_point = determine_direction_point(center_point, vertices)
            else:
                direction_point = (cx, cy)

            # 计算刻度点相对于表盘中心的角度
            if center_point:
                angle_rad = calculate_angle(center_point[0], center_point[1],
                                            direction_point[0], direction_point[1])
                angle_deg = math.degrees(angle_rad)
            else:
                angle_deg = measure.get('angle', 0)

            # 存储刻度点信息
            point_info = {
                'name': name,
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'angle': angle_deg,
                'value': measure['measure'],
                'effective_point': direction_point
            }

            key_points['angle_measures'].append(point_info)

            # 单独存储起始和结束刻度点
            if name.startswith('start'):
                key_points['start'] = point_info
                print(f"   ✓ 提取起始刻度: {name}, 值={measure['measure']}")
            elif name.startswith('final'):
                key_points['final'] = point_info
                print(f"   ✓ 提取终止刻度: {name}, 值={measure['measure']}")

        except Exception as e:
            print(f"   ⚠️ 处理刻度点 {i} 时出错: {str(e)}")
            continue

    # 添加最小值和最大值
    if 'angleMeasures' in bp_info and bp_info['angleMeasures']:
        try:
            min_value = min(m['measure'] for m in bp_info['angleMeasures'])
            max_value = max(m['measure'] for m in bp_info['angleMeasures'])
            key_points['min_value'] = min_value
            key_points['max_value'] = max_value
            print(f"   ✓ 量程: {min_value} ~ {max_value}")
        except Exception as e:
            print(f"   ⚠️ 计算量程时出错: {str(e)}")

    print(f"✅ 关键点提取完成，共 {len(key_points)} 个")
    return key_points


def apply_crop_transform(points: Dict, crop: Dict) -> Dict:
    """将关键点坐标从原图转换到裁剪图坐标系"""
    xmin = crop['xmin']
    ymin = crop['ymin']

    transformed = {}
    for key, point in points.items():
        if key in ['min_value', 'max_value']:
            transformed[key] = point
            continue

        if isinstance(point, dict):
            new_point = point.copy()
            if 'cx' in new_point:
                new_point['cx'] -= xmin
            if 'cy' in new_point:
                new_point['cy'] -= ymin

            # 处理有效点
            if 'effective_point' in new_point:
                eff_x, eff_y = new_point['effective_point']
                new_point['effective_point'] = (eff_x - xmin, eff_y - ymin)

            transformed[key] = new_point
        elif isinstance(point, list):
            # 处理列表（如 angle_measures）
            transformed[key] = []
            for item in point:
                if isinstance(item, dict):
                    new_item = item.copy()
                    if 'cx' in new_item:
                        new_item['cx'] -= xmin
                    if 'cy' in new_item:
                        new_item['cy'] -= ymin
                    if 'effective_point' in new_item:
                        eff_x, eff_y = new_item['effective_point']
                        new_item['effective_point'] = (eff_x - xmin, eff_y - ymin)
                    transformed[key].append(new_item)
        else:
            transformed[key] = point

    return transformed


def zhizhen_postprocess_debug(pointers, scores, key_points, CONFIG, meter_config):
    """调试版本的 zhizhen_postprocess"""

    print("\n" + "=" * 60)
    print("🔍 zhizhen_postprocess 调试信息")
    print("=" * 60)

    # 检查关键点
    required_keys = ['o', 'start', 'final']
    missing_keys = [k for k in required_keys if k not in key_points]

    print(f"📋 关键点检查:")
    print(f"   需要的关键点: {required_keys}")
    print(f"   当前关键点: {list(key_points.keys())}")
    print(f"   缺失的关键点: {missing_keys if missing_keys else '无'}")

    if missing_keys:
        print(f"❌ 失败原因: 缺少关键点 {missing_keys}")
        print("=" * 60 + "\n")
        return "分析失败：表盘模糊未找到指针"

    # 检查指针
    print(f"\n🎯 指针检测结果:")
    print(f"   检测到的指针数量: {len(pointers) if pointers else 0}")

    if not pointers:
        print(f"❌ 失败原因: 未检测到任何指针")
        print("=" * 60 + "\n")
        return "分析失败：表盘模糊未找到指针"

    # 如果有指针，打印详细信息
    print(f"   指针置信度:")
    for i, score in enumerate(scores):
        print(f"      指针{i + 1}: {score:.4f}")

    # 打印关键点详细信息
    print(f"\n📍 关键点详细信息:")
    print(f"   表盘中心(o): cx={key_points['o']['cx']:.2f}, cy={key_points['o']['cy']:.2f}")
    print(f"   起始刻度(start): {key_points['start'].get('name', 'unknown')}")
    print(f"   终止刻度(final): {key_points['final'].get('name', 'unknown')}")

    # 获取配置信息
    min_val = key_points.get('min_value', CONFIG['min_value'])
    max_val = key_points.get('max_value', CONFIG['max_value'])
    print(f"   量程: {min_val} ~ {max_val}")

    # 调用原来的处理逻辑
    try:
        reading = zhizhen_model.handle_multiple_pointers(
            pointers=pointers,
            scores=scores,
            center=key_points['o'],
            start=key_points['start'],
            end=key_points['final'],
            min_val=min_val,
            max_val=max_val,
            direction=CONFIG['direction'],
            meter_config=meter_config,
            key_points=key_points
        )

        print(f"\n📊 计算结果:")
        print(f"   原始读数: {reading}")

        if reading is not None:
            bp_info = meter_config.get('bp_info', [{}])[0]
            unit = bp_info.get('unit', '')
            dial_type = bp_info.get('dialType', 2)
            pointer_num = bp_info.get('pointer_num', 1)

            is_360_dual_pointer = (dial_type == 1 and unit == "次" and pointer_num == 2)

            if is_360_dual_pointer:
                display_reading = str(reading)
            else:
                if unit in ['档', '次']:
                    reading = round(reading)
                else:
                    reading = round(reading, 2)
                display_reading = str(reading)

            print(f"   最终读数: {display_reading} {unit}")
            print(f"✅ 成功")
            print("=" * 60 + "\n")
            return display_reading
        else:
            print(f"❌ 失败原因: reading 为 None")
            print("=" * 60 + "\n")
            return "分析失败：表盘模糊未找到指针"

    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        return "分析失败：表盘模糊未找到指针"


# ============= 主要函数 =============

def detect_dial_boundary(model: YOLO, image: np.ndarray, conf_thresh: float = 0.5,
                         target_class_name: str = None, json_crop_info: Dict = None) -> Optional[Dict]:
    """使用YOLO模型检测表盘外框"""
    try:
        results = model(image)

        if len(results) == 0 or results[0].boxes is None:
            logger.warning("未检测到表盘外框")
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        if target_class_name is not None:
            return filter_detections_by_type(model, boxes, confidences, class_ids,
                                             target_class_name, conf_thresh, image, json_crop_info)

        if len(confidences) == 0:
            return None

        # 计算参考中心点
        if json_crop_info is not None:
            ref_center_x = (json_crop_info['xmin'] + json_crop_info['xmax']) / 2
            ref_center_y = (json_crop_info['ymin'] + json_crop_info['ymax']) / 2
        else:
            height, width = image.shape[:2]
            ref_center_x = width / 2
            ref_center_y = height / 2

        # 计算每个检测框到参考中心点的距离
        distances = []
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            if conf < conf_thresh:
                continue

            x1, y1, x2, y2 = box
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2

            distance = math.sqrt((box_center_x - ref_center_x) ** 2 + (box_center_y - ref_center_y) ** 2)
            distances.append((i, distance, conf))

        if not distances:
            return None

        distances.sort(key=lambda x: x[1])
        best_idx = distances[0][0]

        x1, y1, x2, y2 = boxes[best_idx]
        xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)

        height, width = image.shape[:2]
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(xmin + 1, min(xmax, width))
        ymax = max(ymin + 1, min(ymax, height))

        return {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }

    except Exception as e:
        logger.error(f"表盘检测出错: {str(e)}")
        return None


def filter_detections_by_type(model: YOLO, boxes: np.ndarray, confidences: np.ndarray,
                              class_ids: np.ndarray, target_class_name: str, conf_thresh: float,
                              image: np.ndarray = None, json_crop_info: Dict = None) -> Optional[Dict]:
    """根据目标类别名称过滤检测结果"""
    try:
        class_names = model.names

        target_class_id = None
        for class_id, class_name in class_names.items():
            if class_name == target_class_name:
                target_class_id = class_id
                break

        if target_class_id is None:
            return None

        target_indices = []
        for i, class_id in enumerate(class_ids):
            if class_id == target_class_id and confidences[i] >= conf_thresh:
                target_indices.append(i)

        if not target_indices:
            return None

        # 计算参考中心点
        if json_crop_info is not None:
            ref_center_x = (json_crop_info['xmin'] + json_crop_info['xmax']) / 2
            ref_center_y = (json_crop_info['ymin'] + json_crop_info['ymax']) / 2
        elif image is not None:
            height, width = image.shape[:2]
            ref_center_x = width / 2
            ref_center_y = height / 2
        else:
            ref_center_x = 960
            ref_center_y = 540

        distances = []
        for idx in target_indices:
            x1, y1, x2, y2 = boxes[idx]
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            distance = math.sqrt((box_center_x - ref_center_x) ** 2 + (box_center_y - ref_center_y) ** 2)
            distances.append((idx, distance, confidences[idx]))

        if distances:
            distances.sort(key=lambda x: x[1])
            best_idx = distances[0][0]

            x1, y1, x2, y2 = boxes[best_idx]
            xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)

            return {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
        else:
            return None

    except Exception as e:
        logger.error(f"过滤检测结果时出错: {str(e)}")
        return None


def get_target_class_name(bj_type: int) -> str:
    """根据表计类型ID获取对应的类别名称"""
    type_mapping = {
        1: "zhizhen",
        5: "youwei",
        6: "shuxian"
    }
    return type_mapping.get(bj_type, "zhizhen")


def load_json_config(json_path: str) -> Tuple[List[Dict], List[str], Dict[str, float]]:
    if not os.path.exists(json_path):
        logger.error(f"JSON配置文件不存在 {json_path}")
        return [], [], {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        stretch_ids = []
        config = []
        manual_results = {}

        for item in config_data:
            if 'stretch_ids' in item:
                stretch_ids = item.get('stretch_ids', [])
            elif 'pointID' in item:
                config.append(item)
            elif 'manual_results' in item:
                manual_results = item.get('manual_results', {})

        return config, stretch_ids, manual_results
    except Exception as e:
        logger.error(f"加载JSON配置时出错: {str(e)}")
        return [], [], {}


def stretch_to_4_3(image: np.ndarray) -> np.ndarray:
    """根据图片方向拉伸图片"""
    height, width = image.shape[:2]

    if height < width:
        target_ratio = 4 / 3
    else:
        target_ratio = 3 / 4

    current_ratio = width / height

    if current_ratio > target_ratio:
        new_height = height
        new_width = int(new_height * target_ratio)
    else:
        new_width = width
        new_height = int(new_width / target_ratio)

    stretched_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return stretched_image


def find_meter_config(json_config: List[Dict], pointID: str) -> Optional[Dict]:
    for config in json_config:
        if config.get('pointID') == pointID:
            return config
    return None


def crop_image(img: np.ndarray, crop: Dict) -> np.ndarray:
    xmin = int(crop['xmin'])
    ymin = int(crop['ymin'])
    xmax = int(crop['xmax'])
    ymax = int(crop['ymax'])

    height, width = img.shape[:2]
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(xmin + 1, min(xmax, width))
    ymax = max(ymin + 1, min(ymax, height))

    return img[ymin:ymax, xmin:xmax]


def bj_model_init(CONFIG):
    """初始化所有模型"""
    zhizhen_predictor = None
    youwei_predictor = None
    shuxian_predictor = None
    dial_detection_model = None

    try:
        zhizhen_predictor = zhizhen_model.model_init(CONFIG['main_model_file'])
        logger.info("✓ 指针检测模型加载成功")

        youwei_predictor = youwei_model.LiquidGaugeProcessor(CONFIG['liquid_seg_model'], CONFIG)
        logger.info("✓ 液位表计处理器加载成功")

        # 数显模型初始化 - 添加异常处理
        try:
            shuxian_predictor = shuxian_model.model_init()
            logger.info("✓ 数显表计模型加载成功")
        except Exception as e:
            logger.warning(f"⚠️ 数显表计模型加载失败（不影响指针表计）: {str(e)}")
            shuxian_predictor = None

        dial_model_path = CONFIG.get('dial_detection_model')

        if not os.path.exists(dial_model_path):
            raise FileNotFoundError(f"模型文件不存在: {dial_model_path}")

        try:
            with torch.serialization.safe_globals(['*']):
                dial_detection_model = YOLO(dial_model_path)
        except:
            dial_detection_model = YOLO(dial_model_path)

        logger.info(f"✓ 表盘检测模型加载成功")
        logger.info("所有模型加载成功！")

        return zhizhen_predictor, youwei_predictor, shuxian_predictor, dial_detection_model

    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return zhizhen_predictor, youwei_predictor, shuxian_predictor, dial_detection_model


def bj_model_inference(zhizhen_predictor, youwei_predictor, shuxian_predictor, dial_detection_model,
                       image_path, object_ID, json_config, stretch_ids, CONFIG, crop_info=None):
    """执行表计推理"""

    reading_result = []
    bbox_result = []
    unit_result = []
    detection_failed = False

    # 读取原图
    original_img = cv2.imread(image_path)
    if original_img is None:
        logger.error(f"无法读取图片: {image_path}")
        return reading_result, unit_result, bbox_result, detection_failed

    pointID = object_ID

    # 判断是否需要拉伸图片
    if pointID in stretch_ids:
        original_img = stretch_to_4_3(original_img)
        logger.info(f"图片已拉伸: {original_img.shape[1]}x{original_img.shape[0]}")

    # 查找仪表配置
    meter_config = find_meter_config(json_config, pointID)
    if not meter_config:
        logger.warning(f"未找到pointID为 {pointID} 的仪表配置")
        return reading_result, unit_result, bbox_result, detection_failed

    # 获取仪表类型和单位信息
    bj_type = meter_config.get('bj_type', 1)

    # 获取单位信息
    unit = ""
    if 'bp_info' in meter_config and meter_config['bp_info']:
        bp_info = meter_config['bp_info'][0]
        unit = bp_info.get('unit', '')
    elif meter_config.get('bj_type') == 5:
        unit = "%"

    # 使用传入的crop_info
    if crop_info is None:
        logger.warning("未提供有效的表盘检测结果")
        detection_failed = True
        return reading_result, unit_result, bbox_result, detection_failed

    # 裁剪图片
    cropped_img = crop_image(original_img, crop_info)
    logger.info(f"裁剪后图片尺寸: {cropped_img.shape[1]}x{cropped_img.shape[0]}")

    # 存储边界框信息
    bbox_result.append([
        crop_info['xmin'],
        crop_info['ymin'],
        crop_info['xmax'],
        crop_info['ymax']
    ])

    # 根据仪表类型处理
    logger.info(f"仪表类型: {bj_type}")

    # 液位表计处理
    if bj_type == 5:
        try:
            reading = youwei_predictor.process_image(cropped_img)
            if reading is not None:
                reading_result.append(float(reading))
                unit_result.append(unit)
                logger.info(f"液位表计读数: {reading} {unit}")
        except Exception as e:
            logger.error(f"液位表计处理失败: {str(e)}")
            reading_result.append("分析失败：识别目标模糊/偏移/缺失")
            unit_result.append(unit)

    # 数显表计处理
    elif bj_type == 6:
        if shuxian_predictor is None:
            logger.error("数显表计模型未加载")
            reading_result.append("分析失败：数显模型未加载")
            unit_result.append(unit)
        else:
            try:
                reading = shuxian_model.model_inference(shuxian_predictor, cropped_img)
                if reading is not None:
                    reading_result.append(float(reading))
                    unit_result.append(unit)
                    logger.info(f"数显表计读数: {reading} {unit}")
            except Exception as e:
                logger.error(f"数显表计处理失败: {str(e)}")
                reading_result.append("分析失败：识别目标模糊/偏移/缺失")
                unit_result.append(unit)

    # 指针式仪表处理
    else:
        try:
            print("\n" + "🔍" * 30)
            print("开始处理指针式仪表")
            print("🔍" * 30)

            # 步骤1：提取关键点信息
            print(f"\n📍 提取关键点...")
            key_points = extract_key_points_from_config(meter_config)

            # 步骤2：坐标变换（从原图坐标转到裁剪图坐标）
            if crop_info and key_points:
                print(f"\n🔄 坐标变换（原图 → 裁剪图）...")
                print(f"   裁剪偏移: xmin={crop_info['xmin']}, ymin={crop_info['ymin']}")
                key_points = apply_crop_transform(key_points, crop_info)

                if 'o' in key_points:
                    print(f"   变换后表盘中心: cx={key_points['o']['cx']:.2f}, cy={key_points['o']['cy']:.2f}")

            # 步骤3：指针检测
            print(f"\n🎯 开始指针检测...")
            conf_thresh = CONFIG['conf_thresh']
            pointers, scores = zhizhen_model.model_inference(zhizhen_predictor, cropped_img, conf_thresh, CONFIG)

            print(f"   检测结果: {len(pointers) if pointers else 0} 个指针")
            if pointers:
                for i, (pointer, score) in enumerate(zip(pointers, scores)):
                    cx, cy, w, h, angle = pointer
                    print(f"   指针{i + 1}: 置信度={score:.4f}, cx={cx:.1f}, cy={cy:.1f}")

            # 步骤4：后处理计算读数
            print(f"\n📊 计算读数...")
            display_reading = zhizhen_postprocess_debug(pointers, scores, key_points, CONFIG, meter_config)

            print(f"\n✅ 最终结果: {display_reading}")
            print("🔍" * 30 + "\n")

            if display_reading is not None:
                if isinstance(display_reading, str) and display_reading.startswith("分析失败"):
                    reading_result.append(display_reading)
                else:
                    reading_result.append(float(display_reading))
                unit_result.append(unit)
                logger.info(f"指针表计结果: {display_reading}")
        except Exception as e:
            logger.error(f"指针表计处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            reading_result.append("分析失败：表盘模糊未找到指针")
            unit_result.append(unit)

    return reading_result, unit_result, bbox_result, detection_failed


def save_image(img, reading_result, unit_result, bbox_result, detection_failed=False):
    """在图像上绘制检测框和读数结果"""
    if img is None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img

    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("SimHei.ttf", 35)
    except:
        font = ImageFont.load_default()

    if detection_failed:
        warning_text = "识别目标偏移/缺失/模糊"
        bbox = draw.textbbox((0, 0), warning_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        bg_x1, bg_y1 = 10, 10
        bg_x2 = bg_x1 + text_width + 20
        bg_y2 = bg_y1 + text_height + 10

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0))
        draw.text(xy=(bg_x1 + 10, bg_y1 + 5), text=warning_text, font=font, fill=(255, 0, 0))

        img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_result

    boxes = bbox_result if bbox_result is not None else []

    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 4:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=3)

            if i < len(reading_result) and reading_result[i] is not None:
                if isinstance(reading_result[i], str) and reading_result[i].startswith("分析失败"):
                    text = reading_result[i]
                else:
                    unit = unit_result[i] if i < len(unit_result) and unit_result[i] else ""
                    if unit:
                        text = f"读数：{reading_result[i]} {unit}"
                    else:
                        text = f"读数：{reading_result[i]}"
            else:
                text = "分析失败"

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            bg_x1 = float(box[0])
            bg_y1 = float(box[1]) - text_height - 10
            bg_x2 = bg_x1 + text_width + 10
            bg_y2 = float(box[1]) - 5

            if bg_y1 < 0:
                bg_y1 = float(box[1]) + 5
                bg_y2 = bg_y1 + text_height + 10

            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(img_pil.width, bg_x2)
            bg_y2 = min(img_pil.height, bg_y2)

            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0))

            text_x = bg_x1 + 5
            text_y = bg_y1
            draw.text(xy=(text_x, text_y), text=text, font=font, fill=(255, 0, 0))

    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_result