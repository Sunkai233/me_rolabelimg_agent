#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import math
from ultralytics import YOLO

def get_rotated_rect_vertices(cx: float, cy: float, w: float, h: float, angle: float) -> List[Tuple[float, float]]:
    """计算旋转矩形的四个顶点"""
    # 计算旋转角度（弧度）
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # 计算未旋转前的矩形半宽半高
    half_w = w / 2
    half_h = h / 2
    
    # 计算四个顶点的相对坐标（未旋转）
    dx = [-half_w, half_w, half_w, -half_w]
    dy = [-half_h, -half_h, half_h, half_h]
    
    # 旋转并平移顶点
    vertices = []
    for i in range(4):
        # 旋转后的坐标
        x_rot = dx[i] * cos_a - dy[i] * sin_a
        y_rot = dx[i] * sin_a + dy[i] * cos_a
        
        # 平移后的坐标
        x = cx + x_rot
        y = cy + y_rot
        vertices.append((x, y))
    
    return vertices

def determine_direction_point(center: Tuple[float, float], 
                             vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    确定方向点（距离表盘中心最远的窄边中点）
    
    参数:
        center: 表盘中心坐标 (cx, cy)
        vertices: 框的四个顶点 [左上, 右上, 右下, 左下]
        
    返回:
        方向点坐标 (x, y)
    """
    cx_o, cy_o = center
    
    # 计算四个边的中点
    mid_points = []
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        mid_points.append((mid_x, mid_y))
    
    # 计算各边长度，找出窄边（较短的两条边）
    edge_lengths = []
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        edge_lengths.append(length)
    
    # 对边长度排序，找出最短的两条边（窄边）
    sorted_indices = sorted(range(len(edge_lengths)), key=lambda i: edge_lengths[i])
    narrow_edge_indices = sorted_indices[:2]  # 两个最短边的索引
    
    # 获取窄边的中点
    narrow_mid_points = [mid_points[i] for i in narrow_edge_indices]
    
    # 计算窄边中点到表盘中心的距离，找出距离最远的点
    max_distance = -1
    direction_point = narrow_mid_points[0]  # 默认取第一个点
    
    for point in narrow_mid_points:
        px, py = point
        distance = math.sqrt((px - cx_o)**2 + (py - cy_o)**2)
        if distance > max_distance:
            max_distance = distance
            direction_point = point
    
    return direction_point


# 角度归一化
def normalize_angle(angle: float) -> float:
    """将角度归一化到[0, 2π)范围"""
    angle_normalize = angle % (2 * math.pi)

    return angle_normalize


def calculate_angle(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """计算两点之间的角度（弧度）"""
    dx = cx2 - cx1
    dy = cy1 - cy2  # 图像坐标系y轴向下，转换为数学坐标系
    angle = math.atan2(dy, dx)

    if angle >= 0: 
       angle = angle
    else:
       angle = angle + 2 * math.pi

    return angle 


def calculate_nonlinear_reading(pointer_angle: float, processed_measures: List[Dict]) -> float:
    """
    计算非线性表计的读数，基于指针角度和多个刻度点进行插值计算
    
    参数:
        pointer_angle: 指针角度（度）
        processed_measures: 预处理过的刻度点信息列表，每个元素包含:
            'name': 刻度点名称 (如 'start_0', 'add1_1.5')
            'angle_deg': 刻度点角度（度）
            'measure': 刻度点对应的值
    
    返回:
        计算得到的仪表读数
    """
    try:
        # 1. 排序刻度点（按角度）
        sorted_measures = sorted(processed_measures, key=lambda x: x['angle_deg'])
        
        # 2. 检查指针角度是否在刻度范围内
        min_angle = sorted_measures[0]['angle_deg']
        max_angle = sorted_measures[-1]['angle_deg']
        
        if pointer_angle < min_angle:
            return sorted_measures[0]['measure']
        
        if pointer_angle > max_angle:
            return sorted_measures[-1]['measure']
        
        # 3. 找到指针所在的区间（两个相邻刻度点）
        lower_idx = None
        upper_idx = None
        
        for i in range(len(sorted_measures) - 1):
            if sorted_measures[i]['angle_deg'] <= pointer_angle <= sorted_measures[i+1]['angle_deg']:
                lower_idx = i
                upper_idx = i + 1
                break
        
        if lower_idx is None:
            # 处理角度刚好等于某个刻度点的情况
            for i in range(len(sorted_measures)):
                if abs(pointer_angle - sorted_measures[i]['angle_deg']) < 1e-5:
                    return sorted_measures[i]['measure']
            
            # 如果仍然未找到，返回最近刻度点
            closest_idx = min(range(len(sorted_measures)), 
                             key=lambda i: abs(pointer_angle - sorted_measures[i]['angle_deg']))
            return sorted_measures[closest_idx]['measure']
        
        # 4. 获取区间信息
        lower_measure = sorted_measures[lower_idx]
        upper_measure = sorted_measures[upper_idx]
        
        
        # 5. 计算角度比例
        angle_range = upper_measure['angle_deg'] - lower_measure['angle_deg']
        if abs(angle_range) < 1e-5:
            return (lower_measure['measure'] + upper_measure['measure']) / 2
        
        angle_offset = pointer_angle - lower_measure['angle_deg']
        ratio = angle_offset / angle_range
        
        # 6. 计算读数（线性插值）
        value_range = upper_measure['measure'] - lower_measure['measure']
        reading = lower_measure['measure'] + ratio * value_range
        
        
        return reading
    
    except Exception as e:
        import traceback
        # 返回默认值（刻度点的平均值）
        values = [m['measure'] for m in processed_measures]
        return sum(values) / len(values) if values else 0.0





# 档位表特殊映射函数
def map_gear_reading(reading: float, min_val: float, max_val: float) -> str:
    """
    档位表特殊映射函数
    将1-19的读数映射为1-17档位，其中9分为9A、9B、9C
    映射规则:
        1-8 -> 1-8
        9 -> 9A
        10 -> 9B
        11 -> 9C
        12-19 -> 10-17
    """
    # 确保读数是整数（因为档位表单位是"档"）
    int_reading = int(round(reading))
    
    # 检查是否在有效范围内
    if int_reading < min_val or int_reading > max_val:
        return str(reading)
    
    # 应用特殊映射规则
    if int_reading <= 8:
        return str(int_reading)
    elif int_reading == 9:
        return "9A"
    elif int_reading == 10:
        return "9B"
    elif int_reading == 11:
        return "9C"
    else:  # 12-19
        # 12 -> 10, 13 -> 11, ..., 19 -> 17
        mapped_value = int_reading - 2
        return str(mapped_value)




# 初始化YOLO模型
def model_init(model_file: str) -> YOLO:
    """初始化YOLOv8 OBB模型"""
    predictor = YOLO(model_file) 

    return predictor

# 模型推理
def model_inference(model: YOLO, img: np.ndarray, conf_thresh,  CONFIG) -> Tuple[List, List]:
    """对图片执行推理，并返回所有检测结果"""
    # 执行模型推理
    results = model(img)
    
    # 检查推理结果有效性
    if (len(results) == 0) or  (not hasattr(results[0], 'obb')) or (results[0].obb is None):
        return [], []  # 返回空结果
    
    # 提取旋转边界框(OBB)信息
    obb = results[0].obb
    
    # 提取置信度分数并转换为numpy数组
    conf = obb.conf.cpu().numpy()
    
    # 提取旋转框参数 (cx, cy, w, h, angle) 并转换为numpy数组
    xywhr = obb.xywhr.cpu().numpy()
    
    # 提取类别索引并转换为整数numpy数组
    cls_idx = obb.cls.cpu().numpy().astype(int)
    
    # 获取指针类别ID
    pointer_class_id = None
    for cls_id, name in CONFIG['classes'].items():
        if name == 'pointer':
            pointer_class_id = cls_id
            break
    
    # 处理未找到指针类别的情况
    if pointer_class_id is None:
        pointer_class_id = 0
    
    # 创建指针检测结果的过滤掩码
    pointer_mask = (cls_idx == pointer_class_id) & (conf > conf_thresh)
    
    # 应用掩码获取指针框参数
    pointer_boxes = xywhr[pointer_mask].tolist()
    
    # 应用掩码获取指针置信度
    pointer_scores = conf[pointer_mask].tolist()
    
    return pointer_boxes, pointer_scores

# 计算指针读数
def calculate_pointer_reading(center: Dict, start: Dict, end: Dict, 
                             pointer_box: List[float],
                             min_val: float, max_val: float, 
                             direction: str,
                             meter_config: Dict,
                             key_points: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    计算指针读数（使用新的窄边中点距离判断方法确定方向）
    """
    try:
        # 获取表盘中心坐标
        cx_o, cy_o = center['cx'], center['cy']
        
        # 使用刻度框的有效点（窄边中点）
        cx_s, cy_s = start['effective_point']
        cx_f, cy_f = end['effective_point']
        
        # 提取指针框参数
        cx_p, cy_p, w, h, angle = pointer_box
        
        # 计算指针长度（取w和h中的较大值）
        pointer_length = max(w, h)
        
        # 计算指针框的四个顶点
        pointer_vertices = get_rotated_rect_vertices(cx_p, cy_p, w, h, angle)
        
        # === 修改部分开始：新的指针方向确定方法 ===
        # 计算四个边的中点
        mid_points = []
        for i in range(4):
            x1, y1 = pointer_vertices[i]
            x2, y2 = pointer_vertices[(i + 1) % 4]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            mid_points.append((mid_x, mid_y))
        
        # 计算各边长度，找出窄边（较短的两条边）
        edge_lengths = []
        for i in range(4):
            x1, y1 = pointer_vertices[i]
            x2, y2 = pointer_vertices[(i + 1) % 4]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            edge_lengths.append(length)
        
        # 对边长度排序，找出最短的两条边（窄边）
        sorted_indices = sorted(range(len(edge_lengths)), key=lambda i: edge_lengths[i])
        narrow_edge_indices = sorted_indices[:2]  # 两个最短边的索引
        
        # 获取窄边的中点
        narrow_mid_points = [mid_points[i] for i in narrow_edge_indices]
        
        # 计算两个窄边中点到表盘中心的距离
        distances = []
        for point in narrow_mid_points:
            px, py = point
            distance = math.sqrt((px - cx_o)**2 + (py - cy_o)**2)
            distances.append(distance)
        
        # 找出距离圆心近的点（起点）和远的点（终点）
        if distances[0] <= distances[1]:
            near_point = narrow_mid_points[0]  # 近的点作为起点
            far_point = narrow_mid_points[1]   # 远的点作为终点
        else:
            near_point = narrow_mid_points[1]  # 近的点作为起点
            far_point = narrow_mid_points[0]   # 远的点作为终点
        
        # 从近点向远点做射线，计算射线的角度
        angle_pointer = calculate_angle(near_point[0], near_point[1], 
                                      far_point[0], far_point[1])
        
        # 归一化角度
        angle_pointer = normalize_angle(angle_pointer)
        pointer_angle_deg = math.degrees(angle_pointer)
        

        # === 修改部分结束 ===
        
        # 获取仪表配置信息
        bp_info = meter_config.get('bp_info', [{}])[0]
        pointer_num = bp_info.get('pointer_num', 1)
        unit = bp_info.get('unit', '')
        dial_type = bp_info.get('dialType', 2)  # 默认为非360度表计
        
        # 直接使用key_points中的刻度点信息
        processed_measures = []
        if 'angle_measures' in key_points:
            for measure in key_points['angle_measures']:
                processed_measures.append({
                    'name': measure.get('name', ''),
                    'angle_deg': measure['angle'],  # 使用预先计算的角度
                    'measure': measure['value']
                })

        # 处理360度表计 (dialType=1)
        if dial_type == 1:
            # 计算起始刻度点角度
            angle_start = calculate_angle(cx_o, cy_o, cx_s, cy_s)
            angle_start = normalize_angle(angle_start)
            angle_start_deg = math.degrees(angle_start)
            
            # 根据方向计算正确的偏移角度
            if direction == 'clockwise':
                # 顺时针表盘：计算顺时针旋转的角度
                if angle_pointer < angle_start:
                    pointer_offset = angle_start - angle_pointer
                else:
                    pointer_offset = angle_start + (2 * math.pi - angle_pointer)
            else:  # counter-clockwise
                # 逆时针表盘：计算逆时针旋转的角度
                if angle_pointer > angle_start:
                    pointer_offset = angle_pointer - angle_start
                else:
                    pointer_offset = (2 * math.pi - angle_start) + angle_pointer
            
            pointer_offset_deg = math.degrees(pointer_offset)
            
            # 计算读数
            reading_ratio = pointer_offset / (2 * math.pi)
            reading = min_val + (max_val - min_val) * reading_ratio
            

            
            # 特殊处理：当单位为"次"且最大量程为10时
            if unit == "次" and abs(max_val - 10.0) < 1e-5:
                # 四舍五入取整
                rounded_reading = round(reading)
                
                # 如果取整后为10，则输出0
                if abs(rounded_reading - 10.0) < 1e-5:
                    return 0.0, pointer_length
                else:
                    return rounded_reading, pointer_length
            
            return reading, pointer_length
        
        # 非线性量程处理
        elif len(processed_measures) > 2:
            
            # 调用非线性计算函数
            reading = calculate_nonlinear_reading(
                pointer_angle=pointer_angle_deg,
                processed_measures=processed_measures
            )
            return reading, pointer_length
        
        # 线性量程处理 (非360度表计)
        else:
            # 计算各点相对于表盘中心的角度
            angle_start = calculate_angle(cx_o, cy_o, cx_s, cy_s)
            angle_final = calculate_angle(cx_o, cy_o, cx_f, cy_f)
            
            # 归一化角度
            angle_start = normalize_angle(angle_start)
            angle_final = normalize_angle(angle_final)
            angle_start_deg = math.degrees(angle_start)
            angle_final_deg = math.degrees(angle_final)
            

            
            # 计算角度范围和指针位置
            if direction == 'counter-clockwise':
                angle_range = (angle_final - angle_start) % (2 * math.pi)
                pointer_offset = (angle_pointer - angle_start) % (2 * math.pi)
            else:
                angle_range = (angle_start - angle_final) % (2 * math.pi)
                pointer_offset = (angle_start - angle_pointer) % (2 * math.pi)
            
            angle_range_deg = math.degrees(angle_range)
            pointer_offset_deg = math.degrees(pointer_offset)
            
            # 处理角度范围为零的情况
            if abs(angle_range) < 1e-5:
                return None, pointer_length
            
            # 特殊处理：当指针角度出现在起始刻度的逆时针一侧时
            if direction == 'clockwise' and pointer_offset > angle_range:
                
                # 计算起始刻度和终止刻度的中线
                if angle_start > angle_final:
                    # 处理跨越0度的情况
                    midpoint = (angle_start + angle_final + 2 * math.pi) / 2
                    if midpoint >= 2 * math.pi:
                        midpoint -= 2 * math.pi
                else:
                    midpoint = (angle_start + angle_final) / 2
                
                # 将中线旋转180度得到分界线
                divider = (midpoint + math.pi) % (2 * math.pi)
                
                # 判断指针相对于分界线的位置
                if angle_pointer >= divider:
                    angle_from_divider = angle_pointer - divider
                else:
                    angle_from_divider = (2 * math.pi - divider) + angle_pointer
                
                # 如果指针在分界线的顺时针180度范围内，返回最小值
                if angle_from_divider <= math.pi:
                    return min_val, pointer_length
                else:
                    # 重新计算指针偏移角度（使用正常逻辑）
                    pointer_offset = (angle_start - angle_pointer) % (2 * math.pi)
                    reading_ratio = pointer_offset / angle_range
                    reading = min_val + (max_val - min_val) * reading_ratio
                    return reading, pointer_length
            
            # 正常计算读数
            reading_ratio = pointer_offset / angle_range
            reading = min_val + (max_val - min_val) * reading_ratio
            

            return reading, pointer_length
    
    except Exception as e:
        import traceback
        return None, None

def handle_multiple_pointers(pointers: List, scores: List, 
                            center: Dict, start: Dict, end: Dict,
                            min_val: float, max_val: float, 
                            direction: str,
                            meter_config: Dict,
                            key_points: Dict) -> Optional[float]:
    """
    处理多指针情况
    参数:
        pointers: 检测到的指针列表
        scores: 对应的置信度分数
        center: 表盘中心点信息
        start: 起始刻度点信息
        end: 结束刻度点信息
        min_val: 最小值
        max_val: 最大值
        direction: 方向
        meter_config: 仪表配置
        key_points: 关键点信息
    """
    try:
        # 获取表盘中心点坐标
        cx_o, cy_o = key_points['o']['cx'], key_points['o']['cy']
        
        # 获取仪表配置信息
        bp_info = meter_config.get('bp_info', [{}])[0]
        pointer_num = bp_info.get('pointer_num', 1)
        unit = bp_info.get('unit', '')
        dial_type = bp_info.get('dialType', 2)  # 默认为非360度表计
        bj_type = meter_config.get('bj_type', 1)
        
        # 检查是否为单指针表计 (bj_type-1-*-1-0)
        is_single_pointer_meter = (bj_type == 1 and pointer_num == 1)
        
        # 特殊处理：单指针表计但检测到多个指针
        if is_single_pointer_meter and len(pointers) > 1:
            
            # 计算每个指针的窄边中点距离圆心的距离
            pointer_distances = []
            for i, pointer in enumerate(pointers):
                cx, cy, w, h, angle = pointer
                
                # 计算指针框的四个顶点
                pointer_vertices = get_rotated_rect_vertices(cx, cy, w, h, angle)
                
                # 确定指针方向点（窄边中点）
                direction_point = determine_direction_point((cx_o, cy_o), pointer_vertices)
                dir_x, dir_y = direction_point
                
                # 计算到圆心的距离
                distance = math.sqrt((dir_x - cx_o)**2 + (dir_y - cy_o)**2)
                pointer_distances.append((i, distance))
            
            # 按距离排序，选择距离最远的指针
            pointer_distances.sort(key=lambda x: x[1], reverse=True)
            selected_index = pointer_distances[0][0]
            
            
            # 只保留选中的指针
            pointers = [pointers[selected_index]]
            scores = [scores[selected_index]]
        
        # 检查是否为360度双指针表计 (o-1类型且单位为"次")
        is_360_dual_pointer = (dial_type == 1 and unit == "次" and pointer_num == 2)
        
        # 按置信度排序
        sorted_indices = np.argsort(scores)[::-1]
        readings = []
        pointer_lengths = []  # 存储指针长度
        
        # 计算每个指针的读数和长度
        for i in sorted_indices[:pointer_num]:
            pointer_box = pointers[i]
            reading, pointer_length = calculate_pointer_reading(
                center=key_points['o'],
                start=key_points['start'],
                end=key_points['final'],
                pointer_box=pointer_box,
                min_val=min_val,
                max_val=max_val,
                direction=direction,
                meter_config=meter_config,
                key_points=key_points
            )
            if reading is not None:
                readings.append(reading)
                pointer_lengths.append(pointer_length)
        
        # 根据指针数量确定最终读数
        if not readings:
            return None
        
        # 特殊处理360度双指针表计
        if is_360_dual_pointer:
            
            # 如果只检测到一个指针，假设两个指针重合
            if len(readings) == 1:
                single_reading = readings[0]
                # 四舍五入到整数
                rounded_reading = round(single_reading)
                # 计算最终读数: 读数*10 + 读数
                final_reading = rounded_reading * 10 + rounded_reading
                return final_reading
            
            # 如果检测到多个指针，根据指针长度区分长短指针
            elif len(readings) >= 2:
                # 根据指针长度排序，较长的为长指针
                sorted_by_length = sorted(zip(readings, pointer_lengths), key=lambda x: x[1], reverse=True)
                
                # 取最长的两个指针
                long_pointer_reading, long_pointer_length = sorted_by_length[0]
                short_pointer_reading, short_pointer_length = sorted_by_length[1]
                
                
                # 四舍五入到整数
                long_rounded = round(long_pointer_reading)
                short_rounded = round(short_pointer_reading)
                
                # 计算最终读数: 短指针读数*10 + 长指针读数*1
                final_reading = short_rounded * 10 + long_rounded
                
                return final_reading
        
        # 非360度双指针表计的正常处理
        if pointer_num == 1:
            return readings[0]
        elif pointer_num == 2:
            return min(readings)
        elif pointer_num >= 3:
            sorted_readings = sorted(readings)
            return sorted_readings[len(sorted_readings) // 2]
        else:
            return readings[0]
    
    except Exception as e:
        return None






def zhizhen_postprocess(pointers, scores, key_points, CONFIG, meter_config):

    display_reading = None

    # 检查是否所有关键点都已获取
    required_keys = ['o', 'start', 'final']
    missing_keys = [k for k in required_keys if k not in key_points]
            
    if missing_keys:
       return "分析失败：表盘模糊未找到指针"  # 修改：返回具体的失败原因
    
    elif pointers:
       # 从配置中获取最小值和最大值
       min_val = key_points.get('min_value', CONFIG['min_value'])
       max_val = key_points.get('max_value', CONFIG['max_value'])
                
                
       # 处理多指针
       reading = handle_multiple_pointers(
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

                
       if reading is not None:

          # 获取单位
          bp_info = meter_config.get('bp_info', [{}])[0]
          unit = bp_info.get('unit', '')
          dial_type = bp_info.get('dialType', 2)
          pointer_num = bp_info.get('pointer_num', 1)

          # 检查是否为360度双指针表计
          is_360_dual_pointer = (dial_type == 1 and unit == "次" and pointer_num == 2)
          
          # 对于360度双指针表计，已经处理了四舍五入，直接使用结果
          if is_360_dual_pointer:
                display_reading = str(reading)
          # 对于其他表计，根据单位处理读数精度
          else:
                if unit in ['档', '次']:
                    reading = round(reading)
                else:
                    reading = round(reading, 2)
                display_reading = str(reading)
                                       
          return display_reading
       else:
          return "分析失败：表盘模糊未找到指针"  # 修改：返回具体的失败原因
    else:
        return "分析失败：表盘模糊未找到指针"  # 修改：返回具体的失败原因
