#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
表计预测模块（集成用户的预测代码）
"""

import os
import sys
import json
import tempfile

# 添加当前目录到Python路径（假设用户的预测模块在同一目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class MeterPredictor:
    """表计预测器"""

    def __init__(self):
        self.models_loaded = False
        self.zhizhen_predictor = None
        self.youwei_predictor = None
        self.shuxian_predictor = None
        self.dial_detection_model = None

    def initialize_models(self, config):
        """
        初始化模型

        Args:
            config: 模型配置字典
        """
        try:
            # 导入预测模块
            import bj_inference_main

            # 构建CONFIG
            CONFIG = {
                'main_model_file': config['zhizhen_file'],
                'liquid_seg_model': config['youwei_file'],
                'dial_detection_model': config['dial_detection_model'],
                'conf_thresh': config['conf_thresh'],
                'min_value': 1,
                'max_value': 10.0,
                'direction': 'clockwise',
                'classes': {
                    0: 'pointer',
                    1: 'start',
                    2: 'final',
                    3: 'o'
                },
                'meter_types': {
                    1: '指针式仪表',
                    5: '液位刻度表计',
                    6: '数显类表计'
                },
                'liquid_classes': {
                    'filled_area': 0,
                    'empty_area': 1
                }
            }

            # 初始化模型
            (self.zhizhen_predictor,
             self.youwei_predictor,
             self.shuxian_predictor,
             self.dial_detection_model) = bj_inference_main.bj_model_init(CONFIG)

            self.CONFIG = CONFIG
            self.models_loaded = True

            return True, "模型加载成功"

        except Exception as e:
            return False, f"模型加载失败：{str(e)}"

    def predict(self, image_path, json_data):
        """
        执行预测

        Args:
            image_path: 图片路径
            json_data: JSON配置数据（从XML转换来的）

        Returns:
            tuple: (success, result_image_path, message)
        """
        if not self.models_loaded:
            return False, None, "模型未加载"

        try:
            import bj_inference_main
            import cv2

            # 创建临时JSON文件
            temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
            json.dump([json_data], temp_json, ensure_ascii=False, indent=4)
            temp_json.close()

            # 读取图片
            original_img = cv2.imread(image_path)
            if original_img is None:
                return False, None, f"无法读取图片：{image_path}"

            # 获取pointID
            pointID = json_data.get('pointID', '001')

            # 创建JSON配置列表
            json_config = [json_data]
            stretch_ids = []

            # 步骤1：表盘检测
            crop_info = None
            if json_data.get('crop'):
                crop_info = json_data['crop'][0] if isinstance(json_data['crop'], list) else json_data['crop']
            else:
                # 使用表盘检测模型
                bj_type = json_data.get('bj_type', 1)
                target_class_name = bj_inference_main.get_target_class_name(bj_type)

                crop_info = bj_inference_main.detect_dial_boundary(
                    self.dial_detection_model,
                    original_img,
                    conf_thresh=self.CONFIG['conf_thresh'],
                    target_class_name=target_class_name,
                    json_crop_info=None
                )

            if crop_info is None:
                return False, None, "表盘检测失败"

            # 步骤2：表计推理
            reading_result, unit_result, bbox_result, detection_failed = bj_inference_main.bj_model_inference(
                self.zhizhen_predictor,
                self.youwei_predictor,
                self.shuxian_predictor,
                self.dial_detection_model,
                image_path,
                pointID,
                json_config,
                stretch_ids,
                self.CONFIG,
                crop_info=crop_info
            )

            # 步骤3：生成结果图片
            result_img = bj_inference_main.save_image(
                original_img,
                reading_result,
                unit_result,
                bbox_result,
                detection_failed
            )

            # 保存结果图片到临时文件
            result_path = tempfile.NamedTemporaryFile(suffix='_result.jpg', delete=False).name
            cv2.imwrite(result_path, result_img)

            # 清理临时JSON文件
            try:
                os.unlink(temp_json.name)
            except:
                pass

            # 构建结果消息
            if detection_failed:
                message = "检测失败：识别目标偏移/缺失/模糊"
            elif reading_result:
                if isinstance(reading_result[0], str) and reading_result[0].startswith("分析失败"):
                    message = reading_result[0]
                else:
                    unit = unit_result[0] if unit_result else ""
                    message = f"读数：{reading_result[0]} {unit}"
            else:
                message = "预测完成，但无读数结果"

            return True, result_path, message

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, None, f"预测失败：{str(e)}"


# 全局预测器实例
_predictor = None


def get_predictor():
    """获取全局预测器实例"""
    global _predictor
    if _predictor is None:
        _predictor = MeterPredictor()
    return _predictor