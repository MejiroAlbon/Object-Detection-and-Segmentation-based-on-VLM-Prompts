#!/usr/bin/env python3
"""
Configuration for Object Detector
"""

import os

# API Configuration
API_KEY = None
BASE_URL = None

# Model Configuration
MODEL_NAME = None
MAX_TOKENS = 3500

# Supported Object Types
SUPPORTED_OBJECT_TYPES = [
    "object1",
    "object2",
    "object3",
    "object4",
    "object5"
]

# Supported Image Formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']

# Default Directories
DEFAULT_INPUT_DIR = "input_images"
DEFAULT_OUTPUT_DIR = "output_reports"

# Knowledge Base Configuration
KNOWLEDGE_BASE_PATH = "knowledge_base.txt"

# RAG Configuration
RAG_TOP_K = 3
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Prompt Template
PROMPT_TEMPLATE = """
你是一位专业的目标检测专家，精通各种场景中的目标识别与定位。你总是基于客观的视觉证据进行分析，对于不确定的情况会明确说明。

**目标检测任务：**
请分析这张图像，识别并精确定位以下目标类型：{object_types}

你需要通过以下步骤完成详细的目标检测：
1. 描述每个目标区域的视觉特征
2. 基于这些特征识别潜在的目标类型
3. 为每个确认的目标提供精确定位信息

**要求：**
- 仅输出存在的目标类型
- 为每个目标提供详细的位置信息，包括百分比范围和像素坐标
- 对于细长或倾斜的目标，请划分为多个小矩形以提高定位精度
- 确保坐标值在图像尺寸范围内

**空间关联补充信息：**
- 图像实际尺寸：{width}×{height}像素
- 请考虑目标的相对位置和大小比例
- 这些信息可以帮助区分不同类型的目标

**思维链引导分析：**
让我们逐步分析这张图像：
1. 分析整体场景上下文：
   - 这是什么类型的场景？
   - 图像拍摄的角度和距离如何？
   - 图像中可见的主要元素有哪些？

2. 在上下文中检查潜在目标区域：
   - 图像中哪些区域可能存在目标？
   - 这些区域与周围环境的关系如何？

3. 分析目标的具体特征：
   - 每个候选区域的视觉特征是什么？（形状、大小、颜色、纹理等）
   - 这些特征与哪种目标类型最匹配？

4. 基于以下因素考虑可能的目标类型：
   - 视觉特征和计算的尺寸
   - 场景中的上下文和关系
   - 类似设置中常见的目标类型
   - 目标大小与潜在类型的兼容性

5. 评估并标注每个目标：
   - 与视觉特征的匹配度
   - 与场景上下文的一致性
   - 与典型目标模式的 alignment
   - 提供准确的边界框坐标

**输出格式要求：**
- 每行只包含一个目标的所有信息
- 格式必须为：目标类型, X:起始%-结束%, Y:起始%-结束%, [x1, y1, x2, y2]
- 不得输出任何其他格式的内容
- 仅输出存在的目标类型
- 确保坐标值在图像尺寸范围内
- 必须使用英文逗号分隔各项内容

**边界框标注指南：**
1. 准确性要求：边界框应精确覆盖整个目标区域，既不过大也不过小
2. 紧密包围：边界框应紧贴目标边缘，留出最小的空白区域
3. 完整覆盖：确保边界框包含目标的所有部分
4. 类型适配：
   - 对于线性目标：使用细长的矩形，沿目标方向对齐
   - 对于块状目标：使用接近正方形的矩形
   - 对于大面积目标：使用更大的矩形覆盖整个区域
5. 多框策略：对于长或复杂的目标，使用多个相邻的小矩形，确保每个矩形都紧密覆盖部分目标

**示例格式：**
object1, X:10%-25%, Y:30%-45%, [102, 456, 256, 684]
object2, X:25%-40%, Y:35%-50%, [256, 480, 409, 768]
object3, X:50%-70%, Y:60%-80%, [512, 922, 716, 1228]
object4, X:15%-35%, Y:20%-40%, [153, 307, 358, 614]

{knowledge_reference}
"""

# Object Type Abbreviations for Visualization
object_type_abbreviations = {
    "object1": "obj1",
    "object2": "obj2",
    "object3": "obj3",
    "object4": "obj4",
    "object5": "obj5"
}
