# 目标检测系统 (General Object Detector)

一个基于 VLM (Vision Language Model) 的通用目标检测系统，能够自动识别图像中的目标类型、精确定位目标位置并生成详细的分析报告。

## 功能特性

- **智能目标检测**：自动识别并定位多种目标类型
- **精确定位**：为每个目标提供百分比范围和像素坐标的精确定位
- **详细报告**：生成包含标注图像和目标分析的结构化报告
- **批处理能力**：支持处理目录中的多个图像
- **MPO 格式支持**：自动将 MPO 格式图像转换为 JPEG 格式
- **中文输出**：所有分析结果和报告均以中文呈现

## 技术栈

- Python 3.9+
- VLM API (Qwen/Qwen3-VL-235B-A22B-Thinking)
- PIL (图像处理)
- scikit-learn (TF-IDF 嵌入)
- requests (API 调用)

## 快速开始

### 环境要求

- Python 3.9+
- pip 21.0+

### 安装步骤

1. **克隆仓库**

```bash
git clone <仓库地址>
cd VLM_identification
```

2. **创建虚拟环境**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置系统**

编辑 `config.py` 文件，填写相关配置：

```python
# API Configuration (必填)
API_KEY = "your_siliconflow_api_key_here"  # 请填写您的SiliconFlow API密钥
BASE_URL = "https://api.siliconflow.cn"  # SiliconFlow API基础URL

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Thinking"
MAX_TOKENS = 3500

# Supported Object Types (可自定义)
SUPPORTED_OBJECT_TYPES = [
    "object1",
    "object2",
    "object3",
    "object4",
    "object5"
]

# Default Directories
DEFAULT_INPUT_DIR = "input_images"  # 输入图像目录
DEFAULT_OUTPUT_DIR = "output_reports"  # 输出报告目录

# Knowledge Base Configuration (可选)
KNOWLEDGE_BASE_PATH = "knowledge_base.txt"  # 知识库文件路径
```

## 使用方法

### 1. 批处理模式

**处理目录中的所有图像**：

```bash
python main.py --dir <输入目录> --output <输出目录>
```

**处理指定数量的随机图像**：

```bash
python main.py --dir <输入目录> --count 10 --output <输出目录>
```

### 2. 单文件模式

**处理单个图像**：

```bash
python main.py --file <图像路径> --output <输出目录>
```

### 3. 自定义输出格式

**生成 Markdown 格式报告**（默认）：

```bash
python main.py --dir <输入目录> --format md
```

**生成文本格式报告**：

```bash
python main.py --dir <输入目录> --format txt
```

## 项目结构

```
VLM_identification/
├── output_reports/           # 分析报告输出目录
│   ├── *.jpg                 # 标注后的图像
│   └── *.md/txt              # 分析报告
├── config.py                 # 配置文件
├── main.py                   # 主脚本
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明
└── .gitignore                # Git 忽略文件
```

## 输出示例

### 标注图像

![标注示例](output_reports/example_annotated.jpg)

### 分析报告

```
# 目标检测报告

生成时间: 2026-02-09 10:30:00
分析图像: sample_image.jpg

## 标注后图像

![标注后图像](sample_image_annotated.jpg)

## 检测结果

object1, X:10%-25%, Y:30%-45%, [100, 200, 300, 400]
object2, X:50%-75%, Y:40%-60%, [400, 300, 600, 500]

## 检测说明

本报告检测以下目标类型：
- object1
- object2
- object3
- object4
- object5
```

## 配置说明

`config.py` 文件包含以下主要配置项：

| 配置项 | 描述 | 默认值 | 必填 |
|-------|------|--------|------|
| API_KEY | SiliconFlow API 密钥 | "your_siliconflow_api_key_here" | ✅ |
| BASE_URL | SiliconFlow API 基础 URL | https://api.siliconflow.cn | ✅ |
| MODEL_NAME | 使用的模型名称 | Qwen/Qwen3-VL-235B-A22B-Thinking | ❌ |
| MAX_TOKENS | 模型最大输出 tokens | 3500 | ❌ |
| SUPPORTED_OBJECT_TYPES | 支持的目标类型列表 | ["object1", "object2", "object3", "object4", "object5"] | ❌ |
| DEFAULT_INPUT_DIR | 默认输入目录 | input_images | ❌ |
| DEFAULT_OUTPUT_DIR | 默认输出目录 | output_reports | ❌ |
| KNOWLEDGE_BASE_PATH | 知识库文件路径 | knowledge_base.txt | ❌ |


## 注意事项

1. **API 配置**：必须在 `config.py` 文件中填写有效的 SiliconFlow API 密钥
2. **知识库配置**：
   - 知识库文件是可选的，如果不存在，系统会正常运行但不会使用知识库信息
   - 如果需要使用知识库，请创建 `knowledge_base.txt` 文件并填充相关知识
   - 知识库内容应包含与目标检测相关的信息，以提高检测准确性
3. **图像大小**：过大的图像可能会导致 API 调用失败，建议使用适中大小的图像
4. **网络连接**：需要稳定的网络连接才能调用 SiliconFlow API
5. **处理时间**：分析图像可能需要几秒钟到几分钟的时间，具体取决于图像复杂度
6. **目标类型**：可以在 `config.py` 中自定义需要检测的目标类型

## 故障排除

### 常见问题

1. **API Error: 401 Unauthorized**
   - 解决方案：检查 `config.py` 文件中的 API_KEY 是否正确填写

2. **API Error: 403 Forbidden**
   - 解决方案：检查您的 SiliconFlow API 密钥是否有足够的权限

3. **ModuleNotFoundError**
   - 解决方案：确保已激活虚拟环境并安装了所有依赖

4. **Unsupported image format: mpo**
   - 解决方案：系统会自动将 MPO 格式转换为 JPEG 格式，无需手动处理

5. **Error loading knowledge base**
   - 解决方案：知识库文件是可选的，如果不存在，系统会正常运行但不会使用知识库信息

6. **编码错误**
   - 解决方案：确保终端使用 UTF-8 编码

## 许可证

本项目采用 MIT 许可证

## 更新日志

### v1.0.0 (2026-02-09)

- 初始版本发布
- 实现通用目标检测和定位功能
- 支持生成详细的分析报告
- 支持批处理模式
- 提供模板化的配置系统

## 致谢

- Qwen 团队开发的视觉语言模型

