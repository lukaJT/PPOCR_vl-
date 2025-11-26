# PaddleOCR-VL 部署




## 项目概述

本文档详细记录 PaddleOCR-VL 的本地虚拟环境部署流程，核心逻辑为 **虚拟环境隔离 → 系统依赖打底 → 核心依赖版本配对安装 → 脚本适配 → 验证运行**，请注意 “3.0专属版本兼容” 和 “依赖顺序” 

## 一、前置准备）

### 服务器环境要求



| 类别        | 具体要求                               |
| --------- | ---------------------------------- |
| 硬件        | NVIDIA GPU（计算能力≥7.0，支持 CUDA）       |
| 系统        | Ubuntu 20.04/22.04（其他 Linux 需适配命令） |
| Python 版本 | 3.8\~3.11（以 3.10 为例）             |
| CUDA 版本   | 11.2\~12.6（以 12.4 为例）            |



## 二、部署流程

### 步骤 1：创建并激活本地虚拟环境（依赖隔离）



```
1. 安装虚拟环境工具

sudo apt-get install -y python3-venv

2. 创建虚拟环境（路径自定义）

python3 -m venv /back/aiengine/paddleocr/paddleocr\_gpu\_new

3. 激活

source /back/aiengine/paddleocr/paddleocr\_gpu\_new/bin/activate
```

#### 验证：

终端前缀显示 `(paddleocr_gpu_new)` 即为成功。



### 步骤 2：安装系统编译依赖（解决 gcc 编译失败）



```
1. 更新系统源

sudo apt-get update

2. 安装编译工具+Python开发文件

sudo apt-get install -y build-essential python3.10-dev
```

#### 说明：



* `build-essential`：包含 gcc、g++、make，解决 “x86\_64-linux-gnu-gcc failed” 错误。

* `python3.10-dev`：对应 Python 3.10 的头文件，避免 “找不到 Python.h” 编译错误（其他 Python 版本替换为`python3.x-dev`）。

### 步骤 3：安装核心 Python 依赖（强版本配对！）

#### 核心原则：先装底层依赖 → 再装核心框架 → 最后装应用依赖

##### 3.1 升级 pip



```
python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 3.2 安装 numpy（底层依赖，必须 1.x 系列）



```
适配Python3.10、cv2、PaddlePaddle，避开2.x系列ABI冲突

python3 -m pip install numpy==1.26.4 --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 3.3 安装 PaddlePaddle-GPU（核心框架）



```
适配CUDA12.4，配对PaddleOCR3.0

python3 -m pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

#### 验证：



```
python3 -c "import paddle; paddle.utils.run_check()"
```

输出 `PaddlePaddle works well on x GPU` 即为成功。

##### 3.4 安装 PaddleOCR（内置 VL 功能）



```
[doc-parser] 额外依赖包含文档解析和VL模型支持
python3 -m pip install -U "paddleocr[doc-parser]" \
  --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

```

##### 3.5 安装指定版本 safetensors



```
Linux x86_64 专用版本
python3 -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

##### 3.5 安装辅助依赖



```
\# 图像处理+工具类，版本适配numpy1.26.4

python3 -m pip install opencv-python==4.8.0.76 pillow requests --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 4：脚本适配（解决 PaddleOCRVL 导入失败）

最新版 PaddleOCR 使用全新独立类 PaddleOCRVL，旧版 enable_vl=True 参数已废弃。

#### 可用脚本（`PaddleOCRVL.py`）



```
from paddleocr import PaddleOCRVL

import os

\# 初始化VL功能

ocr = PaddleOCRVL(

&#x20;   device="gpu:0",        # 启用GPU加速

&#x20;   enable\_hpi=False,      # 高性能推理（如开启需另下插件，本身已经很快了）

&#x20;   precision="fp16",           # 半精度加速（这里只是为了快速出个结果）

&#x20;   use\_layout\_detection=True,          # 启用版面分析

&#x20;    use\_doc\_orientation\_classify=True,  # 文档方向分类

&#x20;    use\_doc\_unwarping=True,            # 文档去畸变

)

\# 测试图片

img\_path = "/your/path/to/input/(/back/aiengine/paddleocr/pic1.jpg)"

\# check文件

if not os.path.exists(img\_path):

&#x20;   print(f"错误：图片不存在 → {img\_path}")

else:

&#x20;   try:

&#x20;       output = ocr.predict(img\_path)   # 执行识别（3.x使用predict方法）

&#x20;       for res in output:

&#x20;           res.print()                 #打印结构化结果

&#x20;       ressavr\_to\_json(save_path="/back/aiengine/paddleocr/output")    #save json文档（内含检测框和内容）

&#x20;       res.save\_to_markdown(
            
&#x20;       save\_path="/back/aiengine/paddleocr/output",

&#x20;       pretty=True #美化

&#x20;       )

&#x20;       res.save\_to\_img(save_path="/your/path/to/output(/back/aiengine/paddleocr/output)")

&#x20;       print("处理完成！结果已保存到/your/path/to/output(/back/aiengine/paddleocr/output)")

&#x20;       except Exception as e:

&#x20;       print(f"发生错误:{str(e)}")

```

#### 注意：



* 脚本需放在 `/back/aiengine/paddleocr` 目录（与虚拟环境同级）。

* 测试图片 `pic1.jpg` 需上传到同一目录，或修改`img_path`为绝对路径。

### 步骤 5：验证部署（最终运行）



```
\# 确保虚拟环境已激活，进入脚本目录

cd /back/aiengine/paddleocr

\# 运行脚本

python3 PaddleOCRVL.py
```

#### 成功标志：



* 无导入错误、无编译错误。

* 首次运行会自动下载 VL 模型（约 5-10 分钟）。

* 输出目录/your/path/to/output(/back/aiengine/paddleocr/output)的识别结果（.jsom：结构化识别结果、.md ：Markdown格式文档等）。

## 三、本次部署版本配对表



| 组件               | 版本选择                | 配对原因                                  |
| ---------------- | ------------------- | ------------------------------------- |
| Python           | 3.8\~3.11（本次 3.10）  | 兼容 PaddlePaddle3.2.1 和 PaddleOCR3.x |
| CUDA             | ≥12.4(本次 12.4)        | PaddlePaddle向下兼容           |
| numpy            | 1.26.4(1.x0 系列)      | 兼容 cv2 和 PaddlePaddle，避开 2.x ABI 冲突   |
| PaddlePaddle-GPU | 3.2.1(cu126)          | 提供3.x API支持      |
| PaddleOCR        | 3.x                    | VL功能独立为PaddleOCRVL 类，兼容API                |
| safetensors      | 0.6.2.dev0           | 官方文档强制要求，PaddleOCR-VL 3.0 专用               |

## 四、指南



1. **虚拟环境激活**：所有操作必须在`(paddleocr_gpu_new)`环境中执行，否则依赖缺失。

2. **编译依赖齐全**：`build-essential`和`python3.10-dev`必须先装，否则 numpy 编译失败。

3. **API参数不兼容**：PaddlePaddle的API参数新旧2.0/3.0不兼容，详情见官方文档`https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html`

4. **依赖冲突**：若环境中已存在PyTorch，可能会与PaddlePaddle的CUDA库冲突，卸载即可`pip uninstall -y torch torchvision torchaudio`

5. **大批量处理优化**：对于大批量的图片处理，可根据需求调整参数，更多可调节功能详见官方文档：
```
 &#x20;  ocr = PaddleOCRVL(
 &#x20;    device="gpu:0",
 &#x20;    enable_hpi=False,
 &#x20;    precision="fp16",           # 半精度，显存占用减半
 &#x20;    cpu_threads=16,             # CPU 线程数（根据服务器核心数调整）
 &#x20;    enable_mkldnn=True,         # 启用 MKL-DNN CPU 加速
 &#x20;    # use_queues=True,          # 启用异步队列（处理大量图片时加速）
&#x20;     )
```
