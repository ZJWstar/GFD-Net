# GFD-Net: Global-Frequency Dual-head Network for Real-Time Defect Detection

 ![image](https://github.com/ZJWstar/GFD-Net/blob/main/FIG1.pdf)
*图1: GFD-Net整体架构示意图（示意图来自论文）*

## 📖 项目简介
GFD-Net 是一种面向工业缺陷检测的高效实时目标检测算法，通过以下三大核心创新实现SOTA性能：
- **GCAS模块**：全局卷积加性自注意力机制，增强空间特征学习能力
- **FF-FPN架构**：全聚合频率感知特征金字塔网络，优化多尺度特征融合
- **TDSC检测头**：共享卷积的双任务检测头设计，实现分类与定位解耦

在COCO2017数据集上达到41.18% mAP，参数量仅2M；在自建MECD微电子连接器缺陷数据集上实现97%检测精度，模型大小仅2MB。



## ✨ 核心特性
- **轻量化设计**：相比YOLOv8减少40%参数量，适配嵌入式设备
- **频率感知融合**：通过FF-FPN模块提升微小缺陷检测能力
- **动态任务对齐**：TDSC检测头实现分类与定位任务自适应优化
- **即插即用模块**：GCAS/CDF等模块可独立集成到其他检测框架

## 🛠️ 快速开始

### 环境配置
```bash
# 基础环境
conda create -n gfdnet python=3.8
conda activate gfdnet

# 主要依赖
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install ultralytics==8.0.0 einops==0.6.1 timm==0.6.12
```

### 训练示例
```python
from backbone.creatbackbone import rcvit_m
from creation import Detect_TDSC

# 构建GCAS主干网络
backbone = rcvit_m(pretrained=True)

# 初始化TDSC检测头
detect_head = Detect_TDSC(nc=80, hidc=256)

# 完整模型构建
model = nn.Sequential(backbone, detect_head)
```

## 📊 性能对比
### COCO2017 数据集
| Model       | Params | mAP@50 | mAP@50-95 | FPS  |
|-------------|--------|--------|-----------|------|
| YOLOv8n     | 3.2M   | 52.6%  | 37.3%     | 131  |
| **GFD-Net** | **2.0M** | **56.6%** | **41.18%** | **76** |

### MECD 缺陷数据集
| 缺陷类型   | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| 裂纹检测   | 98.2%     | 89.7%  | 0.937    |
| 气泡检测   | 96.4%     | 83.1%  | 0.892    |
| 边缘缺陷   | 97.8%     | 86.5%  | 0.918    |

## 🧩 核心模块
### 1. GCAS (Global Convolutional Additive Self-attention)
```python
class GCAAtten(nn.Module):
    def __init__(self, ch_in, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3*dim, 1)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = GlobalSpatial(ch_in, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, groups=dim)
```

### 2. FF-FPN 特征融合
```python
class CDF(C2f):
    def __init__(self, c1, c2, n=1, k=2, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DynamicGraphConvBlock(in_dim=self.c, K=k) for _ in range(n))
```

### 3. TDSC 检测头
```python
class Detect_TDSC(nn.Module):
    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.share_conv = nn.Sequential(
            Conv_GN(hidc, hidc//2, 3), 
            Conv_GN(hidc//2, hidc//2, 3))
        self.cls_decomp = TaskDecomposition(hidc//2, 2, 16)
```


---
**提示**：完整训练代码与数据集预处理脚本将在近期更新，敬请关注项目更新！  
如有任何问题，欢迎提交Issue或联系 zjw@home.hpu.edu.cn
