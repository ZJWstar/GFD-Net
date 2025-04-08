# GFD-Net: Defect Localization Network based on Global Additive Attention and Frequency-aware Feature Fusion

 ![image](https://github.com/ZJWstar/GFD-Net/blob/main/image.png)

## 📖 Project introduction
GFD net is an efficient real-time target detection algorithm for industrial defect detection. It achieves SOTA performance through the following three core innovations：
- **GCAS Module**：Global convolution additive self attention mechanism enhances spatial feature learning ability
- **FF-FPN Architecture**：Fully aggregate frequency aware feature pyramid network to optimize multi-scale feature fusion
- **TDSC Detection head**：Design of dual task detection head with shared convolution to realize decoupling of classification and positioning

It reaches 41.18% map on the coco2017 dataset, and the parameter quantity is only 2m; 97% detection accuracy is achieved on the self built MeCD microelectronic connector defect data set, and the model size is only 2MB.


## ✨ Core Features
- **Lightweight Design**：It only has 2m parameters and is suitable for embedded devices
- **Frequency Aware Fusion**：Improve the ability of micro defect detection through FF-FPN module
- **Dynamic Task Alignment**：Adaptive optimization of TDSC detector head for classification and positioning tasks
- **Plug and Play Module**：Gcas/cdf and other modules can be independently integrated into other detection frameworks

## 🛠️ Quick Start

### Environment Configuration
```bash
# Basic environment
conda create -n gfdnet python=3.8
conda activate gfdnet

# Major dependencies
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install ultralytics==8.0.0 einops==0.6.1 timm==0.6.12
```

### Training examples
```python
from backbone.creatbackbone import rcvit_m
from creation import Detect_TDSC

# Building GCAS backbone network
backbone = rcvit_m(pretrained=True)

# Initialize TDSC detector head
detect_head = Detect_TDSC(nc=80, hidc=256)

# 完Complete model construction
model = nn.Sequential(backbone, detect_head)
```

## 📊 Performance Comparison
 ![image](https://github.com/ZJWstar/GFD-Net/blob/main/TABLE.png)

## 🧩 Core Module
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

### 2. FF-FPN feature fusion
```python
class CDF(C2f):
    def __init__(self, c1, c2, n=1, k=2, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DynamicGraphConvBlock(in_dim=self.c, K=k) for _ in range(n))
```

### 3. TDSC Detection head
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
**Prompt**：The complete training code and data set preprocessing script will be updated after the paper is published. Please pay attention to the project update！  
If you have any questions, please submit issue or contact zjw@home.hpu.edu.cn
