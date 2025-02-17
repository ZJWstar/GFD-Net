# Enhancing Real-Time Object Detection with Global Additive Attention and Frequency-Aware Feature Fusion

![image](https://github.com/user-attachments/assets/9825f66d-f160-4937-a7d2-ab57706c0d2d)

In view of the fact that the existing YOLO model is difficult to meet people’s
needs in terms of detection accuracy and computational efficiency, especially in
small target detection and complex background, a GFP-YOLO real-time target
detection algorithm based on global convolution addition self attention mechanism (GCAS) and fully aggregated frequency sensing pyramid network structure
(FF-FPN) is proposed. 

Some libraries needed：
pip uninstall ultralytics
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
Modules that require compilation to run: dcnv3, dcnv4

You can run it by configuring the parameters in the 'MyTrain.py' or 'Myrun.cpy' files. Please note that the parameters in the 'Myrun.cpy' file are in parameters.txt

    
