# YOLOv5_OBB_KLD
<img src=".\doc\img.png" alt="img" style="zoom: 50%;" />
![img](.\doc\img_1.png)![img](.\doc\img_2.png)

**[代码](https://github.com/lx-cly/YOLOv5_OBB_KLD)实现了基于YOLOv5的遥感旋转框检测。利用CSL和KLD实现角度的学习，并且加入注意力机制提高检测效果。**


## 数据集和权重文件
* `数据集` : [预处理过后的数据集](https://pan.baidu.com/s/1eyiZyjOMH9dQ8nCsPfxTTQ ).若是想训练自定义数据集，预处理过程参看[项目](https://github.com/CAPTAIN-WHU/DOTA_devkit).
* `原始预训练的权重文件` : 
    * `yolov5x.pt、yolov5l.pt、yolov5m.pt、yolov5s.pt`:   [Baidu Drive(6666)](https://pan.baidu.com/s/1-YmcCv25f7OHzx8sBg5bpA ).


* `训练好的部分权重文件` : 
    * `YOLOv5_DOTAv1.5_OBB.pt`:   [Baidu Drive(6666)](https://pan.baidu.com/s/1iu7QZUPlVSzghFNSXk5P4w )

## 项目安装  (支持Linux系统)
`1.` Python 3.8 with all requirements.txt dependencies installed, including torch==1.6, opencv-python==4.1.2.30, To install run:
```shell
pip install -r requirements.txt
```
`2.` Install swig
```shell
cd  \.....\yolov5_OBB_KLD\utils
sudo apt-get install swig
```
`3.` Create the c++ extension for python

```shell
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## 训练
* `train.py`.  Note：修改`.\models\yolo.py`的`Detect类`中初始化函数的`self.angle = 180   #CSL对应180  KLD对应1`，默认使用CSL.

```python
python train.py --weights weights/yolov5m.pt --cfg models/yolov5m.yaml --use_kld False --device 0 --epochs 300 --batch_size 4 --workers 4 --logdir runs/    
```


## 评估
* `detect.py`. Detect and visualize the detection result. Get the detection result txt.

* `evaluation.py`.  Merge the detection result and visualize it. Finally evaluate the detector
```python
python detect.py --weights runs/exp/weights/best.pt --source 'dataset path' --output 'output path' --conf_thres 0.35 --iou_thres 0.4 --device 0 --kld False 
python evaluation.py 
''' example
检测结果已merge
检测结果已按照类别分类
校验数据集名称文件已生成
classname: ship
P: 0.8550878121966288
R: 0.900046446818393
map@0.5: 0.8889719225631516
classaps:  [     88.897]
原始存在文件,删除
检测结果已按照类别分类
校验数据集名称文件已生成
classname: ship
P: 0.8511538986754063
R: 0.8677432827509397
map@0.5: 0.8096364184338725
classaps:  [     80.964]
'''
```

## 结果展示

数据集图片尺寸裁剪为1024*1024，gap为10%。实验中NMS时统一使用的置信度阈值是0.35，IoU阈值是0.4。

![img](.\doc\img_4.png)


## 感激
感谢以下的项目,排名不分先后
* [BossZard/rotation-yolov5](https://github.com/BossZard/rotation-yolov5)
* [hukaixuan19970627/YOLOv5_DOTA_OBB](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB).
* [SJTU-Thinklab-Det/DOTA-DOAI](https://github.com/SJTU-Thinklab-Det/DOTA-DOAI)
* [buzhidaoshenme/YOLOX-OBB](https://github.com/buzhidaoshenme/YOLOX-OBB)

## 关于作者
```javascript
  Name  : "lx"
  describe myself："good man"
```
