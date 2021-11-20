# YOLOv5_OBB_KLD
基于kld_loss的YOLOv5 旋转目标检测。因为基于[项目](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB)
也可以YOLOv5 in DOTA_OBB dataset with CSL_label.(Oriented Object Detection)


## Datasets and pretrained checkpoint
* `Datasets` : [DOTA](https://link.zhihu.com/?target=http%3A//captain.whu.edu.cn/DOTAweb/)
* `Pretrained Checkpoint or Demo Files` : 
    * `train,detect_and_evaluate_demo_files`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/19BGy_UIdk8N-mSjHBMI0QQ). |  [Google Drive](https://drive.google.com/file/d/1MdKTgXQpHFBk_RN9UDSIB42M5e8zQaTP/view?usp=sharing) |
    * `yolov5x.pt`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/1pH6EGKZiIyGtoqUe3F8eWQ). |  [Google Drive](https://drive.google.com/file/d/1hGPB7iOl3EmB2vfm44xMpHJV8hPufHn2/view?usp=sharing) |
    * `yolov5l.pt`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/16is2mx879jk9_4RHwcIgKw). |  [Google Drive](https://drive.google.com/file/d/12ljwafulmAP1i9XsaeYvEnIUd18agJcT/view?usp=sharing) |
    * `yolov5m.pt`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/1ZQoxEB-1mtBAk3A-Rt85-A). |  [Google Drive](https://drive.google.com/file/d/1VSDegIUgTh-fMDIjuwTSQaZ1w5bVx2Vd/view?usp=sharing) |
    * `yolov5s.pt`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/1jm7ijb0a3LVkg8P2bkmJnw). |  [Google Drive](https://drive.google.com/file/d/1ePo6OM8MbxG8nAkZS_Bt7cmnChSlKBmo/view?usp=sharing) |
    * `YOLOv5_DOTAv1.5_OBB.pt`:  | [Baidu Drive(6666)](https://pan.baidu.com/s/1WSJFwwM5nyWgPLzAV6rp8Q). |  [Google Drive](https://drive.google.com/file/d/171xlq49JEiKJ3L-UEV9tICXltPs92dLk/view?usp=sharing) |

## Fuction
* `train.py`.  Train.

* `detect.py`. Detect and visualize the detection result. Get the detection result txt.

* `evaluation.py`.  Merge the detection result and visualize it. Finally evaluate the detector



## Installation  (Linux Recommend, Windows not Recommend)
`1.` Python 3.8 with all requirements.txt dependencies installed, including torch==1.6, opencv-python==4.1.2.30, To install run:
```
$   pip install -r requirements.txt
```
`2.` Install swig
```
$   cd  \.....\yolov5_OBB_KLD\utils
$   sudo apt-get install swig
```
`3.` Create the c++ extension for python
```
$   swig -c++ -python polyiou.i
$   python setup.py build_ext --inplace
```

## 效果
只是用DOTAv1.5的`ship`一类进行训练,超参数相同,KLD比CSL的AP50高0.3%,不过收敛很快。
## More detailed explanation
想要了解其他相关实现的细节和原理可以参看[项目](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB) 的`README.md`
这里主要介绍修改成KLD_LOSS的部分。
`1.` `'train.py' `

* `parser.add_argument('--use_kld', type=bool, default=True, help='use kld')`选择KLD or CSL
*  修改`.\models\yolo.py`的`Detect类`中初始化函数的`self.angle = 1   #CSL---180  KLD--1`

`2.` `'test.py'`
* 新增了在线推断代码

`3.` `'detect.py'` 
    
* 新增了多batch_size的检测,修改`ManyPi=True`
* `parser.add_argument('--kld', type=bool, default=True, help='use kld')` 对应KLD or CSL的检测

`4.` `'evaluation.py'` 

* Run the detect.py demo first. Then change the path with yours:
* 添加了merged前后的预测结果.



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