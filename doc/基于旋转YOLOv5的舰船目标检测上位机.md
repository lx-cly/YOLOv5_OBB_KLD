# 基于旋转YOLOv5的舰船目标检测上位机

## 项目打包

### 使用工具

利用pyinstaller工具打包，使用  ` pip install pyinstaller `安装，然后进入项目根目录，运行`pyinstaller 文件名.py`即可。因为项目部署在Linux系统上，生成的不是后缀名exe的可执行文件。因为代码并不是在一个py文件实现的，所以打包后在dist中有很多依赖文件，可执行文件也在dist目录中，进入目录运行`./Qt_main`该文件即可。

把资源放到与`dist`目录下,`data`目录存放数据集，`weights`存放预训练权重文件，`runs`存放训练结果，`inference`保存detect的图片。

## 项目安装

（Linux ubuntun 18.04 Recommend, Windows not Recommend)
`1.` Python 3.8 with all requirements.txt dependencies installed, including torch==1.6, opencv-python==4.1.2.30, To install run:

```shell
$   pip install -r requirements.txt
```
`2.` Install swig
```shell
$   cd  \.....\yolov5_DOTA_OBB\utils
$   sudo apt-get install swig
```
`3.` Create the c++ extension for python
```shell
$   swig -c++ -python polyiou.i
$   python setup.py build_ext --inplace
```
### 安装pyqt5
`1.` 更新pip3:

```shell
$  apt install python3-pip
$  python3 -m pip install --upgrade pip
```
`2.`安装pyqt5
```shell
$   pip3 install pyqt5
$   apt install pyqt5* #安装pyqt5的依赖项
$   apt install qt5-default qttools5-dev-tools # 安装qtdesigner
```

## 上位机简介

功能可以有五种：数据集信息，模型训练，模型评估，结果可视化，模型导出。点击会隐藏改界面，切换至新界面。

![image-20211018222758790](C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211018222758790.png)

数据集信息控制器选择训练集还是验证集，确定后，信息显示框会显示相应数据集的信息，包括舰船目标数目，目标大小，图片数目，图片大小及图片均值方差等。然后会弹出显示一张数据集信息化成的图片。点击浏览可以在文件夹查找数据集图片，然后会根据标签画出的标注图片。

<img src="C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211019162815798.png" alt="image-20211019162815798" style="zoom:50%;" />

风格切换支持三种。



![image-20211018222830503](C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211018222830503.png)

模型训练控制区，根据请根据提示输入参数，也可以不需要更改(使用默认参数)，点击确定开始训练。然后每10个iter，信息显示框会显示训练信息，下面也会实时显示loss曲线图。 <img src="C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211019190358894.png" alt="image-20211019190358894" style="zoom:70%;" />

模型评估控制区可以选择大图和小图，点击模型可以选择模型的路径，也可以不选择，直接点击确定有个默认的模型。信息显示区会显示相应的P,R，mAP等信息，并画出P-R图。



![image-20211018222854180](C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211018222854180.png)

<img src="C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211019173402366.png" alt="image-20211019173402366" style="zoom:50%;" />

<img src="C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211019173559257.png" alt="image-20211019173559257" style="zoom:50%;" />

模型可视化，点击选择任何一张含有舰船的图片，后边会出现检测之后的照片，点击保存可以把图片保存到任何路径。

![image-20211018222925122](C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211018222925122.png)

<img src="C:\Users\p'c\AppData\Roaming\Typora\typora-user-images\image-20211019163254770.png" alt="image-20211019163254770" style="zoom:50%;" />

