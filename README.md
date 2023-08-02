# yolov5数据准备

视频抽帧标注、剪裁物体图片、标注格式（JSON->XML）转换、yolov5数据集生成和更新。


## 使用

### 视频抽帧标注

从视频中抽帧标注，补充数据集。

由于在监控等视频中，我们关注的物体可能并不是每帧都出现，而是在某一段时间集中出现，其余大部分时间都不出现。例如，在垃圾场工作的环卫工人，在某个时间点集中作业，其余时间外出收垃圾。如果采用均匀抽帧的方式，会导致抽取大量无效、不包含目标物体的图片。因此，利用[yolov5预训练模型](https://github.com/ultralytics/yolov5/releases/tag/v5.0)，在抽帧前先判断该帧内有没有目标的物体（例如人、车），若有则抽帧，若无则继续。官方的预训练模型基于COCO数据集，标签共[80](https://zhuanlan.zhihu.com/p/263454360)个，常用的是person，当然也可以用自己预训练好的模型和权重。

**均匀抽帧用法：**

```shell
$ python video_extract_frames.py \ 
		--video_folder_path ./videos \
		--video_format .mp4 \
		--save_dir ./output \
		--save_format .jpg \
		--frame_inter 600
```

**目标检测抽帧用法：**

```shell
$ python video_extract_frames.py \ 
		--video_folder_path ./videos \
		--video_format .mp4 \
		--save_dir ./output \
		--save_format .jpg \
		--frame_inter 600 \
		--filter \
		--classes person car \
		--relationship and \
		--device cpu
```

终端打印以下类似信息：

```shell
[INFO] Options: Namespace(classes=None, conf_thres=0.25, device='cpu', filter=False, first_video=None, frame_inter=60, img_size=640, iou_thres=0.45, relationship='and', save_dir='./output', save_format='.jpg', video_folder_path='./videos', video_format='.mp4', weights='./yolov5m.pt')
[INFO] 1/3 video: suit_hat.mp4, W: 1920.0, H: 1080.0, FPS: 22.04, FC: 2360.0, Time(min): 1.78.
[INFO] Creating save_dir: ./output\suit_hat, Done.
[INFO] Frame at Time(s): 0.00/107.07 of 1/3 video: suit_hat.mp4 saved with suit_hat_0.jpg!
[INFO] Frame at Time(s): 2.72/107.07 of 1/3 video: suit_hat.mp4 saved with suit_hat_1.jpg!
···
[INFO] 3/3 video: suit_hat3.mp4, W: 1920.0, H: 1080.0, FPS: 22.04, FC: 2360.0, Time(min): 1.78.
[INFO] Frame at Time(s): 0.00/107.07 of 3/3 video: suit_hat3.mp4 saved with suit_hat3_0.jpg!
[INFO] Frame at Time(s): 2.72/107.07 of 3/3 video: suit_hat3.mp4 saved with suit_hat3_1.jpg!
...
[INFO] Frames extracting done in 0.74 min!
```

可选参数列表：

- video_folder_path：视频文件的存放位置
- video_format：指定抽取的视频文件的格式
- save_dir：抽取的图片存放位置
- save_format：抽取的图片的格式
- first_video：指定要抽取的第一个视频文件名，如6.mp4，跳过前面的视频，视频文件路径列表是通过glob读取
- frame_inter：帧间隔，每隔frame_inter帧抽取一次
- filter：使用yolov5预训练模型过滤无目标物体的帧
- classes：目标物体类别
- relationship：当有多个目标物体时，指定and/or过滤模式。and：多个目标物体都必须在帧内出现，or：多个目标物体中有一个在帧内出现
- weights：预训练权重位置
- img-size：输入图片大小
- conf-thres：置信度阈值
- iou-thres：交并比阈值
- device：模型推理设备，**目前指定为cpu**

### 裁剪物体图片

对于大图，有时我们需要利用在大图上的标注信息，裁剪出目标物体，也就是将大图变为小图，后续再训练分类模型等。例如，将监控中的环卫员，人体的部分裁剪出来。对于裁剪出的图片，相应的XML文件也要修改并对应生成。

**用法：**

```shell
$ python crop_object.py \
		--img_folder_path ./images \
		--xml_folder_path ./annotations \
		--save_dir ./output \
		--width_scaler 0.1 0.4 \
		--height_scaler 0.05 0.3
```

终端打印以下类似信息：

```shell
[INFO] Options: Namespace(height_scaler=[0.05, 0.3], img_folder_path='./images', save_dir='./output', width_scaler=[0.1, 0.4], xml_folder_path='./annotations')     
[INFO] Find 5000 images.
[INFO] Cropping: 100%|█████████████████████████████████████████████████████████| 5000/5000 [02:53<00:00, 28.81it/s] 
[INFO] Crop object regions, Done.
```

可选参数列表：

- img_folder_path：大图文件夹
- xml_folder_path：标注文件夹
- save_dir：保存位置，小图保存至save_dir/images子文件夹，xml保存至save_dir/Annotations子文件夹
- width_scaler：宽度缩放范围[min, max]，在大图标注框的宽度上随机增加[min * width, max * width]
- height_scaler：高度缩放范围[min, max]，在大图标注框的高度上随机增加[min * height, max * height]，scaler本质上是对目标区域图像的padding

### 标注格式转换

部分数据集提供的标注信息是JSON格式的，而yolov5数据集生成需要的是XML格式，XML格式也较为方便。对于JSON格式转XML格式，我们只需要读取JSON文件，并按结构创建结点，保存在XML文件里即可。

以下程序是针对DeepFashion2数据集所写，并将原始json中的object整合为XML的一个object，针对其他数据集的json转XML，可以参考修改。

在命令行中调用`make_XML_from_JSON.py`：

```shell
$ python make_XML_from_JSON.py \
		--json_folder_path ./annos \
		--img_folder_path ./image \
		--save_dir ./test \
		--object_name suit
```
终端打印以下类似信息：
```shell
[INFO] Options: Namespace(img_folder_path='./image', json_folder_path='./annos', object_name='suit', save_dir='./test')
[INFO] Creating save_dir: ./test, Done.
[INFO] Converting JSON to XML: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7697/7697 [01:53<00:00, 67.60it/s] 
[INFO] Convert jsons to XMLs, Done.
```
可选参数列表：
- json_folder_path：JSON文件夹位置
- img_path：图片文件夹位置
- save_dir：目标XML文件夹位置
- object_name：物体标签名称


### 标签统计与修改

有时，我们需要对数据集标注文件中的标签进行统计，来判断标签分布是否均衡。如果分布不均衡，或者想将一些小标签合并为大标签（如：safety_hat, mesh_hat, body_hat->hat），或者有一些错误标注标签（如：cigarette写成了cigarettee），我们还需要对原始XML文件的标签（name）进行修改。

**标签统计用法：**

```shell
$ python modify_XML_label.py \ 
		--anno_folder_path ./Annotations \
		--stats
```

终端打印以下类似信息：

```shell
[INFO] Options: Namespace(after_class=None, anno_folder_path='./Annotations', before_classes=None, rename=False, stats=True)
[INFO] Stats Label: 100%|██████████████████████████████████████████████████████| 5533/5533 [01:46<00:00, 51.90it/s] 
[INFO] Label stats: {'head': 90839, 'safety_hat': 1532, 'mesh_hat': 4521, 'body_hat': 611}.
```

**标签修改用法:**

```shell
$ python modify_XML_label.py \ 
		--anno_folder_path ./Annotations \
		--rename \
		--before_classes cigarette. cigarettee cigar \
		--after_class cigarette
```

终端打印以下类似信息：

```shell
[INFO] Options: Namespace(after_class=cigarette, anno_folder_path='./Annotations', before_classes=['cigarette.', 'cigarettee', 'cigar'], rename=True, stats=False)
[INFO] Rename Label: 100%|█████████████████████████████████████████████████████| 5533/5533 [01:46<00:00, 51.90it/s] 
[INFO] Label rename, Done.
```

可选参数列表：

- anno_folder_path：XML文件夹位置
- stats：统计标签
- rename：修改标签(inplace)
- before_classes：需要修改的标签
- after_class：修改后的标签

### yolov5数据集生成和更新

整理数据是一个麻烦的事情，包括数据的重命名、图片与标注要匹配对应、数据划分等等。更麻烦的是，项目经常会迭代更新，补充新的数据到数据集中。而且，yolov5的标签格式是专用的，需要通过XML计算得到txt文本。为了方便解决这些问题，设置imgs和annos两个文件夹，也就是不管是第一次生成数据集，还是补充新数据到数据集中，把图片和标注文件（文件夹亦可）一股脑分别放到这两个文件夹中，再调用命令，选择功能，来生成或更新数据集到指定的位置。

在命令行中调用`prepare_yolo_data.py`：

```shell
$ python prepare_yolo_data.py \
		--img_folder_path ./imgs \
		--anno_folder_path ./annos \
		--save_dir ./test \
		--rename \
		--split \
		--label \
		--classes head safety_hat \
		--prefix /path/to/your/save_dir/images/folder/when/training/ \
		--plot \
		--plot_num 20
```
终端打印以下类似信息：

```shell
[INFO] Options: Namespace(anno_folder_path='./annos', classes=['head', 'safety_hat'], img_folder_path='./imgs', label=True, plot=True, plot_num=20, prefix='home/user/Data/test/images/', 
rename=True, save_dir='./test', seed=42, split=True, test_rate=0.05, train_rate=0.9, val_rate=0.05)
[INFO] Match and Copy ./imgs, ./annos to ./test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1806/1806 [00:01<00:00, 1474.65it/s] 
[INFO] Run Split Process!
[INFO] txt file saved! split is done.
[INFO] Run Label Process!
[INFO] Convertor runs from train.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3251/3251 [00:06<00:00, 489.63it/s] 
[INFO] Convertor runs from val.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 180/180 [00:00<00:00, 477.43it/s]
[INFO] Convertor runs from test.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 181/181 [00:00<00:00, 504.08it/s]
[INFO] Data convert done!
[INFO] Plot done!
```

可选参数列表：
- img_folder_path：原始（或补充）的图片文件夹
- anno_folder_path：原始（或补充）的标注文件夹
- save_dir：生成后/更新时的数据集位置
- rename：是否对本次的图片重命名，这是考虑到原始图片可能会有重复名字，在复制粘贴到数据集位置时，可能会因为重名导致图片覆盖，从0.jpg开始递增命名
- split：是否划分数据集为训练集、验证集、测试集
- train_rate：训练集比例
- val_rate：验证集比例
- test_rate：测试集比例
- label：是否将XML文件的标注信息转为yolov5所用的格式，保存为txt文件
- prefix：前缀，yolov5所用数据集中必须包含train.txt、val.txt、test.txt三个文本文件，里面要注明图片的位置，建议用绝对路径，文本文件里每行就是prefix+图片名，prefix本质上就是save_dir的绝对路径+images/
- classes：XML文件可能包含多个类别信息，但我们可能只关心其中的某几类，才需要生成对应的label文本文件
- plot：是否在数据集生成或更新完毕后，在一些图片上绘制Bounding Box并保存
- plot_num：绘制Bounding Box的图片的数量
- seed：随机划分时种子

rename、split、label、plot、四个基本功能可以组合使用，也可以单独使用。数据集格式：
```text
/home/user/Data/test/
├── Annotations
│   ├── 0.xml
│   ├── 1.xml
│   └── 2.xml
├── images
│   ├── 0.jpg
│   ├── 1.jpg
│   └── 2.jpg
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── labels
│   ├── 0.txt
│   ├── 1.txt
│   └── 2.txt
├── test.txt
├── train.txt
├──── home/user/Data/test/images/0.jpg
├──── home/user/Data/test/images/1.jpg
├──── home/user/Data/test/images/2.jpg
└── val.txt
```
数据集生成完毕后，在yolov5训练所需的yaml文件中，填好相应的txt文件地址即可开始训练：
```yaml
train: /home/user/Data/test/train.txt
val: /home/user/Data/test/val.txt
test: /home/user/Data/test/test.txt
# number of classes
nc: 4
# class names
names: ['head', 'safety_hat', 'mesh_hat', 'body_hat']
```
## 更新日志

[2023/05/30] 创建Repo，更新README

[2023/06/02] 更新README，完善数据抽帧标注部分的代码逻辑，增加标注文件的标签统计与标签修改功能

[2023/06/20] 更新README，完善裁剪物体图片代码

[2023/08/02] 更新README，完善yolov5数据集生成和更新、标注格式转换的代码

## TODO：增加更多功能

## 如果本项目对您有帮助，欢迎点一个:star:！欢迎提出ISSUES，共同完善项目！