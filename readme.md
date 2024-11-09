# 本文档记录在某项目中[Fast-Drone-XI35](https://github.com/Longer95479/Fast-Drone-XI35)的数字目标识别部分
## 文档组成：

```python
└─dataset_opensource
    ├─captured_CSI_camera_indoor
    │  ├─1
    │  ├─...
    ├─captured_CSI_camera_outdoor
    ├─emnist_dataset
    ├─final_used_dataset
    │  ├─1
    │  ├─...
    │  └─Z
    ├─generated_dataset_YOLO
    │  ├─images
    │  │  ├─test
    │  │  └─train
    │  └─labels
    │      ├─test
    │      └─train
    ├─letters_testset
    │  ├─1
    │  ├─...
    │  └─9
    └─letters_trainset
        ├─A
        ├─...
        └─Z
```

## 任务目标：识别目标区域和禁飞区域
示例：
目标区：

![目标区域2号](assets/2.png "目标区域2号") <!-- 此路径表示图片和MD文件，处于同一目录 -->

禁飞区：

![目标区域2号](assets/x.png "目标区域2号") <!-- 此路径表示图片和MD文件，处于同一目录 -->