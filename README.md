# YOLOv8n-SMALL-OBJECTS-DETECTION
Modify yolov8n's structure or functions to optimize its performance on small objections

## Dataset

### SOD Dataset
SOD (Salient Object Detection) dataset is a collection of images that are annotated and labeled to identify the salient objects present in them. Salient objects refer to the visually distinct or important elements in an image that tend to attract human attention. SOD datasets are commonly used in computer vision research for developing and evaluating algorithms and models that can automatically detect and segment salient objects in images. These datasets typically consist of images along with corresponding ground truth annotations where salient regions or objects are marked or outlined.

![image](https://github.com/zhuty2001/YOLOv8n-SMALL-OBJECTS-DETECTION/assets/68087747/30931aba-b36b-4668-8bc5-8b7b569f0302)

### Urban Zone Aerial Object Detection Dataset
The dataset is a combination of 3 others dataset, being them Stanford Drone Dataset, Vision Meets Drones, and Umanned Unmanned Aerial Vehicles Benchmark Object Detection and Tracking.

![Stanford_bookstore_video0_1050](https://github.com/zhuty2001/YOLOv8n-SMALL-OBJECTS-DETECTION/assets/68087747/83948415-2eb6-468e-94df-9c9a514165d8)

## Optimize
The original yolov8 structure has three object detectors, which can detect objects larger than 32*32, objects larger than 16*16 and objects larger than 8*8. To optimize the ability of detecting small objects, we add small objects detecting layer to get shallower features and enhance yolov8n's capability of detecting small objects by modifying the yolov8.yaml document.

```bash
# YOLOv8.0s head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 13
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 17 (P3/8-small)
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [128]]  # 20 (P4/16-medium)
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [256]]  # 20 (P4/16-medium)
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [512]]  # 23 (P5/32-large)
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 23 (P5/32-large)
 
  - [[18, 21, 24,27], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

## Train and Result

We use the following instructions to train the model and get the corresponding results on the specified dataset。

```bash
%pip install ultralytics
import ultralytics
ultralytics.checks()
```

```bash
# Load a model
from ultralytics import YOLO
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='uz.yaml', epochs=30, imgsz=640)
```

