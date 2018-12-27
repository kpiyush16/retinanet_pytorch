# Pytorch Implementation for RetinaNet
Pytorch  implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Installation
1) Clone this repo
2) Install the required packages:
```
apt-get install tk-dev python-tk
```
3) Install the python packages:
```
pip install cffi
pip install pandas
pip install pycocotools
pip install cython
pip install opencv-python
pip install requests

```
4) Build the NMS extension.
```
cd pytorch-retinanet/lib
bash build.sh
cd ../
```
## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.

It uses two types of files: one file containing annotations and one file containing a class name to ID mapping.

The dataset can easily be generated using `csv_data_preparator.py` script. You can tweak script according to your requirements as it is build upon download upon 3 custom classes downloaded from [OIDv4_Toolkit](https://github.com/EscVM/OIDv4_ToolKit) i.e., `Person, Mobile Phone, Laptop`.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.

The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```
This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Training
The Network can be trained using `train.py`. This repository is specially for those who like to train on their custom class datasets which are downloaded from Open Image DataSet using `OIDv4_ToolKit` only.

For training and inferencing on COCO dataset, please refer to [Yann Henon's](https://github.com/yhenon/pytorch-retinanet) implementation.

Example training scenario:
```
python train.py --train data/train/train_annot.csv --val data/validation/valid_annot.csv --steps_per_stats 200 --savepath models_3c --depth 50 --classes data/class_ids_3c.txt
```
where `train` provides path for training annotations(similarly for validation set), `steps_per_stats` steps to show the statistics of training, `depth` for the ResNet backbone depth.

You can add `resume` argument and the number of epoch to resume training. Of course, you have to add all the other arguments as well including the `savepath` to search for.

## Inference
I have included an `Inference.ipynb` to provide much better inferencing on both pictures as well as videos. Check it out.
Some inferences are available at:
- https://drive.google.com/open?id=1O5bwhoToiigiqOUVpDfMXXEmOljIR_vP (In the form of pre-trained model which should be saved under `new_ckpts` folder in the repo as model is serialized via `torch.save()`)
- https://drive.google.com/open?id=1q1Q8e3YWu5i61sdu2jPJKv14al1VEq-d (An inference Video Rendered under the model)

> **Note:**
> Significant amount of code has been borrowed from [Yann Henon's](https://github.com/yhenon/pytorch-retinanet) implementation.
> The NMS module used is from the [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)
