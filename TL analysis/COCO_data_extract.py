from gluoncv import data, utils
from matplotlib import pyplot as plt
import pandas as pd


def objects_scale(point_1, point_2, width, height):
    # Normalize the size of the objects w.r.t the size of the image
    x_width = float(abs(point_2[0] - point_1[0]))
    y_height = float(abs(point_2[1] - point_1[1]))
    normalized_area = (x_width * y_height) / (width * height)
    return normalized_area


def aspect_ratio(point_1, point_2):
    x_width = float(abs(point_2[0] - point_1[0]))
    y_height = float(abs(point_2[1] - point_1[1]))
    try:
        aspct_ratio = x_width / y_height
    except:
        aspct_ratio = 0.0
    return aspct_ratio


train_dataset = data.COCODetection('.', splits=['instances_train2017'])
val_dataset = data.COCODetection('.', splits=['instances_val2017'])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

train_image, train_label = train_dataset[37572]
bounding_boxes = train_label[:, :4]
class_ids = train_label[:, 4:5]
print(train_label.shape)
print('Image size (height, width, RGB):', train_image.shape)
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
      bounding_boxes)
print('Class IDs (num_boxes, ):\n', class_ids)

utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=train_dataset.classes)
plt.show()

info = {'class_ids': [], 'scales': [], 'aspect_ratio': []}

for i in range(len(train_dataset)):
    if i % 500 == 0:
        print('percentage done:', (i / (len(train_dataset))) * 100)
    train_image, train_label = train_dataset[i]
    bounding_boxes = train_label[:, :4]
    class_ids = train_label[:, 4:5]
    image = train_image.asnumpy()
    height, width = image.shape[:2]
    for j in range(len(class_ids)):
        x1 = int(bounding_boxes[j][0])
        y1 = int(bounding_boxes[j][1])
        x2 = int(bounding_boxes[j][2])
        y2 = int(bounding_boxes[j][3])
        info['class_ids'].append(class_ids[j][0])
        info['scales'].append(objects_scale((x1, y1), (x2, y2), width, height))
        info['aspect_ratio'].append(aspect_ratio((x1, y1), (x2, y2)))

df = pd.DataFrame(data=info)
df.to_csv('COCO_stats.csv', index=False)

