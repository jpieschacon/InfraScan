import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# Read all text files from a directory to see the YOLO formatted bounding boxes of each image
path_txt = 'C:/Users/juangabriel/Dropbox (Politecnico Di Torino Studenti)/LabelImages/ShufflingAll/shufflingAll/obj/*.txt'
files_txt = glob.glob(path_txt)
# To save the width and height of each object (i.e. spall/crack)
data = {'obj_width': [], 'obj_height': []}
for file in files_txt:
    if file[(len(path_txt)-5):] != 'classes.txt':
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                annotation = line.replace('\n', '')
                y_height = float(annotation.split(sep=' ')[-1:][0])
                x_width = float(annotation.split(sep=' ')[-2:-1:][0])
                data['obj_width'].append(x_width)
                data['obj_height'].append(y_height)
damage_df = pd.DataFrame(data=data)
damage_df['aspect_ratios'] = damage_df['obj_width']/damage_df['obj_height']
damage_df['object_scale'] = damage_df['obj_width']*damage_df['obj_height']

# Whole COCO dataset
COCO_df = pd.read_csv('COCO_stats.csv')
scales = np.arange(0, 1.05, 0.05)
aspect_ratios_COCO = [0, 1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, COCO_df['aspect_ratio'].max()]
aspect_ratios_damage = [0, 1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, damage_df['aspect_ratios'].max()]
scale = {'scales': scales[1:]}
aspect_ratio_COCO = {'aspect_ratios': aspect_ratios_COCO[1:]}
aspect_ratio_damage = {'aspect_ratios': aspect_ratios_damage[1:]}
COCO_scales_df = pd.DataFrame(data=scale)
COCO_aspect_rat_df = pd.DataFrame(data=aspect_ratio_COCO)
damage_scales_df = pd.DataFrame(data=scale)
damage_aspect_rat_df = pd.DataFrame(data=aspect_ratio_damage)
COCO_scales_df['all'] = COCO_df.groupby(pd.cut(COCO_df['scales'], scales))['scales'].count().array/len(COCO_df)  # Divide by the total number of objects to normalize
COCO_aspect_rat_df['all'] = COCO_df.groupby(pd.cut(COCO_df['aspect_ratio'], aspect_ratios_COCO))['aspect_ratio'].count().array/len(COCO_df)
damage_scales_df['all'] = damage_df.groupby(pd.cut(damage_df['object_scale'], scales))['object_scale'].count().array/len(damage_df)
damage_aspect_rat_df['all'] = damage_df.groupby(pd.cut(damage_df['aspect_ratios'], aspect_ratios_damage))['aspect_ratios'].count().array/len(damage_df)

# Discriminate the statistics for every class
for i in range(len(COCO_df['class_ids'].unique())):
    COCO_scales_df[f'scales_{i}'] = COCO_df.loc[COCO_df['class_ids'] == i].groupby(pd.cut(COCO_df['scales'].loc[COCO_df['class_ids'] == i], scales))['scales'].count().array/len(COCO_df.loc[COCO_df['class_ids'] == i])
    COCO_aspect_rat_df[f'aspect_ratios_{i}'] = COCO_df.loc[COCO_df['class_ids'] == i].groupby(pd.cut(COCO_df['aspect_ratio'].loc[COCO_df['class_ids'] == i], aspect_ratios_COCO))['aspect_ratio'].count().array/len(COCO_df.loc[COCO_df['class_ids'] == i])

# Calculate the euclidean distance error
euclidean = {'scales_dist': [], 'aspect_ratios_dist': []}
for column in COCO_scales_df.columns:
    euclidean['scales_dist'].append(np.linalg.norm(damage_scales_df['all']-COCO_scales_df[column]))
for column in COCO_aspect_rat_df.columns:
    euclidean['aspect_ratios_dist'].append(np.linalg.norm(damage_aspect_rat_df['all']-COCO_aspect_rat_df[column]))
euclidean_df = pd.DataFrame(data=euclidean, index=COCO_scales_df.columns)
euclidean_df['sum'] = euclidean_df['scales_dist']+euclidean_df['aspect_ratios_dist']
# Sort in ascending order the sum of the euclidean distances of scales and aspect ratios
euclidean_df.sort_values(by=['sum'])

# 4 classes with minimum error
classes = [69, 16, 5, 76]
extracted_COCO_df = pd.concat([COCO_df.loc[COCO_df['class_ids'] == classes[0]],
                               COCO_df.loc[COCO_df['class_ids'] == classes[1]],
                               COCO_df.loc[COCO_df['class_ids'] == classes[2]],
                               COCO_df.loc[COCO_df['class_ids'] == classes[3]]])
COCO_scales_df['extracted'] = extracted_COCO_df.groupby(pd.cut(extracted_COCO_df['scales'], scales))['scales'].count().array/len(extracted_COCO_df)  # Divide by the total number of objects to normalize
COCO_aspect_rat_df['extracted'] = extracted_COCO_df.groupby(pd.cut(extracted_COCO_df['aspect_ratio'], aspect_ratios_COCO))['aspect_ratio'].count().array/len(extracted_COCO_df)

plt.figure()
plt.plot(scales[1:], COCO_scales_df['all'], marker='o', linestyle='--', color='r', label='COCO')
plt.plot(scales[1:], damage_scales_df['all'], marker='v', linestyle='--', color='b', label='Damages')
plt.plot(scales[1:], COCO_scales_df['extracted'], marker='D', linestyle='--', color='g', label='Extracted COCO')
plt.xlabel('Object scale')
plt.ylabel('Percent of objects')
plt.xticks(scales[1:], rotation=45)
plt.title('Object scale')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('report_images/object_scale.pdf')

# create an index for each tick position
xi = list(range(len(aspect_ratios_COCO)-1))
x_ax = ['1/10', '1/9', '1/8', '1/7', '1/6', '1/5', '1/4', '1/3', '1/2', '1', '2', '3', '4', '5', '6', '7', '8', '9', '>10']
# plot the index for the x-values
plt.figure()
plt.plot(xi, COCO_aspect_rat_df['all'], marker='o', linestyle='--', color='r', label='COCO')
plt.plot(xi, damage_aspect_rat_df['all'], marker='v', linestyle='--', color='b', label='Damages')
plt.plot(xi, COCO_aspect_rat_df['extracted'], marker='D', linestyle='--', color='g', label='Extracted COCO')
plt.xlabel('Aspect ratio')
plt.ylabel('Percent of objects')
plt.xticks(xi, x_ax, rotation=45)
plt.title('Object aspect ratio')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('report_images/aspect_ratio.pdf')
