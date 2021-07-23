#!/usr/bin/env python3

import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

'''
Script to evaluated clot detection on benchmark dataset
User provides a csv with two columns:
column 0 "id" contains the identifying tags for the subject
column 1 "idx" contains the index location of clot as [x, y, z] within the 512X512X320 image
'''


def get_score(path_to_annotations, subject, idx, threshold):

    annotation = sitk.ReadImage(os.path.join(path_to_annotations, subject + '_seg.nii.gz'))

    # connected component labelling
    annotation_cc = sitk.ConnectedComponent(annotation)

    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(annotation_cc)

    # if there is no clot
    if len(lsif.GetLabels()) == 0:
        # if you have provided an index
        if not idx:
            return 'tn'
        if idx:
            return 'fp'

    else:
        if idx:
            centroid = lsif.GetCentroid(1)

            circle = sitk.GaussianSource(
                sitk.sitkInt16,
                size=annotation_cc.GetSize(),
                sigma=[30, 30, 15],
                mean=annotation_cc.TransformPhysicalPointToIndex(centroid))

            circle.CopyInformation(annotation_cc)
            circle = circle > 80

            prediction_circle = sitk.GaussianSource(
                sitk.sitkInt16,
                size=annotation.GetSize(),
                sigma=[30, 30, 15],
                mean=idx)

            prediction_circle.CopyInformation(annotation_cc)
            prediction_circle = prediction_circle > 80

            circle_array = sitk.GetArrayFromImage(circle)
            circle_array = circle_array/np.max(circle_array)

            # set rois to 1
            pred_array = sitk.GetArrayFromImage(prediction_circle)
            pred_array = pred_array/np.max(pred_array)

            # change background values so they are different from each other
            circle_array[circle_array == 0] = 10
            pred_array[pred_array == 0] = 5

            # subtract the prediction circle from the ground truth circle
            iou_array = circle_array - pred_array

            # count intersecting voxels
            intersection = len(np.where(iou_array == 0)[0])

            # count union voxels
            union = len(
                np.where(iou_array == -4)[0]) + len(
                np.where(iou_array == 9)[0]) + intersection

            IoU = intersection/union

            if IoU > threshold:
                return 'tp'
            else:
                return 'fp'
        if not idx:
            return 'fn'


# read in csv
path_to_csv = sys.argv[1]
path_to_ann = sys.argv[2]
save_fig = sys.argv[3]

df = pd.read_csv(path_to_csv)

subjects = df['id'].to_list()

df.set_index('id', inplace=True, drop=True)


thresholds = np.arange(10, 100, 5)
score_df = pd.DataFrame(
    index=subjects,
    columns=[str(thresh) + '%' for thresh in thresholds])

for sub in subjects:
    idx = df.loc[sub, 'idx']
    if pd.isnull(df.loc[sub, 'idx']):
        idx = []

    score = get_score(path_to_ann, sub, idx, 0.5)
    df.loc['sub', 'score'] = score

    for thresh in thresholds:
        score = get_score(path_to_ann, sub, idx, thresh/10)
        column = str(thresh) + '%'
        score_df.loc[sub, column] = score

# Calculate TPR and FPR over entire dataset
tpr_row = pd.DataFrame(
    index=['tpr'],
    columns=[str(thresh) + '%' for thresh in thresholds]
)
fpr_row = pd.DataFrame(
    index=['fpr'],
    columns=[str(thresh) + '%' for thresh in thresholds]
)

subjects = [str(num) for num in np.arange(100)]
score_df = pd.DataFrame(
    index=subjects,
    columns=[str(thresh) + '%' for thresh in thresholds])

for col in score_df:
    series = score_df[col]
    # get tpr
    counts = series.value_counts()
    tpr = counts['tp'] / (counts['tp'] + counts['fp'])
    fpr = counts['fp'] / (counts['fp'] + counts['tn'])
    tpr_row.loc['tpr', col] = round(tpr, 3)
    fpr_row.loc['fpr', col] = round(fpr, 3)


score_df = score_df.append(tpr_row)
score_df = score_df.append(fpr_row)


def create_triangle(tpr_0, tpr_1, fpr_0, fpr_1):
    plt.plot([tpr_0, tpr_1], [fpr_0, fpr_1], '-', lw=2, color='#4285F4')
    plt.plot([tpr_0, tpr_1], [fpr_1, fpr_1], '-', lw=2, color='#4285F4')
    plt.plot([tpr_0, tpr_0], [fpr_0, fpr_1], '-', lw=2, color='#4285F4')


def create_rectangle(tpr_0, fpr_0):
    plt.plot([tpr_0, tpr_0], [fpr_0, 0], '-', lw=2, color='#4285F4')


sns.set()
plt.figure(figsize=(15, 7))

plt.scatter(fpr_row, tpr_row, color='#0F9D58', s=100)


plt.title('ROC Curve', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)

# pending testing (adding rectangles underneath)
# plt.plot([1, 0], [0, 0], "-", lw=2, color="#4285F4")
# for j in range(len(thresholds) - 1):
#     create_rectangle(
#         fpr_row.iloc[0, j],
#         tpr_row.iloc[0, j])
# for k in range(len(thresholds) - 1):
#     create_triangle(
#         fpr_row.iloc[0, k],
#         fpr_row.iloc[0, k + 1],
#         tpr_row.iloc[0, k],
#         tpr_row.iloc[0, k + 1])

plt.savefig(os.path.join(save_fig, "roc_curve.png"))

rectangle_auc = 0
for k in range(len(thresholds)-1):
    rectangle_auc += (fpr_row.iloc[0, k + 1] - fpr_row.iloc[0, k]) * tpr_row.iloc[0, k]

