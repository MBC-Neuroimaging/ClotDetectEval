#!/usr/bin/env python3

import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import sys

'''
Script to evaluated clot detection on benchmark dataset
User provides a csv with two columns:
column 0 "id" contains the identifying tags for the subject
column 1 "idx" contains the index location of clot as [x, y, z] within the 512X512X320 image
'''


def get_score(path_to_annotations, subject, idx, threshold):

    annotation = sitk.ReadImage(os.path.join(path_to_annotations, subject + '_seg.nii.gz'))
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(annotation)

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
                size=annotation.GetSize(),
                sigma=[30, 30, 15],
                mean=annotation.TransformPhysicalPointToIndex(centroid))

            circle.CopyInformation(annotation)
            circle = circle > 80

            prediction_circle = sitk.GaussianSource(
                sitk.sitkInt16,
                size=annotation.GetSize(),
                sigma=[30, 30, 15],
                mean=idx)

            prediction_circle.CopyInformation(annotation)
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
df = pd.read_csv(path_to_csv)

subjects = df['id'].to_list()

df.set_index('id', inplace=True, drop=True)

for sub in subjects:
    idx = df.loc[sub, 'idx']
    if pd.isnull(df.loc[sub, 'idx']):
        idx = []

    score = get_score(path_to_ann, sub, idx, 0.5)

    # calcualte TPR FPR etc