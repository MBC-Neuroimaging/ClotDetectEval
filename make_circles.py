#!/usr/bin/env python3

import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

seg_dir = 'D:/cta_annotations/october_annotations/new'
out_dir = 'D:/cta_annotations/annotations_circles/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

segs = [file for file in os.listdir(seg_dir) if 'seg.nii.gz' in file]

for seg in segs:
    subject = seg.split('_')[0] + '_' + seg.split('_')[1]
    if os.path.exists(os.path.join(out_dir, subject + '_circle.nii.gz')):
        continue
    print(subject)
    gt = sitk.Cast(sitk.ReadImage(os.path.join(seg_dir, seg)),
                   sitk.sitkInt8)
    # connected component labelling
    gt_cc = sitk.ConnectedComponent(gt)
    # label statistics
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(gt_cc)
    if lsif.GetLabels():
        centroid = lsif.GetCentroid(1)
        # construct a 3D Gaussian sphere around it
        # sigma = [30, 30, 15]
        circle = sitk.GaussianSource(
            sitk.sitkInt16,
            size=gt_cc.GetSize(),
            sigma=[5, 5, 3],
            mean=gt_cc.TransformPhysicalPointToIndex(centroid))

        circle.CopyInformation(gt_cc)
        # calculate half max, to keep the circle diameter 2 x sigma
        max_val = np.max(sitk.GetArrayFromImage(circle))
        circle_binary = circle > max_val / 2
        sitk.WriteImage(circle_binary, os.path.join(out_dir, subject + '_circle.nii.gz'))
    else:
        sitk.WriteImage(gt, os.path.join(out_dir, subject + '_circle.nii.gz'))