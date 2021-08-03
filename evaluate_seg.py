#!/usr/bin/env python3

import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

'''

***********
PLEASE READ
***********

Script to evaluated clot detection on benchmark dataset

Usage: python3 evaluate_seg.py path_to_roi path_to_ann save_results

Input arguments:

path_to_rois:
Full path to where ROIs are located
* User provides predictions in the form of roi labelled '<subject_id>_roi.nii.gz
* Subject IDs are in format sub-test* 
* ROI images must have the same image properties (size, direction, origin, spacing) as the corresponding raw image:
    e.g., roi.GetSize() == annotation.GetSize()

path_to_annotations:
Full path to where the ground truth annotations are held
* Annotations are in the format '<subject_id>_seg.nii.gz
* A 3D ellipse that is [60, 60, 30] voxels wide is built around the ground truth annotations

save_results:
Full path to location for output ROC figure

'''


def get_score(path_to_roi, path_to_gt, subject, threshold):

    gt = sitk.ReadImage(path_to_gt)
    roi = sitk.ReadImage(path_to_roi)

    # connected component labelling
    gt_cc = sitk.ConnectedComponent(gt)

    # label statistics
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(gt_cc)

    # read in segmentation
    lsif_roi = sitk.LabelShapeStatisticsImageFilter()
    lsif_roi.Execute(roi)

    # if there is no clot
    if not lsif.GetLabels():
        # if you have provided an roi
        if not lsif_roi.GetLabels():
            return 'tn'
        # if you have provided an roi
        if lsif_roi.GetLabels():
            return 'fp'

    # if there is a clot
    else:
        # if you have provided an roi
        if lsif_roi.GetLabels():
            # get the location of the annotation centroid
            centroid = lsif.GetCentroid(1)
            # construct a 3D Gaussian sphere around it
            # sigma = [30, 30, 15]
            circle = sitk.GaussianSource(
                sitk.sitkInt16,
                size=gt_cc.GetSize(),
                sigma=[30, 30, 15],
                mean=gt_cc.TransformPhysicalPointToIndex(centroid))

            circle.CopyInformation(gt_cc)
            # calculate half max, to keep the circle diameter 2 x sigma
            max_val = np.max(sitk.GetArrayFromImage(circle))
            circle = circle > max_val /2

            # normalize labels
            circle_array = sitk.GetArrayFromImage(circle)
            circle_array = circle_array/np.max(circle_array)

            roi_array = sitk.GetArrayFromImage(roi)
            roi_array = roi_array/np.max(roi_array)

            # change background values so they are different from each other
            circle_array[circle_array == 0] = 10
            roi_array[roi_array == 0] = 5

            # subtract the prediction circle from the ground truth circle
            iou_array = circle_array - roi_array

            # count intersecting voxels
            intersection = len(np.where(iou_array == 0)[0])

            # count union voxels
            union = len(
                np.where(iou_array == -4)[0]) + len(
                np.where(iou_array == 9)[0]) + intersection

            # calculate Jaccard
            IoU = intersection/union

            # assign score
            if IoU > threshold:
                return 'tp'
            else:
                return 'fp'
        # if you have not provided an index
        if not lsif_roi.GetLabels():
            return 'fn'


# main part of the code
def main(path_to_rois, path_to_anns, save_results):
    # get list of roi files
    rois = [file for file in os.listdir(path_to_rois) if 'roi' in file]
    # subjects are in format sub-test*
    subs = [file.split('_')[0] for file in rois]

    # make an array of thresholds for ROC curve
    thresholds = np.arange(10, 100, 5)
    # make an empty DataFrame
    score_df = pd.DataFrame(
        index=subs,
        columns=[str(thresh) + '%' for thresh in thresholds])

    for roi, sub in zip(rois, subs):
        # get full paths to images
        path_to_roi = os.path.join(path_to_rois, roi)
        path_to_ann = os.path.join(path_to_anns, sub + '_seg.nii.gz')

        for thresh in thresholds:
            score = get_score(path_to_roi, path_to_ann, sub, thresh/10)
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

    for col in score_df:
        series = score_df[col]
        # get TPR and TPR for each column
        counts = series.value_counts()
        tpr = counts['tp'] / (counts['tp'] + counts['fn'])
        fpr = counts['fp'] / (counts['fp'] + counts['tn'])
        tpr_row.loc['tpr', col] = round(tpr, 3)
        fpr_row.loc['fpr', col] = round(fpr, 3)

    # append FPR and TPR for the bottom of the score card
    score_df = score_df.append(tpr_row)
    score_df = score_df.append(fpr_row)

    # make the ROC curve
    def create_triangle(tpr_0, tpr_1, fpr_0, fpr_1):
        plt.plot([tpr_0, tpr_1], [fpr_0, fpr_1], '-', lw=2, color='#4285F4')
        plt.plot([tpr_0, tpr_1], [fpr_1, fpr_1], '-', lw=2, color='#4285F4')
        plt.plot([tpr_0, tpr_0], [fpr_0, fpr_1], '-', lw=2, color='#4285F4')

    def create_rectangle(tpr_0, fpr_0):
        plt.plot([tpr_0, tpr_0], [fpr_0, 0], '-', lw=2, color='#4285F4')

    # create ROC image
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

    # save figure and DataFrame
    plt.savefig(os.path.join(save_results, "roc_curve.png"), bbox_inches='tight')
    score_df.to_csv(os.path.join(save_results, 'scores.csv'))

    # construct the area under the ROC curve
    rectangle_auc = 0
    for k in range(len(thresholds)-1):
        rectangle_auc += (fpr_row.iloc[0, k + 1] - fpr_row.iloc[0, k]) * tpr_row.iloc[0, k]

    print('Rectangular AUC = {}'.format(rectangle_auc))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Insufficient arguments.')
