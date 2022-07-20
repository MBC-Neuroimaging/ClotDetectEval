#! /usr/bin/env python3

import SimpleITK as sitk
import os
import sys

'''
Gets an empty segmentation file for patient with no lesion
'''

raw_file = sys.argv[1]
out_dir = sys.argv[2]
basename = os.path.basename(raw_file)
subject = basename.split('_')[0] + '_' + basename.split('_')[1]
im = sitk.ReadImage(raw_file)

size = im.GetSize()

seg_im = sitk.Image(size, sitk.sitkInt16)

seg_im.CopyInformation(im)

sitk.WriteImage(seg_im, os.path.join(out_dir, subject + '_seg.nii.gz'))

