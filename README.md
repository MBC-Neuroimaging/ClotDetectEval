# Evaluation on the CODEC-IV dataset
 ## What is CODEC-IV?
 The CODEC-IV dataset consists of 127 annotated vessel-enhanced brain CT images and was created with the purpose of driving an initiative towards excellence in the automated detection of blood clots associated with Acute Ischemic Stroke. Each image was annotated by stroke experts. CODEC-IV was created as part of a special issue on Benchmarks in Artifical Intelligence for Neuroimage journal. 
 ## How to I access CODEC-IV?
Details on the dataset can be found in the [published article](). Our [online form](https://forms.microsoft.com/r/Xt88X58K8p) can be used to request data access and agree to the terms of use. We will be in contact with you after you submit a request.
## Standardised Evaluation protocol
In order to create a Benchmark for clot detection, there must be a standard means to evaluate performance of clot detection algorithms. Algorithms may be developed and then evaluated using the CODEC-IV dataset, which has been divided into training and testing groups, using the protocol supplied here.
### evaluate_seg.py
The ground truth annotation have been made available for the training group and hidden for the testing group. You must supply a prediction for the location of the clot for each of the patient images in the testing set. This must be in the format of a binary region-of-interest (ROI) image, in a compressed NIFTI format (`.nii.gz` extension). ROI files **must in a single location** and be named as:
> <subject_id>_roi.nii.gz

The supplied ROI is compared to the ground truth annotation. This is done by construction a 3D ellipse around the ground truth annotation of size (in voxels) (60, 60, 30) and observings its overlap with the ROI. The amount of overlap determines whether the prediction is a true positive or a false positive. The threshold for overlap is varied to obtain the [receiver-operating characteristic curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

Usage:
```
python3 evalute_seg.py path/to/rois path/to/ground/truth/annotation path/to/save/results > path/to/save/results/evaluation_output.txt
```

All ROIs as well as the output of the evaluate_seg code must be supplied to be included in the leaderboard.
