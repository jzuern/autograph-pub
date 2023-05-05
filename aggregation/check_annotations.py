from glob import glob
import cv2
import matplotlib.pyplot as plt

annotation_files = glob('/data/autograph/2402/austin/train/austin-*-masks.png')

frac_values = []

for annotation_file in annotation_files:

    annotation = cv2.imread(annotation_file, 1)
    annotation_successor = annotation[:, :, 2] > 0


    frac_value = annotation_successor.sum() / annotation_successor.size


    frac_values.append(frac_value)

    print(frac_value, annotation_file)

# make histogram of frac_value
plt.hist(frac_values, bins=100)
plt.show()


