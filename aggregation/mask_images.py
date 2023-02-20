from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import numpy as np
import cv2


city_names = [
    # "PaloAlto",
    # "Austin",
    # "Detroit",
    # "Washington",
    # "Pittsburgh",
    "Miami",
]



for city_name in city_names:

    df = 10


    print("Processing", city_name)

    print("Processing Drivable")


    drivable = np.asarray(Image.open("/data/lanegraph/woven-data/{}_drivable.png".format(city_name))).astype(np.uint8)

    # downsample file by df
    drivable = cv2.resize(drivable, (drivable.shape[1] // df,
                                     drivable.shape[0] // df),
                          interpolation=cv2.INTER_AREA).astype(np.uint8)

    # binarize file
    drivable[drivable > 0] = 255

    # remove the borders of the image
    bwidth = 1
    drivable[0:bwidth, :] = 0
    drivable[-bwidth:, :] = 0
    drivable[:, 0:bwidth] = 0
    drivable[:, -bwidth:] = 0

    drivable_ = drivable.copy()

    # infill all holes with cv2.floodFill
    mask = np.zeros((drivable.shape[0]+2, drivable.shape[1]+2), np.uint8)
    filled = (255 - cv2.floodFill(drivable, mask, (0, 0), 255)[1]).astype(np.uint8)


    # merge with original image
    filled_final = (filled.copy() + drivable_.copy()).astype(np.uint8)

    # dilate image with circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1000, 1000))
    filled_final_dilated = cv2.dilate(filled_final, kernel, iterations=1)


    # resize back to original size
    filled_final_dilated = cv2.resize(filled_final_dilated, (drivable.shape[1]*df,
                                                             drivable.shape[0]*df),
                                      interpolation=cv2.INTER_AREA).astype(np.uint8)
    filled_final_dilated = filled_final_dilated.astype(bool)

    print("Opening image")
    rgb = np.asarray(Image.open("/data/lanegraph/woven-data/{}.png".format(city_name))).astype(np.uint8)

    # mask with filled_final_dilated
    print("Masking image")

    # do it separately for each channel due to memory constraints
    for c in range(3):
        m = np.zeros(rgb.shape[0:2]).astype(bool)
        m[0:filled_final_dilated.shape[0], 0:filled_final_dilated.shape[1]] = filled_final_dilated
        rgb[:, :, c] = rgb[:, :, c] * m

    print("Saving image")
    # save image
    Image.fromarray(rgb).save("/data/lanegraph/woven-data/{}_masked.png".format(city_name))

    print("Done")


