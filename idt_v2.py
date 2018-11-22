# -*- coding: utf-8 -*-
import os
import csv
import glob
import numpy as np
import cv2
from tqdm import tqdm

# force applied to the indenter [kgf]
force = 50


def pt(lst):
    return tuple(np.int0(lst))

def calcDist(fn, dpi=25.4, val=0):
    """Calculate diagonal distance from image.
    Arguments
        fn: input image file path
        dpi(option): DPI of image
        val(option): threshould for image thresholding
                     val=0 -> Otsu's Method
    Returns
        dist: avarage of diagonal distances
        img: result image
    """
    img = cv2.imread(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = cv2.THRESH_OTSU if val==0 else 0
    val,th = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV + otsu)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    if len(cnts) == 0:
        return False, img

    # detect vertexes of contour
    max_idx = np.argsort([len(cnt) for cnt in cnts])[-1]
    max_cnt = cnts[max_idx]
    max_cnt = np.reshape(max_cnt.flatten(), (-1,2))

    x, y = np.hsplit(max_cnt, 2)
    x, y = x.flatten(), y.flatten()

    minXpt = np.average(max_cnt[np.where(x==np.min(x))], axis=0)
    maxXpt = np.average(max_cnt[np.where(x==np.max(x))], axis=0)
    minYpt = np.average(max_cnt[np.where(y==np.min(y))], axis=0)
    maxYpt = np.average(max_cnt[np.where(y==np.max(y))], axis=0)

    # calculate diagonal distance
    Xdist = np.linalg.norm(maxXpt - minXpt) * 25.4 / dpi
    Ydist = np.linalg.norm(maxYpt - minYpt) * 25.4 / dpi
    dist = np.average([Xdist, Ydist])

    # draw results
    cv2.drawContours(img, cnts, max_idx, (0,255,0), 1)
    cv2.line(img, pt(minXpt), pt(maxXpt), (0,0,255), 1)
    cv2.line(img, pt(minYpt), pt(maxYpt), (255,0,0), 1)
    cv2.putText(img, "%6.1f"%Xdist, pt(maxXpt), ff, 0.6, (0,0,255), 1)
    cv2.putText(img, "%6.1f"%Ydist, pt(minYpt), ff, 0.6, (255,0,0), 1)

    #cv2.imshow("results",img)
    #cv2.waitKey(0)

    return dist, img



ff = cv2.FONT_HERSHEY_SIMPLEX
lst = glob.glob("img\\*")
if len(lst) == 0: exit()
try: os.mkdir("res")
except: pass

csvf = open("results.csv", "w")
writer = csv.writer(csvf, lineterminator="\n")
writer.writerow(["input file", "distance", "HV"])

for fn in tqdm(lst):
    dist, img = calcDist(fn, dpi=600)
    HV = 1.8544 * force / dist**2 if dist else "=NA()"
    path = "res\\%s.png"%os.path.splitext(os.path.basename(fn))[0]
    cv2.imwrite(path, img)
    writer.writerow([fn, dist, HV])

csvf.close()
#cv2.destroyAllWindows()