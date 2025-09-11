#!/usr/bine/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2024 Anoduck'
# This software is released under the MIT License.
# https://anoduck.mit-license.org
import math
import cv2 as cv
import numpy as np
from .image_utils import Image


class LocSun:
    def analemma(self, image) -> tuple:
        radius = int(51)
        last_center = (0.0, 0.0)
        # reduce image size by 20px on all sides and auto converts it to gray
        (h, w) = image.shape[:2]
        h2 = h - 20
        w2 = w - 20
        image_copy = cv.resize(image, (w2, h2))
        image = Image.size_reduction(image, 20)
        print(f"Image copy shape: {image_copy.shape} and image shape: {image.shape}")
        # insure radius is odd
        print(f"Radius: {radius}")
        if int(radius) % 2:
            pass
        else:
            radius += 1
        # blur the image
        try:
            blur = cv.GaussianBlur(image, (int(radius), int(radius)), cv.BORDER_DEFAULT)
        except Exception:
            blur = cv.medianBlur(image, int(radius))

        # calculate minMax method
        minMaxMethod = image.copy()
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(blur)
        minMaxCenter = maxLoc
        # apply the minMax method
        cv.circle(minMaxMethod, maxLoc, int(radius), (0, 0, 0), 2)
        # prepare the image for the robust method
        thresh = cv.threshold(blur, 210, 225, cv.THRESH_BINARY)[1]
        erode = cv.erode(thresh, None, iterations=7)
        dilate = cv.dilate(erode, None, iterations=4)
        canny = Image.get_formated_canny(dilate)
        # calculate robust method
        points = np.argwhere(canny > 0)
        robustCenter, radius = cv.minEnclosingCircle(points)
        # apply robust method
        robustMethod = image.copy()
        x = int(robustCenter[1])
        y = int(robustCenter[0])
        rad = int(radius)
        cv.circle(robustMethod, (x, y), rad, (300, 100, 100), 2)
        # debug print
        print("lastCenter: " + str(last_center))
        print("dist: " + str(math.dist(robustCenter, last_center)))
        print("robustCenter: " + str(robustCenter))
        print("maxLoc: " + str(robustCenter))
        print("----------------------------------")
        # determine which method to use
        center = minMaxCenter
        if robustCenter == (0.0, 0.0):
            if last_center != (0.0, 0.0):
                if not (math.dist(minMaxCenter, last_center) < 50):
                    center = robustCenter
            else:
                center = (0.0, 0.0)
        # put text and highlight the center
        Cx, Cy = center
        image1 = cv.circle(image_copy, (Cx, Cy), 5, (255, 255, 255), -1)
        image2 = cv.putText(
            image1,
            "centroid",
            (Cx - 25, Cy - 25),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
        )
        print(f"Center: {center}, Radius: {int(radius)}")
        return image2, center
