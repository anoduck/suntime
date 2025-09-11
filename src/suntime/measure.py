#!/usr/bin/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2025 Anoduck'
# This software is released under the MIT License.
# https:anoduck.mit-license.org
# ---------------------------------------------------------
import cv2
import numpy as np


class ObjMeasure:
    def __init__(self, imgobj):
        self.img = imgobj

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to
        # separate the objects from the background
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find the contours of the objects in the image
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Loop through the contours and calculate the area of each object
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Draw a bounding box around each
            # object and display the area on the image
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                self.img, str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        return self.img
