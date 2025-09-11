#!/usr/bin/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2025 Anoduck'
# This software is released under the MIT License.
# https:anoduck.mit-license.org
import cv2
import numpy as np


class HSVRun:
    def __init__(self):
        pass

    def midpoint(self, ptA, ptB):
        """Helper to compute midpoint between two points."""
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def detect_shape(self, contour):
        """Detect simple shapes based on approximated contour."""
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"  # or ellipse
        return shape

    def discover_shadows(self, cv_image):
        # Optional: Normalize to reduce uneven lighting/shadows (division normalization)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (95, 95), 0)
        normalized = cv2.divide(gray, blur, scale=255)

        # Convert to HSV for thresholding
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)
        # Merge normalized grayscale as the Value channel
        hsv_normalized = cv2.merge([h, s, normalized])
        # Convert back to BGR for visualization (optional)
        image_normalized = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

        # Threshold for shadows: Low Value (adjust thresholds based on image; e.g., V < 100 for dark areas)
        shadow_lower = np.array([0, 0, 0])
        shadow_upper = np.array([180, 255, 100])  # Low V for shadows
        shadow_mask = cv2.inRange(hsv_normalized, shadow_lower, shadow_upper)

        # Threshold for objects: Higher Value (brighter regions)
        object_lower = np.array([0, 0, 150])  # High V for non-shadows
        object_upper = np.array([180, 255, 255])
        object_mask = cv2.inRange(hsv_normalized, object_lower, object_upper)

        # Ensure shadow areas are closed: Apply morphological closing to fill gaps/holes
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
        shadow_mask = cv2.morphologyEx(
            shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # Find contours for shadows
        shadow_contours, _ = cv2.findContours(
            shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find contours for objects
        object_contours, _ = cv2.findContours(
            object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process shadows (outline, label, identify object, measure)
        output = image_normalized.copy()  # Use normalized image for output
        min_area = 500  # Ignore small contours; adjust as needed
        max_area = (
            cv_image.shape[0] * cv_image.shape[1] / 2
        )  # Ignore shadows larger than half the image area; adjust as needed
        object_centers = []  # Store object centers and shapes

        # Collect object info
        for oc in object_contours:
            if cv2.contourArea(oc) > min_area:
                M = cv2.moments(oc)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shape = self.detect_shape(oc)
                    object_centers.append(((cx, cy), shape))

        # Process each shadow contour
        for sc in shadow_contours:
            area = cv2.contourArea(sc)
            if min_area < area < max_area:  # Now checks both min and max area
                # Outline the shadow shape
                cv2.drawContours(output, [sc], -1, (0, 255, 0), 2)  # Green outline

                # Label as "Shadow"
                M = cv2.moments(sc)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        output,
                        "Shadow",
                        (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

                    # Identify corresponding object: Find nearest object by Euclidean distance
                    if object_centers:
                        distances = [
                            np.linalg.norm(np.array((cx, cy)) - np.array(oc[0]))
                            for oc in object_centers
                        ]
                        nearest_idx = np.argmin(distances)
                        object_shape = object_centers[nearest_idx][1]
                        cv2.putText(
                            output,
                            f"From: {object_shape}",
                            (cx - 20, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                    # Measure shadow length: Fit ellipse and get major axis (or use bounding box for simplicity)
                    if len(sc) >= 5:  # Ellipse needs at least 5 points
                        ellipse = cv2.fitEllipse(sc)
                        major_axis = max(ellipse[1])  # Longer dimension
                    else:
                        x, y, w, h = cv2.boundingRect(sc)
                        major_axis = max(w, h)
                    cv2.putText(
                        output,
                        f"Len: {major_axis:.1f}px",
                        (cx - 20, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        return output
