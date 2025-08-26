#!/usr/bine/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2024 Anoduck'
# This software is released under the MIT License.
# https://anoduck.mit-license.org

import cv2 as cv
import numpy as np


class Horizon:
    def __init__(self) -> None:
        self.ksize = 5
        self.lowt = 100
        self.hight = 200

    def region_selection(self, image):
        """
        Determine and cut the region of interest in the input image.
        Parameters:
            image: we pass here the output from canny where we have
            identified edges in the frame
        """
        # create an array of the same size as of the input image
        mask = np.zeros_like(image)
        # if you pass an image with more then one channel
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        # our image only has one channel so it will go under "else"
        else:
            # color of the mask polygon (white)
            ignore_mask_color = 255
        # creating a polygon to focus only on the road in the picture
        # we have created this polygon in accordance to how the camera was placed
        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array(
            [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32
        )
        # filling the polygon with white color and generating the final mask
        cv.fillPoly(mask, vertices, ignore_mask_color)
        # performing Bitwise AND on the input image and mask to get only the edges on the road
        masked_image = cv.bitwise_and(image, mask)
        return masked_image

    def hough_transform(self, image):
        """
        Determine and cut the region of interest in the input image.
        Parameter:
            image: grayscale image which should be an output from the edge detector
        """
        # Distance resolution of the accumulator in pixels.
        rho = 1
        # Angle resolution of the accumulator in radians.
        theta = np.pi / 180
        # Only lines that are greater than threshold will be returned.
        threshold = 20
        # Line segments shorter than that are rejected.
        minLineLength = 20
        # Maximum allowed gap between points on the same line to link them
        maxLineGap = 500
        # function returns an array containing dimensions of straight lines
        # appearing in the input image
        return cv.HoughLinesP(
            image,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap,
        )

    def average_slope_intercept(self, lines):
        """
        Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: output from Hough Transform
        """
        left_lines = []  # (slope, intercept)
        left_weights = []  # (length,)
        right_lines = []  # (slope, intercept)
        right_weights = []  # (length,)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                # calculating slope of a line
                slope = (y2 - y1) / (x2 - x1)
                # calculating intercept of a line
                intercept = y1 - (slope * x1)
                # calculating length of a line
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                # slope of left lane is negative and for right lane slope is positive
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        #
        left_lane = (
            np.dot(left_weights, left_lines) / np.sum(left_weights)
            if len(left_weights) > 0
            else None
        )
        right_lane = (
            np.dot(right_weights, right_lines) / np.sum(right_weights)
            if len(right_weights) > 0
            else None
        )
        return left_lane, right_lane

    def pixel_points(self, y1, y2, line):
        """
        Converts the slope and intercept of each line into pixel points.
            Parameters:
                y1: y-value of the line's starting point.
                y2: y-value of the line's end point.
                line: The slope and intercept of the line.
        """
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))

    def lane_lines(self, image, lines):
        """
        Create full lenght lines from pixel points.
            Parameters:
                image: The input test image.
                lines: The output lines from Hough Transform.
        """
        left_lane, right_lane = self.average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = self.pixel_points(y1, y2, left_lane)
        right_line = self.pixel_points(y1, y2, right_lane)
        return (left_line, right_line)

    def draw_lane_lines(self, image, horizon, color=[255, 0, 0], thickness=12):
        """
        Draw lines onto the input image.
            Parameters:
                image: The input test image (video frame in our case).
                lines: The output lines from Hough Transform.
                color (Default = red): Line color.
                thickness (Default = 12): Line thickness.
        """
        line_image = np.zeros_like(image)
        for i in range(0, len(horizon)):
            line = horizon[i]
            if line is not None:
                cv.line(line_image, *line, color, thickness)
                text_label = str(f"Horizon Line #{i}")
                cv.putText(
                    image,
                    text_label,
                    (10, 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    2,
                )
        return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    def frame_processor(self, image, orientation):
        """
        Process the input frame to detect lane lines.
        Parameters:
            image: image of a road where one wants to detect lane lines
            (we will be passing frames of video to this function)
        """
        # convert the RGB image to Gray scale
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # applying gaussian Blur which removes noise from the image
        # and focuses on our region of interest
        # size of gaussian kernel
        kernel_size = 5
        # Applying gaussian blur to remove noise from the frames
        blur = cv.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
        # first threshold for the hysteresis procedure
        low_t = 50
        # second threshold for the hysteresis procedure
        high_t = 150
        # applying canny edge detection and save edges in a variable
        edges = cv.Canny(blur, low_t, high_t)
        # since we are getting too many edges from our image, we apply
        # a mask polygon to only focus on the road
        # Will explain Region selection in detail in further steps
        region = self.region_selection(edges)
        # Applying hough transform to get straight lines from our image
        # and find the lane lines
        # Will explain Hough Transform in detail in further steps
        hough = self.hough_transform(region)
        # Now here is where we deviate from the original script
        # We only need one line drawn, and only one line is guessed correctly.
        horizon = self.lane_lines(image, hough)
        # lastly we draw the lines on our resulting frame and return it as output
        result = self.draw_lane_lines(image, horizon)
        return result, horizon
