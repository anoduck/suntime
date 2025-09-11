#!/usr/bin/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2025 Anoduck'
# This software is released under the MIT License.
# https:anoduck.mit-license.org
# ---------------------------------------------------
import cv2 as cv
import numpy as np
import math


class Angles:
    """Acquire angle of shadow to object"""

    def __init__(self) -> None:
        self.ideal = 90
        pass

    def angle_corr(self, cvimg, angle):
        pass
