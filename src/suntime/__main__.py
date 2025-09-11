#!/usr/bin/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2025 Anoduck'
# This software is released under the MIT License.
# https:anoduck.mit-license.org
# ---------------------------------------------------
from pathlib import Path
from locsun import LocSun
from shadow import Shadow
from measure import ObjMeasure


class SunTime:
    def __init__(
        self, image_path, lat, lon, tz, orientation, start_year, start_month
    ) -> None:
        self.image_path = Path(image_path)
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.orientation = orientation
        self.start_year = start_year
        self.start_month = start_month

    def suntime(self):
        locsun = LocSun()
        sunid_image = locsun.analemma(self.image_path)
        sdw = Shadow(sunid_image)
        contoured = sdw.process_image()
