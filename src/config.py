#!/usr/bine/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2024 Anoduck'
# This software is released under the MIT License.
# https://anoduck.mit-license.org/
from simple_parsing import choice
from simple_parsing.helpers import list_field
from dataclasses import dataclass
from typing import List
import os


@dataclass
class Options:
    """
    Car detection, identification, and image creation estimations
    ---------------------------------------------------------------
    Usage: poetry run python Obj_Detect/__main__.py [--action <action>] [--inpath <inpath>] [--outpath <outpath>] [--sampledir <sampledir>] [--logfile <logfile>] [--loglevel <loglevel>]

    That's all, yo.

    You have three actions:
    - detect: Detect cars in images.
    - identify: Identify specific cars from sample images.
    - sundial: Estimate time of image creation based on the sun position.

    """

    action: str = choice(
        "detect", "identify", "sundial", default="identify"
    )  # Action to perform
    inpath: List[str] = list_field(
        os.path.expanduser("~/Pictures")
    )  # Dir path to process
    outpath: str = "../results"  # Path to results.
    sampledir: str = "../samples"  # Path for training data
    logfile: str = "../detect.log"  # Path to logfile
    loglevel: str = "DEBUG"  # Logging level
