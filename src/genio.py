#!/usr/bine/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2024 Anoduck'
# This software is released under the MIT License.
# https://anoduck.mit-license.org/
from simple_parsing import parse
from pathlib import Path
import os
import sys
from config import Options as opt
from logio import LogIO


class GenIO:
    def __init__(self):
        super().__init__()
        self.options = opt
        self.Options = parse(self.options, dest="Options")
        self.in_dirs = self.Options.inpath
        self.out_dir = self.Options.outpath
        self.log_file = self.Options.logfile
        self.log_level = self.Options.loglevel
        self.logio = LogIO(self.log_file, self.log_level)
        self.log = self.logio.get_log()

    def gen_io(self):
        self.log.info("Generating input list...")
        plist = []
        self.log.info("Output Path: {}".format(self.out_dir))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.log.info("Input Path: {}".format(self.in_dirs))
        # Remember in_dirs is a list of strings that represent directories.
        # So, they have to parsed individually.
        for item in self.in_dirs:
            self.log.info("Processing Item: {}".format(item))
            if not os.path.isdir(item):
                print("Not a directory: {}".format(item))
                sys.exit(1)
            item_abspath = os.path.abspath(item)
            self.log.info("Processing Directory: {}".format(item_abspath))
            # Globs are not Regular Expressions!!!
            for path in Path(item_abspath).rglob(
                "*.jpg" or "*.JPG" or "*.png" or "*.PNG"
            ):
                self.log.debug("Path: {}".format(path))
                if path not in plist:
                    plist.append(path)
        self.log.info("Image List: {}".format(plist))
        plist.sort(key=os.path.getctime)
        self.log.info("Sorted Image List: {}".format(plist))
        return plist
