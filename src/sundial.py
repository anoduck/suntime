#!/usr/bine/env python3
#! vim:set filetype=python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
# MIT License = 'Copyright (c) 2024 Anoduck'
# This software is released under the MIT License.
# https://anoduck.mit-license.org
from dotenv import load_dotenv
from pysolar import solar
import cv2 as cv
import numpy as np
import pandas as pd
import pytesseract as ocr
import os
import math
import datetime
from pytz import timezone
from locsun import LocSun
from detector import Detector


class Sundial:
    def __init__(self, log, prev_date=None):
        load_dotenv()
        self.log = log
        self.latitude = os.getenv("latitude")
        self.longitude = os.getenv("longitude")
        self.timezone = os.getenv("timezone")
        self.orientation = os.getenv("orientation")
        self.est_start_date = os.getenv("est_start_date")
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.prev_date = prev_date
        self.min_hour = 8
        # If you set to 17, then max will be 16
        # 24 hour clock
        self.max_hour = 18
        self.day_variance = 5

    def increment_range(self, year, month, day, log):
        process_times = []
        min_day = int(day) - self.day_variance
        max_day = int(day) + self.day_variance
        zero_min = 0
        log.debug(f"Timezone: {self.timezone}")
        local = timezone(self.timezone)
        log.debug(f"Timezone type: {type(local)}")
        for vday in range(min_day, max_day):
            for hour in range(self.min_hour, self.max_hour):
                dt = datetime.datetime(
                    int(year),
                    int(month),
                    int(vday),
                    int(hour),
                    int(zero_min),
                    tzinfo=local,
                )
                """ log.debug(f'dt for process times: {dt}') """
                process_times.append(dt)
        return process_times

    def get_date(self, image, log):
        imgrgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        text = ocr.image_to_string(imgrgb)
        try:
            date_text = text.rsplit(" ")[-2]
            date_split = date_text.split("/")
        except Exception:  # if rsplit fails, give dummy values to fail.
            date_split = [1, 2]
        if len(date_split) < 3:
            log.debug("OCR returned invalid date: {}".format(text))
            if self.prev_date is not None:
                year, month, day = self.prev_date
            elif self.est_start_date is not None:
                year, month, day = self.est_start_date
            else:
                log.debug("There was an error acquiring a valid date")
                exit(1)
        else:
            year, month, day = date_split
        return year, month, day

    def get_solar_altitude(self, image, log):
        sunpd = pd.DataFrame(columns=["time", "altitude"])
        return_date = self.get_date(image, log)
        year, month, day = return_date
        process_times = self.increment_range(year, month, day, log)
        log.debug(
            f"Type of process_times: {type(process_times)}, length: {len(process_times)}"
        )
        for x in range(0, len(process_times)):
            dt = process_times[x]
            altitude = solar.get_altitude_fast(
                float(self.latitude), float(self.longitude), dt
            )
            sunpd.loc[x] = {"time": dt, "altitude": altitude}
        return sunpd

    def slope(self, p1, p2):
        return (p2[1] - p1[1]) / (p2[0] - p1[0])

    def findangle(self, img, horizon, center, log):
        log.debug("horizon is {}".format(horizon))
        lhrzn, rhrzn = horizon
        log.debug("lhrzn and rhrzn are {} and {}".format(lhrzn, rhrzn))
        b = rhrzn
        a = lhrzn
        c = center
        m1 = self.slope(b, a)
        m2 = self.slope(b, c)
        Calc_angle = (m2 - m1) / (1 + m1 * m2)
        rad_angle = math.atan(Calc_angle)
        degree_angle = round(math.degrees(rad_angle))
        log.debug("degree angle is {}".format(degree_angle))
        if degree_angle < 0:
            degree_angle = 180 + degree_angle
        cv.putText(
            img,
            str(degree_angle),
            (b[0] + 40, b[1] + 40),
            self.font,
            3,
            (0, 255, 255),
            1,
            cv.LINE_AA,
        )
        return img, degree_angle

    def get_closest(self, sunpd, elevation, log):
        lst = sunpd.altitude.to_numpy()
        log.debug("Value of sunpd: {}".format(sunpd))
        idx = (np.abs(lst - elevation)).argmin()
        log.debug(f"Returned elevation: {idx} and type: {type(idx)}")
        idxseries = sunpd.loc[idx, ["time"]]
        log.debug("IDXSeries: {}".format(idxseries))
        closest_val = idxseries.item()
        log.debug(f"Closest value is {closest_val}, and type is {type(closest_val)}")
        return closest_val

    def write_file(self, filename, result, outpath):
        image_name = str(os.path.basename(filename))
        namefront = image_name.split(".")[0]
        fname = namefront.join(["suntime", ".jpg"])
        fpath = os.path.join(outpath, fname)
        cv.imwrite(fpath, result)

    def proc_time(self, file_list, outpath, log):
        for filename in file_list:
            log.debug(f"Processing file: {os.path.basename(filename)}")
            start_time = datetime.datetime.now()
            log.debug(f"Start time: {start_time}")
            log.debug(filename)
            imgpath = os.path.realpath(filename)
            image = cv.imread(imgpath)
            log.debug(image.shape)
            locsun = LocSun()
            image1, center = locsun.analemma(image, log)
            log.debug(f"results from analemma: {center} and {image1.shape}")
            log.debug(image1.shape)
            time_center = datetime.datetime.now() - start_time
            log.debug(f"Time till center: {time_center}")
            # hrzn = Horizon()
            # image2, horizon = hrzn.frame_processor(image1, self.orientation)  # Horizon L -> R
            detector = Detector()
            image2, horizon = detector.main(image1)
            log.debug(f"Horizon coordinates: {horizon}")
            log.debug(f"Time till horizon: {datetime.datetime.now() - start_time}")
            image3, elevation = self.findangle(image2, horizon, center, log)
            log.debug(f"Time till elevation: {datetime.datetime.now() - start_time}")
            sunvalues = self.get_solar_altitude(image, log)
            est_time = self.get_closest(sunvalues, elevation, log)
            res_time = pd.to_datetime(est_time)
            log.debug("Full time date: {}".format(res_time))
            if res_time is not None:
                tstamp = pd.Timestamp(est_time)
                prev_year = tstamp.year
                prev_month = est_time.month
                prev_day = est_time.day
                self.prev_date = prev_year, prev_month, prev_day
                log.debug("Previous Date is now: {}".format(self.prev_date))
                log.debug(f"Time processing: {datetime.datetime.now() - start_time}")
                caption_time = res_time.strftime("%-m/%-d/%Y %H:%M:%S")
                rimg = cv.putText(
                    image3,
                    str(caption_time),
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    2,
                )
                self.write_file(filename, rimg, outpath)
            else:
                print("Script failed to return res_time value")
                exit(1)

        print("Done")
