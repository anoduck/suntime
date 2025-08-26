# from ast import main
import os
import cv2 as cv
import numpy as np
from simple_parsing import parse
from alive_progress import alive_it

from logio import LogIO
from config import Options
from genio import GenIO
from sundial import Sundial


class Main:
    def __init__(self):
        self.Options = parse(Options, dest="Options")
        self.action = self.Options.action
        self.inpath = self.Options.inpath
        self.outpath = self.Options.outpath
        self.tdir = self.Options.sampledir
        self.log_file = self.Options.logfile
        self.log_level = self.Options.loglevel
        self.logio = LogIO(self.log_file, self.log_level)
        self.log = self.logio.get_log()
        self.genio = GenIO()
        self.file_list = self.genio.gen_io()
        self.cascade = cv.CascadeClassifier(
            cv.data.haarcascades + "./haarcascade_fullbody.xml"
        )
        self.min_count = 10

    def identify(self):
        for imgfile in alive_it(self.file_list):
            qimage = os.path.realpath(imgfile)
            img1 = cv.imread(qimage, cv.IMREAD_GRAYSCALE)  # queryImage
            for timagef in os.listdir(self.tdir):
                img2 = cv.imread(timagef, cv.IMREAD_GRAYSCALE)  # trainImage
                sift = cv.SIFT_create()
                kp1, des1 = sift.detectAndCompute(img1, None)
                kp2, des2 = sift.detectAndCompute(img2, None)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                if len(good) > self.min_count:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(
                        -1, 1, 2
                    )
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(
                        -1, 1, 2
                    )

                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()

                    h, w = img1.shape
                    pts = np.float32(
                        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                    ).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)

                    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

                else:
                    print(
                        "Not enough matches - {}/{}".format(len(good), self.min_count)
                    )
                    matchesMask = None
                draw_params = dict(
                    matchColor=(0, 255, 0),  # draw matches in green color
                    singlePointColor=None,
                    matchesMask=matchesMask,  # draw only inliers
                    flags=2,
                )

                img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

                cv.imwrite(self.outpath + "/" + imgfile.split("/")[-1], img3)

    def detect(self):
        for imgfile in alive_it(self.file_list):
            image = os.path.realpath(imgfile)
            print(f"type for {image}: is {type(image)}")
            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                img2 = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.imwrite(self.outpath + "/" + img.split("/")[-1], img2)
        print("Done!")

    def main(self):
        if self.action == "detect":
            self.detect()
        elif self.action == "identify":
            self.identify()
        elif self.action == "sundial":
            sundial = Sundial(self.log)
            sundial.proc_time(self.file_list, self.outpath, self.log)
        else:
            print("Please specify an action to perform")
            print("Options: detect, identify, sundial")


if __name__ == "__main__":
    main = Main()
    main.main()
