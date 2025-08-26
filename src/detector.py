#######################################
######_SKY-GROUND DETECTOR USING_#####
######_OPENCV IN PYTHON_##############
######################################

import cv2
import numpy as np
from scipy import ndimage
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt


class Detector:
    def __init__(self):
        self.scale = 5.0

    # Downsampling initial_images
    def block_mean(self, ar, fact):
        assert isinstance(fact, int), type(fact)
        sx, sy = ar.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        regions = sy / fact * (X / fact) + Y / fact
        image = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
        image.shape = (sx / fact, sy / fact)
        return image

    # function for calculating pixel data and 3d manipulations for input_image
    def get_pixel_dimensions(self, image):
        # loading images from initial_path
        assert isinstance(image, type(None)), "image not found"
        image = self.block_mean(image, self.scale)
        # modifying sizes of the new downsampled images
        [xSize, ySize] = image.size
        _r = []
        _g = []
        _b = []
        colours = []
        for x in range(0, xSize):
            for y in range(0, ySize):
                # dividing each channel with 255 to change the range of color from 0..255 to 0..1
                [r, g, b] = image[x, y]
                r = r / 255.0
                _r.append(r)
                g = g / 255.0
                _g.append(g)
                b = b / 255.0
                _b.append(b)
                colours.append([r, g, b])
        # return colors.rgb([1.0 * x / 255 for x in rgb_tuple])
        # Three-dimensional plotting using Matplotlib
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.scatter(_r, _g, _b, c=colours, lw=0)
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        fig.add_axes(ax)
        plt.show()
        return

    # function calculating the slope-intercept parameterization
    def straight_line(self, m, b, x, y):
        """
        :param m: denotes the slope of the straight_line
        :param b: denotes the y -inter-cept
        :param x: variable describing a specific point
        :param y: variable describing a specific point
        :return:  returns the slope and intercept values
        """
        return y - m * x - b

    # setting the range of slope and intercept values
    def boundary_detect(self, img2, xSize, ySize):
        zsize = int(50.0)
        slope = np.linspace(float(-1), float(1), zsize)
        inter = np.linspace(float(0), float(ySize), zsize)
        # initializing variables
        maximum = []
        J2 = 0
        # iterating through both the slope and y-intercept
        for m in range(len(slope)):
            for b in range(len(inter)):
                # initializing both sky and ground as an array of pixel values
                sk = []
                gn = []
                # iterate over all the pixels in the image and add them to sky and ground
                for i in range(xSize):
                    for j in range(ySize):
                        # from optimization criterion technique;
                        # J1 = 1/|Σs| + |Σg|
                        # J2 = 1/|Σs| + |Σg| + ( λ1**s + λ2**s + λ3**s )**2 + ( λ1**g + λ2**g + λ3**g )**2
                        # cross product finding every pixel value above and below the straight_line
                        if (
                            self.straight_line(slope[m], inter[b], i, j)
                            * (-1 * inter[b])
                        ) > 0:
                            sk.append(img2[j, i])
                        else:
                            gn.append(img2[j, i])
                # determining covariance of both sky and ground
                sk = np.transpose(sk)
                gn = np.transpose(gn)
                try:
                    cov_s = np.cov(sk)  # calculate covariance of sky
                    cov_g = np.cov(gn)  # calculate covariance of ground
                    covS = np.linalg.det(cov_s)  # calculating determinant of sky
                    covG = np.linalg.det(cov_g)  # calculating dererminant of ground
                    eig_vs, _ = np.linalg.eig(cov_s)  # getting eigenvalues of sky
                    eig_vg, _ = np.linalg.eig(cov_g)  # getting eigenvalues of ground
                    J = 1 / (
                        covS
                        + covG
                        + (eig_vs[0] + eig_vs[1] + eig_vs[2]) ** 2
                        + (eig_vg[0] + eig_vg[1] + eig_vg[2]) ** 2
                    )
                    # get max value of J for all slopes and intercepts
                    if J > J2:
                        J2 = J
                        maximum = [slope[m], inter[b]]
                        print(maximum)
                except Exception:
                    pass
        return maximum

    # Plots the straight_line coming out of a Hough Line Transform
    def get_line(self, img2, horizon):
        # getting width of initial input_image
        xSize = img2.shape[1]
        print("xSize", xSize)
        m = horizon[0]
        b = horizon[1]
        # modify the slope
        y2 = int(m * (xSize - 1) + b)
        # drawing line across the initial input_image at point of horizon;
        # this will be a blue line, with thickness of the line = 4
        img3 = cv2.line(img2, (0, int(b)), (xSize - 1, y2), (255, 10, 10), 4)
        return img3

    def main(self, image):
        # except(RuntimeError, TypeError, NameError):
        # Initial Dimensions of input_image
        ySize = image.shape[0]
        xSize = image.shape[1]
        # Resizing image by changing its dimension
        # where:
        # fx = scale factor along the horizontal axis
        # fy = scale factor along the vertical axis
        img2 = cv2.resize(image, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
        # Dimensions of image after Downsampling/resizing
        ySize = img2.shape[0]
        xSize = img2.shape[1]
        # Detecting the horizon/sky segmentation
        horizon = []
        horizon = self.boundary_detect(img2, xSize, ySize)
        # get scale factor and line inputs to draw the required blue line across
        horizon[1] *= self.scale
        rimage = self.get_line(image, horizon)
        return rimage, horizon
