import sys
import glob
import os
import math
import numpy as np
import cv2
from cv2 import aruco


class GeometricCalibration:
    def __init__(self, cam, projector, checker_pattern=(23, 7), checker_size=0.01, checker_pixels=60,
                 width_mirror=0.3556, height_mirror=0.254, marker_offset=0.005, dict_aruco=aruco.DICT_6X6_250,
                 aruco_size=0.04, imgPattern='*.PNG'):
        # For the extrinsic/geometric calibration you are required to print 8 Aruco markers
        # Checkerboard: CalibrationImages/6x6_40mm.pdf
        # Glue or mount them on the edges of you mirror as well as halfway between the edges
        # The Aruco markers will determine the position of the object relative to the camera
        # To determine the position of the screen, we display a checkerboard pattern on the screen
        # Make sure that the screen is visible on the mirror from the position of the camera
        # Camera:
        self.cam = cam
        # Projector:
        self.proj = projector
        # Checkerboard:
        # Tuple of checkers in width and height
        self.checker_pattern = checker_pattern
        # Checker file:
        self.checker_file = '/CalibrationNumpyData/8_24_checker.npz'
        # Display size of checker
        self.checker_size = checker_size
        # Pixel size
        self.checker_pixels = checker_pixels
        # Per pixel Size
        self.checker_pixel_size = checker_size/checker_pixels
        # Display dimensions:
        self.display_width, self.display_height = self.proj.getResolution()
        # Aruco marker dimension:
        self.aruco_size = aruco_size
        self.aruco_dict = aruco.Dictionary_get(dict_aruco)
        # Mirror/Object dimensions: height and length of mirror
        # Subtract the white marker offset on a marker patch
        self.board_width = width_mirror
        self.board_height = height_mirror
        self.half_width = (width_mirror/2) - marker_offset
        self.half_height = (height_mirror/2) - marker_offset
        # TODO: paths
        # File pattern of images
        self.imgPattern = imgPattern

    def readFileList(self, imgFolder):
        imgFileList = glob.glob(os.path.join(imgFolder, self.imgPattern))
        imgFileList.sort()
        return imgFileList

    def detectChecker(self, img, debug=True):

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            gray = img

        ret, corners = cv2.findChessboardCorners(gray, self.checker_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE)
        corners_refine = corners

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refine = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

            if debug:
                cv2.drawChessboardCorners(img, self.checker_size, corners_refine, ret)
                cv2.namedWindow('Checker', cv2.WINDOW_NORMAL)
                cv2.imshow('Checker', img)
                cv2.waitKey(0)  # any key
                cv2.destroyWindow('Checker')

        return ret, corners_refine

    @staticmethod
    def readCheckerObjPoint(fname):
        data = np.load(fname)
        objp = data["objp"]
        return objp

    def arucoBoard(self):
        # Aruco marker length, board height, board width
        m = self.aruco_size
        h = self.board_height
        w = self.board_width
        # create objPoints for calibration target
        h0 = (0 - h / 2)
        hm = (m - h / 2)
        h1 = (((h - m) / 2) - h / 2)
        h2 = (((h + m) / 2) - h / 2)
        h3 = ((h - m) - h / 2)
        h4 = (h - h / 2)
        w0 = (0 - w / 2)
        wm = (m - w / 2)
        w1 = (((w - m) / 2) - w / 2)
        w2 = (((w + m) / 2) - w / 2)
        w3 = ((w - m) - w / 2)
        w4 = (w - w / 2)

        objPoints = []
        objPoints.append(np.array([[w0, h0, 0], [wm, h0, 0], [wm, hm, 0], [w0, hm, 0]], dtype=np.float32))  # 0
        objPoints.append(np.array([[w0, h1, 0], [wm, h1, 0], [wm, h2, 0], [w0, h2, 0]], dtype=np.float32))  # 1
        objPoints.append(np.array([[w0, h3, 0], [wm, h3, 0], [wm, h4, 0], [w0, h4, 0]], dtype=np.float32))  # 2
        objPoints.append(np.array([[w1, h3, 0], [w2, h3, 0], [w2, h4, 0], [w1, h4, 0]], dtype=np.float32))  # 3
        objPoints.append(np.array([[w3, h3, 0], [w4, h3, 0], [w4, h4, 0], [w3, h4, 0]], dtype=np.float32))  # 4
        objPoints.append(np.array([[w3, h1, 0], [w4, h1, 0], [w4, h2, 0], [w3, h2, 0]], dtype=np.float32))  # 5
        objPoints.append(np.array([[w3, h0, 0], [w4, h0, 0], [w4, hm, 0], [w3, hm, 0]], dtype=np.float32))  # 6
        objPoints.append(np.array([[w1, h0, 0], [w2, h0, 0], [w2, hm, 0], [w1, hm, 0]], dtype=np.float32))  # 7

        ids = np.linspace(0, 7, 8).astype(np.int32)[:, None]
        arucoCornerBoard = aruco.Board_create(objPoints, self.aruco_dict, ids)

        return arucoCornerBoard, objPoints

    def detectAruco(self, img, debug=True):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            gray = img

        parameters = aruco.DetectorParameters_create()
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)

        if debug:
            frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
            cv2.namedWindow('Aruco', cv2.WINDOW_NORMAL)
            cv2.imshow('Aruco', frame_markers)
            cv2.waitKey(0)  # any key
            cv2.destroyWindow('Aruco')

        return corners, ids

    def postEst(self, corners, ids, camMat, distCoeffs):
        arucoCornerBoard, _ = self.arucoBoard()
        retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, arucoCornerBoard, camMat, distCoeffs, None, None)
        return rvec, tvec

    def reProjAruco(self, img, camMat, distCoeffs, rvec, tvec, cornersAruco):
        print("reProjAruco")
        _, objPoints = self.arucoBoard()  # yunhao

        ids = np.linspace(0, 7, 8).astype(np.int32)[:, None]
        corners_reproj = []
        for i in range(len(objPoints)):
            imgPoints, _ = cv2.projectPoints(np.array(objPoints[i]), rvec, tvec, camMat, distCoeffs)
            corners_reproj.append(imgPoints)

        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners_reproj, ids)
        # TODO: change path
        cv2.imwrite("./reproejct_markers.png", frame_markers)
        cv2.namedWindow('Reproject', cv2.WINDOW_NORMAL)
        cv2.imshow('Reproject', frame_markers)
        cv2.waitKey(0)  # any key
        cv2.destroyWindow('Reproject')

    @staticmethod
    def householderTransform(n, d):
        I3 = np.identity(3, dtype=np.float32)
        e = np.array([0, 0, 1])
        p1 = I3 - 2 * np.outer(n, n)
        p2 = I3 - 2 * np.outer(e, e)
        p3 = 2 * d * n
        return p1, p2, p3

    @staticmethod
    def invTransformation(R, t):
        Rinv = R.T
        Tinv = -(Rinv @ t)
        return Rinv, Tinv

    def calib(self, imgPath, cameraCalibPath):
        # imgFileList = readFileList(imgFolder)

        data = np.load(cameraCalibPath)
        camMtx = data["mtx"]
        dist = data["dist"]

        objP_pixel = np.ceil(self.readCheckerObjPoint(self.checker_file))
        objP_pixel[:, 2] = 0
        objP = np.array(objP_pixel)
        for i in range(self.checker_pattern[1]):
            for j in range(math.floor(self.checker_pattern[0] / 2)):
                tmp = objP[self.checker_pattern[0] * i + j, 0]
                objP[self.checker_pattern[0] * i + j, 0] = objP[
                    self.checker_pattern[0] * i + self.checker_pattern[0] - j - 1, 0]
                objP[self.checker_pattern[0] * i + self.checker_pattern[0] - j - 1, 0] = tmp
        objP[:, 0] -= (self.display_width / 2 - 1)
        objP[:, 1] -= (self.display_height / 2 - 1)

        objP *= self.checker_size

        rtA = []
        rB = []
        tB = []
        rC2Ss = []
        tC2Ss = []

        # define valid image
        validImg = -1
        # for i in trange(len(imgFileList), desc="Images"):
        for i in range(1):

            img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

            # Yunhao
            # img = (img/65535*255).astype(np.uint8)

            # Aruco marker for Mirror position
            cornersAruco, ids = self.detectAruco(img, debug=False)
            if cornersAruco is None and ids is None and len(cornersAruco) <= 3:
                continue

            # Checker for Display
            ret, cornersChecker = self.detectChecker(img, debug=False)
            if not ret:
                print("no Checker!!!")
                continue

            # for a valid image, aruco and checker must be both detected
            validImg += 1

            # Calibrate Mirror Pose with Aruco

            rvecMirror, tvecMirror = self.postEst(cornersAruco, ids, camMtx, dist)
            img_axis = aruco.drawAxis(img, camMtx, dist, rvecMirror, tvecMirror, self.aruco_size)
            cv2.namedWindow('Img_axis', cv2.WINDOW_NORMAL)
            cv2.imshow('Img_axis', img_axis)
            cv2.waitKey(0)  # any key
            cv2.destroyWindow('Img_axis')

            ## Reproejct Camera Extrinsic
            self.reProjAruco(img, camMtx, dist, rvecMirror, tvecMirror, cornersAruco)

            rMatMirror, _ = cv2.Rodrigues(rvecMirror)  # rotation vector to rotation matrix
            normalMirror = rMatMirror[:, 2]

            rC2W, tC2W = self.invTransformation(rMatMirror, tvecMirror)
            dW2C = abs(np.dot(normalMirror, tvecMirror))

            # Householder transformation
            p1, p2, p3 = self.householderTransform(normalMirror, dW2C)

            # Calibrate virtual to Camera with Checker
            rpe, rvecVirtual, tvecVirtual = cv2.solvePnP(objP, cornersChecker, camMtx, dist,
                                                         flags=cv2.SOLVEPNP_IPPE)  # cv2.SOLVEPNP_IPPE for 4 point solution #cv2.SOLVEPNP_ITERATIVE
            # iterationsCount=200, reprojectionError=8.0,

            rvecVirtual, tvecVirtual = cv2.solvePnPRefineLM(objP, cornersChecker, camMtx, dist, rvecVirtual,
                                                            tvecVirtual)

            proj, jac = cv2.projectPoints(objP, rvecVirtual, tvecVirtual, camMtx, dist)
            img_rep = img

            cv2.drawChessboardCorners(img_rep, self.checker_size, proj, True)
            width = 960
            height = int(img_rep.shape[0] * 960 / img_rep.shape[1])
            smallimg = cv2.resize(img_rep, (width, height))
            cv2.imshow("img_rep", smallimg)
            cv2.waitKey(0)  # any key
            cv2.destroyWindow("img_rep")

            rMatVirtual, _ = cv2.Rodrigues(rvecVirtual)  # rotation vector to rotation matrix

            print(tvecVirtual)
            if validImg == 0:
                rtA = p1
                rB = np.matmul(rMatVirtual, p2)
                tB = np.squeeze(tvecVirtual) + p3
            else:
                rtA = np.concatenate((rtA, p1))
                rB = np.concatenate((rB, np.matmul(rMatVirtual, p2)))
                tB = np.concatenate((tB, np.squeeze(tvecVirtual) + p3))

            rS2C = p1 @ rMatVirtual
            tS2C = p1 @ np.squeeze(tvecVirtual) + p3

            rC2S, tC2S = self.invTransformation(rS2C, tS2C)
            print("rC2S:", rC2S)
            print("tC2S:", tC2S)
            rC2Ss.append(rC2S)
            tC2Ss.append(tC2S)

        # rC2Ss = np.array(rC2Ss)
        # tC2Ss = np.array(tC2Ss)
        # fout = os.path.join(imgFolder, "Cam2Screen.npz")
        # np.savez(fout, rC2S=rC2Ss, tC2S=tC2Ss)
        return rC2Ss, tC2Ss