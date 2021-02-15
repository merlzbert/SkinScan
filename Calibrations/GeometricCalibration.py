import sys
import sys
import glob
import os
import math
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GeometricCalibration:
    def __init__(self, cam, projector, intrinsicCalibFile='CalibrationNumpyData/intrinsic_calibration.npz',
                 imgFile='CalibrationImages/Geometric/geo.PNG', checker_pattern=(23, 7), checker_size=0.01,
                 checker_pixels=60,
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
        self.checker_file = 'CalibrationNumpyData/8_24_checker.npz'
        # Display size of checker
        self.checker_size = checker_size
        # Pixel size
        self.checker_pixels = checker_pixels
        # Per pixel Size
        self.checker_pixel_size = checker_size / checker_pixels
        # Display dimensions:
        self.display_width, self.display_height = self.proj.getResolution()
        # Aruco marker dimension:
        self.aruco_size = aruco_size
        self.aruco_dict = aruco.Dictionary_get(dict_aruco)
        # Mirror/Object dimensions: height and length of mirror
        # Subtract the white marker offset on a marker patch
        self.board_width = width_mirror
        self.board_height = height_mirror
        self.half_width = (width_mirror / 2) - marker_offset
        self.half_height = (height_mirror / 2) - marker_offset
        # TODO: paths
        # File pattern of images
        self.imgPattern = imgPattern
        self.imgFile = imgFile
        self.intrinsicFile = intrinsicCalibFile
        # self.imgUndistortFile = imgUndistortFile

    def readFileList(self, imgFolder):
        # Yunhao Li
        imgFileList = glob.glob(os.path.join(imgFolder, self.imgPattern))
        imgFileList.sort()
        return imgFileList

    def detectChecker(self, img, debug=True):
        # Yunhao
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            gray = img

        ret, corners = cv2.findChessboardCorners(gray, self.checker_pattern,
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
        # Yunhao Li
        data = np.load(fname)
        objp = data["objp"]
        return objp

    def arucoBoard(self):
        # Yunhao Li
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
        # Yunhao Li
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
        # Yunhao Li
        arucoCornerBoard, _ = self.arucoBoard()
        retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, arucoCornerBoard, camMat, distCoeffs, None, None)
        return rvec, tvec

    def reProjAruco(self, img, camMat, distCoeffs, rvec, tvec, cornersAruco):
        # Yunhao Li
        print("reProjAruco")
        _, objPoints = self.arucoBoard()  # yunhao

        ids = np.linspace(0, 7, 8).astype(np.int32)[:, None]
        corners_reproj = []
        for i in range(len(objPoints)):
            imgPoints, _ = cv2.projectPoints(np.array(objPoints[i]), rvec, tvec, camMat, distCoeffs)
            corners_reproj.append(imgPoints)

        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners_reproj, ids)
        cv2.imwrite("./reproejct_markers.png", frame_markers)
        cv2.namedWindow('Reproject', cv2.WINDOW_NORMAL)
        cv2.imshow('Reproject', frame_markers)
        cv2.waitKey(0)  # any key
        cv2.destroyWindow('Reproject')

    @staticmethod
    def householderTransform(n, d):
        # Yunhao Li
        I3 = np.identity(3, dtype=np.float32)
        e = np.array([0, 0, 1])
        p1 = I3 - 2 * np.outer(n, n)
        p2 = I3 - 2 * np.outer(e, e)
        p3 = 2 * d * n
        return p1, p2, p3

    @staticmethod
    def invTransformation(R, t):
        # Yunhao Li
        Rinv = R.T
        Tinv = -(Rinv @ t)
        return Rinv, Tinv

    def calib(self, imgPath, cameraCalibPath):
        # Yunhao Li
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

            cv2.drawChessboardCorners(img_rep, self.checker_pattern, proj, True)
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

    def calibrate(self, imgUndistortFile):
        # Yunhao Li
        # Load intrinsic matrix npz file
        intrinsic_mat = np.load(self.intrinsicFile)

        # Checkerboard:
        CHECKSQUARE_SIZE = self.checker_size
        CHECKSQUARE_PIXELS = self.checker_pixels
        # Display dimensions:
        SCREEN_WIDTH = self.display_width
        SCREEN_HEIGHT = self.display_height

        # half_length = 0.1778 - 0.005
        # half_height = 0.127 - 0.005

        rc2s, tc2s = self.calib(self.imgFile, self.intrinsicFile)
        tc2s = tc2s[0]
        print("new rc2s is ", rc2s)
        print("new tc2s is ", tc2s)
        # rcamera_dis = cam2screen.f.rC2S
        # tcamera_dis = cam2screen.f.tC2S
        rcamera_dis = rc2s
        tcamera_dis = tc2s
        cameraMatrix = intrinsic_mat.f.mtx
        disCoeffs = intrinsic_mat.f.dist
        ret = intrinsic_mat.f.ret
        # additional section: Make sure that the shape of rC2S and tC2S is correct
        rcamera_dis = rcamera_dis[0]
        tcamera_dis = np.reshape(tcamera_dis, (3,))
        print("rc2s is ", rcamera_dis)
        print("tc2s is ", tcamera_dis)
        print("intrinsic matrix is ", cameraMatrix)
        # print(intrinsic_mat.f.dist)
        allCorners = []
        allIds = []
        decimator = 0
        dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        # board =
        img = cv2.imread(self.imgFile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cornerRefine = cv2.aruco.CORNER_REFINE_CONTOUR
        [markerCorners, markerIDs, rejectedImgPoints] = cv2.aruco.detectMarkers(gray, dictionary,
                                                                                cameraMatrix=cameraMatrix,
                                                                                distCoeff=disCoeffs)

        # marker_size = 0.04
        board_corner = np.array([np.array(
            [[-self.half_width, -self.half_height, 0], [-self.half_width + 0.04, -self.half_height, 0],
             [-self.half_width + 0.04, -self.half_height + 0.04, 0], [-self.half_width, -self.half_height + 0.04, 0]]),
                                 np.array([[self.half_width - 0.04, -self.half_height, 0],
                                           [self.half_width, -self.half_height, 0],
                                           [self.half_width, -self.half_height + 0.04, 0],
                                           [self.half_width - 0.04, -self.half_height + 0.04, 0]]),
                                 np.array([[self.half_width - 0.04, self.half_height - 0.04, 0],
                                           [self.half_width, self.half_height - 0.04, 0],
                                           [self.half_width, self.half_height, 0],
                                           [self.half_width - 0.04, self.half_height, 0]]),
                                 np.array([[-self.half_width, self.half_height - 0.04, 0],
                                           [-self.half_width + 0.04, self.half_height - 0.04, 0],
                                           [-self.half_width + 0.04, self.half_height, 0],
                                           [-self.half_width, self.half_height, 0]]),
                                 np.array([[-0.02, -self.half_height, 0], [0.02, -self.half_height, 0],
                                           [0.02, -self.half_height + 0.04, 0], [-0.02, -self.half_height + 0.04, 0]]),
                                 np.array([[-self.half_width, -0.02, 0], [-self.half_width + 0.04, -0.02, 0],
                                           [-self.half_width + 0.04, 0.02, 0], [-self.half_width, 0.02, 0]]),
                                 np.array([[self.half_width - 0.04, -0.02, 0], [self.half_width, -0.02, 0],
                                           [self.half_width, 0.02, 0], [self.half_width - 0.04, 0.02, 0]])],
                                dtype=np.float32)

        board_id = np.array([[0], [6], [4], [2], [7], [1], [5]], dtype=np.int32)
        board = cv2.aruco.Board_create(board_corner, dictionary, board_id)

        if len(markerCorners) > 0:
            allCorners.append(markerCorners)
            allIds.append(markerIDs)
            cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIDs, [0, 200, 0])

        # cv2.aruco.drawPlanarBoard(board, img.shape)
        rvecs, tvecs, _objpoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.04, cameraMatrix, disCoeffs, )
        r_mean = np.mean(rvecs, axis=0)
        t_mean = np.mean(tvecs, axis=0)

        # r_mirror = np.zeros((1,3))
        # t_mirror = np.zeros((1,3))
        r_mirror = None
        t_mirror = None
        ret, r_mirror, t_mirror, = cv2.aruco.estimatePoseBoard(markerCorners, markerIDs, board, cameraMatrix, disCoeffs,
                                                               r_mirror, t_mirror)
        # print("r vec of mirror is ", r_mirror)
        # print("t vec of mirror is ", t_mirror)
        r_mirror_mat = np.zeros((3, 3))
        r_mirror_mat = cv2.Rodrigues(r_mirror, dst=r_mirror_mat, jacobian=None)
        r_mirror_mat = r_mirror_mat[0]
        # print("r mirror mat is ", r_mirror_mat)
        img_axis = aruco.drawAxis(img, cameraMatrix, disCoeffs, r_mirror, t_mirror, 0.04)
        # width = 960
        # height = int(img.shape[0]*960/img.shape[1])
        # smallimg = cv2.resize(img,(width,height))
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow("frame", img_axis)
        cv2.waitKey(0)  # any key
        cv2.destroyWindow("frame")
        decimator += 1
        imsize = img.shape
        print("size of img is ", imsize)

        # rtcam: camera coordinate system to world coordinate system
        # rtMat: world coordinate system to camera coordinate system
        rtMat = np.hstack((r_mirror_mat, t_mirror))
        rtMat = np.vstack((rtMat, np.array([[0, 0, 0, 1]])))
        print("rtMat is", rtMat)
        rt_cam = np.linalg.inv(rtMat)
        r_cam = rt_cam[0:3, 0:3]
        t_cam = rt_cam[0:3, 3]
        print("rtcam in world coor is ", rt_cam)

        # rtdis: display coordinate system to world coordinate system
        # rtdis_inv: world coordinate system to display coordinate system
        t_camdis = np.reshape(tcamera_dis, (tcamera_dis.shape[0], 1))
        rtMat_camdis = np.hstack((rcamera_dis, t_camdis))
        rtMat_camdis = np.vstack((rtMat_camdis, np.array([0, 0, 0, 1])))
        # rtMat_discam = np.linalg.inv(rtMat_camdis)
        # rt_dis_inv = np.matmul(rtMat, rtMat_camdis)
        rt_dis_inv = np.matmul(rtMat_camdis, rtMat)
        rt_dis = np.linalg.inv(rt_dis_inv)
        r_dis = rt_dis[0:3, 0:3]
        t_dis = rt_dis[0:3, 3]
        r_dis_inv = rt_dis_inv[0:3, 0:3]
        t_dis_inv = rt_dis_inv[0:3, 3]

        print("rtscreen in world coor is ", rt_dis)
        print("mirror in screen coor is ", rt_dis_inv)
        # reshape t vector
        t_cam = np.reshape(t_cam, (t_cam.shape[0],))
        t_dis = np.reshape(t_dis, (t_dis.shape[0],))
        t_dis_inv = np.reshape(t_dis_inv, (t_dis_inv.shape[0],))
        # reshape for functionality
        t_cam = np.reshape(t_cam, (3, 1))
        t_dis = np.reshape(t_dis, (3, 1))
        t_dis_inv = np.reshape(t_dis_inv, (3, 1))

        # read undistorted image
        img_undistort = cv2.imread(imgUndistortFile)
        gray_undistort = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)
        img_undistort_size = gray_undistort.shape
        print(img_undistort_size)

        # cam roor mat:store arrival vectors (vector from camera to the object) in camera coordinate system
        # cam world coor mat: store arrival vectors in world coordinate system
        # camera mirror intersect mat: store intersect points between arrival vectors and mirror in world coordinate system
        # reflec mat: store reflectance vectors (vector from object to screen) in world coordinate system
        # mirror intersect mat trans:store intersect points on mirrors at display coordinate system
        # reflect mat trans: store reflectance vectors in display coordinate system
        # display intersect mat: store display intersect points (intersect points between reflectance vectors and display) on display coordinate system
        # display intersect mat trans: store display intersect points on world coordinate system
        img_coor_mat = np.zeros((3, img_undistort_size[0], img_undistort_size[1]))
        img_coor_mat_rs = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        cam_coor_mat = np.zeros((img_undistort_size[0], img_undistort_size[1], 3))
        cam_coor_mat_rs = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        cam_world_coor_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        cam_mirror_intersect_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        reflect_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        mirror_intersect_mat_trans = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        reflect_mat_trans = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        display_intersect_mat = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))
        display_intersect_mat_trans = np.zeros((3, img_undistort_size[0] * img_undistort_size[1]))

        print(cam_coor_mat.shape)
        print(np.linalg.inv(cameraMatrix))
        for i in range(cam_coor_mat.shape[0]):
            for j in range(cam_coor_mat.shape[1]):
                image_coor = np.array([j + 1, i + 1, 1])
                img_coor_mat[:, i, j] = image_coor

        img_coor_mat_rs = np.reshape(img_coor_mat, (3, img_undistort_size[0] * img_undistort_size[1]))
        # calculate arrival vector
        cam_coor_mat_rs = np.matmul(np.linalg.inv(cameraMatrix), img_coor_mat_rs)
        cam_world_coor_mat = np.matmul(r_cam, cam_coor_mat_rs)

        # calculate intersect point
        scale_factor1 = -t_cam[2, 0] / cam_world_coor_mat[2, :]
        scale_factor1 = np.reshape(scale_factor1, (1, scale_factor1.shape[0]))
        cam_mirror_intersect_mat = scale_factor1 * cam_world_coor_mat + np.tile(t_cam, (cam_world_coor_mat.shape[1]))
        # calculate reflectance vector in world coor system
        reflect_mat = cam_world_coor_mat - 2 * np.dot(np.array([0, 0, 1]), cam_world_coor_mat) * np.tile(
            np.array([[0], [0], [1]]), (1, cam_world_coor_mat.shape[1]))
        # calculate intersect point in display coor system
        mirror_intersect_mat_trans = np.dot(r_dis_inv, cam_mirror_intersect_mat) + np.tile(t_dis_inv, (
        cam_world_coor_mat.shape[1]))
        # calculate reflectance vector in display coor system
        reflect_mat_trans = np.dot(r_dis_inv, reflect_mat)
        # calculate display intersect point under display coordinate system
        scale_factor2 = -mirror_intersect_mat_trans[2, :] / reflect_mat_trans[2, :]
        display_intersect_mat = scale_factor2 * reflect_mat_trans + mirror_intersect_mat_trans
        # calculate display intersect point under world coordinate system
        display_intersect_mat_trans = np.dot(r_dis, display_intersect_mat) + np.tile(t_dis,
                                                                                     (cam_world_coor_mat.shape[1]))

        # restore the matrix back to the same shape of undistort image
        display_intersect_mat = np.reshape(display_intersect_mat, (3, img_undistort_size[0], img_undistort_size[1]))
        display_intersect_mat_trans = np.reshape(display_intersect_mat_trans,
                                                 (3, img_undistort_size[0], img_undistort_size[1]))
        cam_mirror_intersect_mat = np.reshape(cam_mirror_intersect_mat,
                                              (3, img_undistort_size[0], img_undistort_size[1]))

        ## this part is to display camera, mirror &projector within a unified world coordinate systems

        # reshape t vector
        t_cam = np.reshape(t_cam, (t_cam.shape[0],))
        t_dis = np.reshape(t_dis, (t_dis.shape[0],))
        t_dis_inv = np.reshape(t_dis_inv, (t_dis_inv.shape[0],))

        # mirror corner index
        markerCornerIDs = np.array(
            [np.argwhere(markerIDs == 0), np.argwhere(markerIDs == 2), np.argwhere(markerIDs == 4),
             np.argwhere(markerIDs == 6)])
        markerCornerIDs = markerCornerIDs[:, 0, 0]

        mirror_corner_value = np.array([markerCorners[markerCornerIDs[0]][0][0],
                                        markerCorners[markerCornerIDs[1]][0][3],
                                        markerCorners[markerCornerIDs[2]][0][2],
                                        markerCorners[markerCornerIDs[3]][0][1],
                                        markerCorners[markerCornerIDs[0]][0][0]], dtype=np.int32)
        print("mirror corner index is", mirror_corner_value)
        mirror_corner_world = np.zeros((mirror_corner_value.shape[0], 3))
        for i in range(mirror_corner_world.shape[0]):
            mirror_corner_world[i, :] = cam_mirror_intersect_mat[:, mirror_corner_value[i, 1],
                                        mirror_corner_value[i, 0]]
        print("mirror corner point is", mirror_corner_world)

        # display corners in world coordinate system
        SCALE_FACTOR = self.checker_pixel_size
        half_length_disp = (SCREEN_WIDTH / 2) * SCALE_FACTOR
        half_height_disp = (SCREEN_HEIGHT / 2) * SCALE_FACTOR
        disp_uleft = np.array([-half_length_disp, -half_height_disp, 0])
        disp_lleft = np.array([-half_length_disp, half_height_disp, 0])
        disp_lright = np.array([half_length_disp, half_height_disp, 0])
        disp_uright = np.array([half_length_disp, -half_height_disp, 0])
        disp_uleft_world = np.reshape(np.matmul(r_dis, disp_uleft) + t_dis, (1, t_dis.shape[0]))
        disp_lleft_world = np.reshape(np.matmul(r_dis, disp_lleft) + t_dis, (1, t_dis.shape[0]))
        disp_lright_world = np.reshape(np.matmul(r_dis, disp_lright) + t_dis, (1, t_dis.shape[0]))
        disp_uright_world = np.reshape(np.matmul(r_dis, disp_uright) + t_dis, (1, t_dis.shape[0]))

        disp_corners_world = np.concatenate(
            (disp_uleft_world, disp_lleft_world, disp_lright_world, disp_uright_world, disp_uleft_world), axis=0)
        print("disp corners in world coordinate is ", disp_corners_world)

        # calculate the (0,0) and z axis of camera and display (for verification)
        cam_z1 = np.matmul(r_cam, np.array([0, 0, 1])) + t_cam
        cam_zero = np.matmul(r_cam, np.array([0, 0, 0])) + t_cam
        display_z1 = np.matmul(r_dis, np.array([0, 0, 1])) + t_dis
        display_zero = np.matmul(r_dis, np.array([0, 0, 0])) + t_dis
        # cam_z = np.concatenate((np.reshape(cam_zero, (1, cam_zero.shape[0])), np.reshape(cam_z1, (1, cam_z1.shape[0]))), axis= 0)
        # display_z = np.concatenate((np.reshape(display_zero, (1,display_zero.shape[0])), np.reshape(display_z1, (1, display_z1.shape[0]))), axis= 0)
        cam_z = cam_z1 - cam_zero
        display_z = display_z1 - display_zero
        print('cam z axis in world coor is ', cam_z)
        print("cam zero point in world coor is ", cam_zero)
        print("display z axis in world coor is ", display_z)
        print("display zero in world coor is ", display_zero)
        # plot 
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(mirror_corner_world[:, 0], mirror_corner_world[:, 1], mirror_corner_world[:, 2], 'b-', label='Mirror')
        ax.plot(disp_corners_world[:, 0], disp_corners_world[:, 1], disp_corners_world[:, 2], 'r-', label='Display')
        ax.scatter3D(cam_zero[0], cam_zero[1], cam_zero[2], 'go', label='Camera')
        ax.quiver(cam_zero[0], cam_zero[1], cam_zero[2], cam_z[0], cam_z[1], cam_z[2], length=0.1)
        ax.quiver(display_zero[0], display_zero[1], display_zero[2], display_z[0], display_z[1], display_z[2],
                  length=0.1)

        plt.legend()
        plt.show()
        return display_intersect_mat
