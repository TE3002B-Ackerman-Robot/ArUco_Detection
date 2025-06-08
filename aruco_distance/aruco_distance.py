#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseWithCovarianceStamped
import cv2.aruco as aruco
import numpy as np
import math
import cv2

# Diccionario de posiciones conocidas de ArUcos (sin orientación)
ARUCO_DIST = {
    1: (0.3, 0.3),
    2: (2.10, 0.3),
    3: (2.10, 2.10),
    4: (0.3, 2.10),
    5: (4.0, 2.24),
    6: (5.0, 2.24)
}

# Mapeo de identificadores de diccionarios ArUco
ARUCO_DICT = {
    "DICT_4X4_50":    aruco.DICT_4X4_50,
    "DICT_4X4_100":   aruco.DICT_4X4_100,
    "DICT_4X4_250":   aruco.DICT_4X4_250,
    "DICT_4X4_1000":  aruco.DICT_4X4_1000,
    "DICT_5X5_50":    aruco.DICT_5X5_50,
    "DICT_5X5_100":   aruco.DICT_5X5_100,
    "DICT_5X5_250":   aruco.DICT_5X5_250,
    "DICT_5X5_1000":  aruco.DICT_5X5_1000,
    "DICT_6X6_50":    aruco.DICT_6X6_50,
    "DICT_6X6_100":   aruco.DICT_6X6_100,
    "DICT_6X6_250":   aruco.DICT_6X6_250,
    "DICT_6X6_1000":  aruco.DICT_6X6_1000,
    "DICT_7X7_50":    aruco.DICT_7X7_50,
    "DICT_7X7_100":   aruco.DICT_7X7_100,
    "DICT_7X7_250":   aruco.DICT_7X7_250,
    "DICT_7X7_1000":  aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5":  aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9":  aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11
}

def estimate_robot_pose_from_marker(Xm, Ym, rvec, tvec):
    """
    Dada la posición conocida del marcador (Xm, Ym) y
    la pose relativa (rvec, tvec) devuelve la pose del robot.
    """
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    t_cam_marker = tvec.reshape(3, 1)
    R_marker_cam = R_cam_marker.T
    t_marker_cam = -R_marker_cam @ t_cam_marker
    x_robot = Xm + t_marker_cam[0, 0]
    y_robot = Ym + t_marker_cam[2, 0]
    yaw     = math.atan2(R_marker_cam[1, 0], R_marker_cam[0, 0])
    return x_robot, y_robot, yaw


class ArucoDistanceDetector(Node):
    def __init__(self):
        super().__init__('aruco_distance_detector')

        # Parámetros
        self.declare_parameters(
            namespace='',
            parameters=[
                ('marker_size', 0.115),
                ('dictionary_id', 'DICT_4X4_250'),
                ('camera_matrix', [465.0, 0.0,   290.0,
                                   0.0,   465.0, 170.0,
                                   0.0,   0.0,   1.0  ]),
                ('dist_coeffs', [0.29240194,  -2.31175373,
                                 -0.01181112, -0.01688533,
                                 6.86271224])
            ]
        )

        # Lectura de parámetros
        self.marker_size   = self.get_parameter('marker_size').value
        dict_id            = self.get_parameter('dictionary_id').value
        self.aruco_dict    = aruco.getPredefinedDictionary(ARUCO_DICT[dict_id])
        self.detector      = aruco.ArucoDetector(self.aruco_dict, aruco.DetectorParameters())
        cam_mat            = self.get_parameter('camera_matrix').value
        self.camera_matrix = np.array(cam_mat, dtype=np.float32).reshape(3,3)
        dist_c             = self.get_parameter('dist_coeffs').value
        self.dist_coeffs   = np.array(dist_c, dtype=np.float32)

        # Bridge
        self.bridge = CvBridge()

        # Publicadores
        self.markers_pub   = self.create_publisher(PoseArray,                 'aruco/marker_poses',     10)
        self.distances_pub = self.create_publisher(PointStamped,              'aruco/marker_distances', 10)
        self.image_pub     = self.create_publisher(Image,                     'aruco/detected_image',   10)
        self.pose_cov_pub  = self.create_publisher(PoseWithCovarianceStamped, 'aruco/robot_pose',       10)

        # Suscripciones
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)

        self.get_logger().info(f'Nodo de detección de ArUco inicializado')


    def image_callback(self, msg: Image):
        # 1) Convertir imagen
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 2) Detectar marcadores y estimar rvec/tvec
        corners, ids, rvecs, tvecs = self.detect_markers(frame)

        # 3) Publicar info y dibujar
        if ids is not None:
            self.publish_marker_info(corners, ids, rvecs, tvecs, msg.header)
            frame_out = self.draw_detections(frame, corners, ids, rvecs, tvecs)
        else:
            frame_out = frame

        # 4) Publicar imagen resultante
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame_out, encoding='bgr8'))


    def detect_markers(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is None or len(ids) == 0:
            return None, None, None, None

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.dist_coeffs
        )

        return corners, ids, rvecs, tvecs


    def publish_marker_info(self, corners, ids, rvecs, tvecs, header):
        pose_array = PoseArray()
        pose_array.header = header

        for idx, aruco_id in enumerate(ids.flatten()):
            # Obtener vectores
            rvec = rvecs[idx][0]
            tvec = tvecs[idx][0]

            # --- Distancia ---
            dist = float(np.linalg.norm(tvec))
            pm = PointStamped()
            pm.header = header
            pm.header.frame_id = f'aruco_{aruco_id}'
            pm.point.x = dist
            self.distances_pub.publish(pm)

            # --- Pose del marcador ---
            p = Pose()
            p.position.x, \
            p.position.y, \
            p.position.z = float(tvec)

            # Quaternion del marcador
            rot_mat = cv2.Rodrigues(rvec)[0]
            quat4x4 = np.eye(4)
            quat4x4[:3,:3] = rot_mat
            p.orientation.x, \
            p.orientation.y, \
            p.orientation.z, \
            p.orientation.w = self.rotation_matrix_to_quaternion(quat4x4)
            pose_array.poses.append(p)

            # --- Pose del robot corregida ---
            if aruco_id in ARUCO_DIST:
                Xm, Ym = ARUCO_DIST[aruco_id]
                xr, yr, yaw = estimate_robot_pose_from_marker(Xm, Ym, rvec, tvec)
                pwm = PoseWithCovarianceStamped()
                pwm.header = header
                pwm.pose.pose.position.x = xr
                pwm.pose.pose.position.y = yr
                pwm.pose.pose.position.z = 0.0

                # Orientación yaw
                qz = math.sin(yaw/2.0)
                qw = math.cos(yaw/2.0)
                pwm.pose.pose.orientation.z = qz
                pwm.pose.pose.orientation.w = qw

                # Covarianza diagonal (0.02 m, 5°)
                var_xy  = 0.02**2
                var_yaw = math.radians(5.0)**2
                cov = [0.0]*36
                cov[0]  = var_xy
                cov[7]  = var_xy
                cov[35] = var_yaw
                pwm.pose.covariance = cov
                self.pose_cov_pub.publish(pwm)

        self.markers_pub.publish(pose_array)


    def draw_detections(self, frame, corners, ids, rvecs, tvecs):
        out_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(out_frame, corners, ids)

        for idx, aruco_id in enumerate(ids.flatten()):
            cv2.drawFrameAxes(
                out_frame,
                self.camera_matrix,
                self.dist_coeffs,
                rvecs[idx][0],
                tvecs[idx][0],
                self.marker_size * 0.5
            )

            # Etiqueta de distancia
            pt = tuple(map(int, corners[idx][0][0]))
            d  = float(np.linalg.norm(tvecs[idx][0]))
            cv2.putText(
                out_frame,
                f'ID {aruco_id}: {d:.2f}m',
                pt,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        return out_frame

    def rotation_matrix_to_quaternion(self, M):
        # Asume M es 4x4
        q = [0.0, 0.0, 0.0, 0.0]
        q[3] = math.sqrt(max(0, 1 + M[0,0] + M[1,1] + M[2,2])) / 2
        q[0] = math.sqrt(max(0, 1 + M[0,0] - M[1,1] - M[2,2])) / 2
        q[1] = math.sqrt(max(0, 1 - M[0,0] + M[1,1] - M[2,2])) / 2
        q[2] = math.sqrt(max(0, 1 - M[0,0] - M[1,1] + M[2,2])) / 2
        q[0] *= math.copysign(1, (M[2,1] - M[1,2]))
        q[1] *= math.copysign(1, (M[0,2] - M[2,0]))
        q[2] *= math.copysign(1, (M[1,0] - M[0,1]))
        return q


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDistanceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
