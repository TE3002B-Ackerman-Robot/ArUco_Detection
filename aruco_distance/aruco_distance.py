#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseWithCovarianceStamped
import numpy as np
import math
import cv2

# Diccionario de posiciones conocidas de ArUcos (sin orientación)
ARUCO_DIST = {
    0: (0.00, 0.00, 0.0),
    1: (0.30, 0.30, math.radians(90.0)),
    2: (2.40, 0.30, math.radians(180.0)),
    3: (0.30, 2.10, math.radians(180.0)),
    4: (2.40, 2.40, math.radians(-90.0)),
    5: (5.0, 2.24, math.radians(-90.0)),
    6: (4.0, 2.24, 0.0),
    7: (5.0, 2.24, 0.0),
    8: (5.0, 2.24, 0.0),
    8: (5.0, 2.24, math.radians(90.0))
}

# Mapeo de identificadores de diccionarios ArUco
ARUCO_DICT = {
    "DICT_4X4_50":    cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":   cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":   cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000":  cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":    cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":   cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":   cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000":  cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":    cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":   cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":   cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000":  cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":    cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":   cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":   cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000":  cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5":  cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def estimate_robot_pose_from_marker(Xm, Ym, theta_m, rvec, tvec):
    # 1. convierte rvec→matriz y extrae tvec
    R_cam_marker, _ = cv2.Rodrigues(rvec)
    t_cam_marker   = tvec.reshape(3,)

    # 2. invierte la tf (cámara en frame marcador)
    t_marker_robot = -R_cam_marker.T @ t_cam_marker

    # 3. gira ese vector desde frame “marcador” a frame “mapa”
    dx = math.cos(theta_m) * t_marker_robot[0] - math.sin(theta_m) * t_marker_robot[1]
    dy = math.sin(theta_m) * t_marker_robot[0] + math.cos(theta_m) * t_marker_robot[1]

    # 4. traslada con la posición fija del marcador
    x_robot = Xm - dx
    y_robot = Ym - dy

    # 5. calcula el yaw global
    yaw_rel = math.atan2(R_cam_marker[1, 0], R_cam_marker[0, 0])
    yaw_robot = (theta_m + yaw_rel + math.pi) % (2 * math.pi)  # Nota el signo negativo aquí

    return x_robot, y_robot, yaw_robot



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
        self.aruco_dict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_id])
        self.detector      = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
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
        self.pose_cov_pub  = self.create_publisher(PoseWithCovarianceStamped, 'aruco/pose_cov',         10)

        # Suscripciones
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)

        self.get_logger().info(f'Nodo de detección de ArUco inicializado')


    def image_callback(self, msg: Image):
        # 1) Convertir imagen
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 2) Detectar marcadores y estimar rvec/tvec
        corners, ids = self.detector.detectMarkers(frame)[:2]

        # 3) Publicar info y dibujar
        if ids is not None and len(ids):
            rvecs, tvecs = self.estimate_poses(frame, corners)
            self.publish_info(corners, ids, rvecs, tvecs, msg.header)
            out = self.draw(frame, corners, ids, rvecs, tvecs)
        else:
            out = frame

        # 4) Publicar imagen resultante
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))


    def estimate_poses(self, frame, corners):
        # Intentar usar cv2.aruco si disponible
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            return rvecs, tvecs

        except AttributeError:
            # Fallback manual con solvePnP
            obj_pts = np.array([
                [-self.marker_size/2,  self.marker_size/2, 0],
                [ self.marker_size/2,  self.marker_size/2, 0],
                [ self.marker_size/2, -self.marker_size/2, 0],
                [-self.marker_size/2, -self.marker_size/2, 0]
            ], dtype=np.float32)
            rvecs = []
            tvecs = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for c in corners:
                img_pts = c.reshape(4,2).astype(np.float32)
                ok, r, t = cv2.solvePnP(obj_pts, img_pts,
                                        self.camera_matrix, self.dist_coeffs)
                if ok:
                    rvecs.append(r)
                    tvecs.append(t)

            # Dar formato igual al estimatePoseSingleMarkers
            if rvecs:
                rvecs = np.array(rvecs).reshape(-1,1,3)
                tvecs = np.array(tvecs).reshape(-1,1,3)
            else:
                rvecs = np.empty((0,1,3))
                tvecs = np.empty((0,1,3))

            return rvecs, tvecs


    def publish_info(self, corners, ids, rvecs, tvecs, header):
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
            p.position.z = tvec

            # Quaternion del marcador
            rot_mat = cv2.Rodrigues(rvec)[0]
            quat4x4 = np.eye(4)
            quat4x4[:3,:3] = rot_mat
            q = self.rotation_matrix_to_quaternion(quat4x4)
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            pose_array.poses.append(p)

            # --- Pose del robot corregida ---
            if aruco_id in ARUCO_DIST:
                Xm, Ym, THETAm = ARUCO_DIST[aruco_id]
                xr, yr, yawr = estimate_robot_pose_from_marker(Xm, Ym, THETAm, rvec, tvec)

                yaw=THETAm+yawr

                pwc = PoseWithCovarianceStamped()
                pwc.header.stamp = header.stamp
                pwc.header.frame_id = 'map'
                pwc.pose.pose.position.x = xr
                pwc.pose.pose.position.y = yr
                pwc.pose.pose.position.z = 0.0

                # Orientación yaw
                qz = math.sin(yaw / 2.0)
                qw = math.cos(yaw / 2.0)
                pwc.pose.pose.orientation.z = qz
                pwc.pose.pose.orientation.w = qw

                # Covarianza diagonal (0.02 m, 5°)
                var_xy  = 0.02**2
                var_yaw = math.radians(5.0)**2
                cov = [0.0]*36
                cov[0]  = var_xy
                cov[7]  = var_xy
                cov[35] = var_yaw
                pwc.pose.covariance = cov

                self.pose_cov_pub.publish(pwc)

        self.markers_pub.publish(pose_array)


    def draw(self, frame, corners, ids, rvecs, tvecs):
        out = frame.copy()
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for i, id_ in enumerate(ids.flatten()):
            rv = rvecs[i][0]; tv = tvecs[i][0]
            cv2.drawFrameAxes(out, self.camera_matrix, self.dist_coeffs, rv, tv, self.marker_size*0.5)
            pt = tuple(map(int, corners[i][0][0]))
            d = float(np.linalg.norm(tv))
            cv2.putText(out, f'ID {id_}: {d:.2f}m', pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return out


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