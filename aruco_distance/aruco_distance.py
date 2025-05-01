#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, PointStamped
import cv2
import numpy as np
import cv2.aruco as aruco

class ArucoDistanceDetector(Node):
    def __init__(self):
        super().__init__('aruco_distance_detector')
        
        # --- Configuración de parámetros ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/camera/color/image_raw'),
                ('marker_size', 0.09),  # en metros
                ('dictionary_id', 'DICT_4X4_250'),
                ('show_image', True),
                ('camera_matrix', [1491.69912, 0.0, 1137.52964, 0.0, 1432.22457, 407.952679, 0.0, 0.0, 1.0]),
                ('dist_coeffs', [0.06383568, 0.15331208, -0.01669957, 0.05292742, -0.28933054])
            ]
        )
        
        # Obtener parámetros
        self.camera_topic = self.get_parameter('camera_topic').value
        self.marker_size = self.get_parameter('marker_size').value
        dictionary_id = self.get_parameter('dictionary_id').value
        self.show_image = self.get_parameter('show_image').value
        
        # Configurar diccionario ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT[dictionary_id])
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Configurar matriz de cámara y coeficientes de distorsión
        camera_matrix = self.get_parameter('camera_matrix').value
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32).reshape(3, 3)
        dist_coeffs = self.get_parameter('dist_coeffs').value
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        
        # Bridge para conversión de imágenes ROS-OpenCV
        self.bridge = CvBridge()
        
        # --- Publicadores y suscriptores ---
        # Suscriptor a la imagen de la cámara
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback,10)
        
        # Publicador para poses de los marcadores
        self.markers_pub = self.create_publisher(PoseArray, 'aruco/marker_poses', 10)
        
        # Publicador para distancias de los marcadores
        self.distances_pub = self.create_publisher(PointStamped, 'aruco/marker_distances', 10)
        
        # Publicador para imagen con anotaciones
        self.image_pub = self.create_publisher(Image, 'aruco/detected_image', 10)
        
        self.get_logger().info(f'Nodo de detección de ArUco inicializado. Esperando imágenes en {self.camera_topic}')

    def image_callback(self, msg):
        try:
            # Convertir imagen ROS a OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_with_markers = frame.copy()
            
            # Detectar marcadores
            corners, ids, _ = self.detector.detectMarkers(frame)
            
            if ids is not None:
                pose_array = PoseArray()
                pose_array.header = msg.header
                
                for i in range(len(ids)):
                    # Estimar pose del marcador
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i], self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    
                    # Calcular distancia euclidiana
                    distance = np.linalg.norm(tvec)
                    
                    # Publicar distancia
                    distance_msg = PointStamped()
                    distance_msg.header = msg.header
                    distance_msg.header.frame_id = f"aruco_{ids[i][0]}"
                    distance_msg.point.x = distance
                    self.distances_pub.publish(distance_msg)
                    
                    # Crear mensaje de pose
                    pose = Pose()
                    pose.position.x = tvec[0][0][0]
                    pose.position.y = tvec[0][0][1]
                    pose.position.z = tvec[0][0][2]
                    
                    # Convertir vector de rotación a cuaternión
                    rotation_matrix = np.eye(4)
                    rotation_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
                    quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                    
                    pose.orientation.x = quaternion[0]
                    pose.orientation.y = quaternion[1]
                    pose.orientation.z = quaternion[2]
                    pose.orientation.w = quaternion[3]
                    
                    pose_array.poses.append(pose)
                    
                    # Dibujar marcador y distancia
                    cv2.aruco.drawDetectedMarkers(frame_with_markers, corners, ids)
                    cv2.drawFrameAxes(frame_with_markers, self.camera_matrix, self.dist_coeffs, 
                                     rvec, tvec, self.marker_size * 0.5)
                    
                    text_pos = tuple(map(int, corners[i][0][0]))
                    cv2.putText(
                        frame_with_markers, 
                        f"ID {ids[i][0]}: {distance:.2f}m", 
                        text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                
                # Publicar poses
                self.markers_pub.publish(pose_array)
            
            # Publicar imagen procesada
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame_with_markers, encoding='bgr8'))
            
            # Mostrar imagen (opcional)
            if self.show_image:
                cv2.imshow("ArUco Detection", frame_with_markers)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f'Error en el procesamiento de imagen: {str(e)}')

    def rotation_matrix_to_quaternion(self, R):
        """Convertir matriz de rotación a cuaternión"""
        q = np.empty((4, ), dtype=np.float32)
        q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
        q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
        q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
        return q

# Diccionario de tipos de marcadores ArUco
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDistanceDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_image:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()