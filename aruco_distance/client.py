import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import asyncio
import websockets
import cv2
import json
from threading import Thread

SERVER_URI = "ws://10.43.59.42:8000/ws"  # Update to your laptop IP

CLASS_DICT = {
    0 : "AVGloadingzone",
    1 : "AVGparking",
    2 : "AVGzone",
    3 : "PROHIBIDO",
    4 : "STOP",
    5 : "peaton"
}

class VisionClientNode(Node):
    def __init__(self):
        super().__init__('vision_client')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.publisher_ = self.create_publisher(String, '/detections', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 👈 Replace with your actual topic
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    async def websocket_loop(self):
        async with websockets.connect(SERVER_URI, max_size=2**25) as websocket:
            self.get_logger().info("Connected to WebSocket server")

            while rclpy.ok():
                if self.latest_frame is None:
                    await asyncio.sleep(0.1)
                    continue

                _, jpeg = cv2.imencode('.jpg', self.latest_frame)
                await websocket.send(jpeg.tobytes())
                response = await websocket.recv()
                data = json.loads(response)
                detections = data["detections"]

                # Extrae el nombre de la clase
                class_number = detections[0]["class"] if detections else None
                class_name = CLASS_DICT.get(class_number, "Desconocido")

                msg = String()
                msg.data = class_name
                self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = VisionClientNode()

    loop = asyncio.get_event_loop()

    def ros_spin():
        rclpy.spin(node)

    spin_thread = Thread(target=ros_spin)
    spin_thread.start()

    try:
        loop.run_until_complete(node.websocket_loop())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()
