import rclpy
from rclpy.node import Node
import asyncio
import websockets
import cv2
import json
from std_msgs.msg import String  # or custom message for detections

SERVER_URI = "ws://192.168.X.X:8000/ws"

class VisionClientNode(Node):
    def __init__(self):
        super().__init__('vision_client')
        self.publisher_ = self.create_publisher(String, '/detections', 10)

        # Launch WebSocket client
        asyncio.create_task(self.websocket_loop())

    async def websocket_loop(self):
        async with websockets.connect(SERVER_URI, max_size=2**25) as websocket:
            self.get_logger().info("Connected to server")
            
            cap = cv2.VideoCapture(0)  # USB or CSI cam depending on your setup
            
            while rclpy.ok():
                ret, frame = cap.read()
                if not ret:
                    continue

                # Encode frame
                _, jpeg = cv2.imencode('.jpg', frame)
                jpeg_bytes = jpeg.tobytes()

                # Send frame
                await websocket.send(jpeg_bytes)

                # Receive detections
                response = await websocket.recv()
                data = json.loads(response)

                # Publish to ROS 2 topic
                msg = String()
                msg.data = json.dumps(data["detections"])
                self.publisher_.publish(msg)

                self.get_logger().info(f"Published detections: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
