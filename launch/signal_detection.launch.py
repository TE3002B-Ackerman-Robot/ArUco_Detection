from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=["ros2", "launch", "astra_camera", "astro_pro_plus.launch.xml"]),
        Node(
            package="web_video_server",
            executable="web_video_server"
        ),
        Node(
            package="aruco_distance",
            executable="client",
        )
    ])
