from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    aruco_detection = Node(
        package="aruco_distance",
        executable="distance",
    )

    container = ComposableNodeContainer(
        name='camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',  # o component_container
        composable_node_descriptions=[
            ComposableNode(
                package='astra_camera',
                plugin='astra_camera::OBCameraNodeFactory',
                name='camera',
                namespace='camera',
                parameters=[  # copia aquí los mismos parámetros de tu XML original
                    # {"param_name": value}, …
                ],
                remappings=[
                    # ['/camera/depth/color/points', '/camera/depth_registered/points'],
                    # …
                ],
            ),
        ],
        output='screen',
    )
    return LaunchDescription([container, aruco_detection])
