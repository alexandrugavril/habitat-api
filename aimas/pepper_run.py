import roslibpy
from roslibpy import Message, Ros, Topic
import base64
import numpy as np
import cv2
import habitat
from habitat import SimulatorActions

rgb_buffer = []
depth_buffer = []
forward_step = 0.25
turn_step = 0.1

#cfg = habitat.get_config("configs/tasks/explore_pepper.yaml")


def fetch_rgb(message):
    img = np.frombuffer(base64.b64decode(message['data']), np.uint8)
    img = img.reshape((message['height'], message['width'], 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rgb_buffer.append(img)
    if len(rgb_buffer) > 10:
        rgb_buffer.pop(0)


def fetch_depth(message):
    img = np.frombuffer(base64.b64decode(message['data']), np.uint16)
    img = img.reshape((message['height'], message['width'], 1))
    depth_buffer.append(img)

    if len(depth_buffer) > 10:
        depth_buffer.pop(0)


def get_movement_ros_message(fw_step, r_step):
    frame_id = 'base_footprint'
    m = Message({
        'header': {
            'frame_id': frame_id
        },
        'pose': {
            'position': {
                'x': fw_step,
                'y': 0,
                'z': 0
            },
            'orientation': {
                'x': 0,
                'y': 0,
                'z': 0,
                'w': r_step
            }
        }
    })
    return m


def send_action(action):
    print("Action:", action)
    if action == "MOVE_FORWARD":
        m = get_movement_ros_message(forward_step, 0)
        publisher_move.publish(m)
    elif action == 'TURN_LEFT':
        m = get_movement_ros_message(0, turn_step)
        publisher_move.publish(m)
    elif action == 'TURN_RIGHT':
        m = get_movement_ros_message(0, -1*turn_step)
        publisher_move.publish(m)


ros = roslibpy.Ros(host='localhost', port=9090)
ros.run()
print(ros.is_connected)

listener = roslibpy.Topic(ros,
                          '/pepper_robot/naoqi_driver/camera/front/image_raw',
                          'sensor_msgs/Image')

listener_depth = roslibpy.Topic(ros,
                          '/pepper_robot/naoqi_driver/camera/depth/image_raw',
                          'sensor_msgs/Image')

publisher_move = roslibpy.Topic(ros,
                                '/move_base_simple/goal',
                                'geometry_msgs/PoseStamped')

listener.subscribe(lambda message: fetch_rgb(message))
listener_depth.subscribe(lambda message: fetch_depth(message))


try:
    while True:
        if len(rgb_buffer) > 0 and len(depth_buffer) > 0:
            cv2.imshow("Image", rgb_buffer[-1])
            cv2.imshow("Depth", depth_buffer[-1])
            key = cv2.waitKey(1)
            if key == ord('w'):
                send_action("MOVE_FORWARD")
            elif key == ord('a'):
                send_action('TURN_LEFT')
            elif key == ord('d'):
                send_action('TURN_RIGHT')


        pass
except KeyboardInterrupt:
    ros.terminate()
