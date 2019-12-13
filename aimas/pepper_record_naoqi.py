# vim: set fileencoding=utf-8 :
import sys
import numpy as np
import cv2
from naoqi import ALProxy
import argparse
from PIL import Image
import numpy
import base64
import re
import pickle
import datetime
import time


def main(ip_addr, port_num):
    """
    First get an image, then show it on the screen with PIL.
    """
    # Get the service ALVideoDevice.

    # get NAOqi module proxy
    videoDevice = ALProxy('ALVideoDevice', ip_addr, port_num)

    # subscribe top camera
    AL_kTopCamera = 0
    AL_kQVGA = 0  # 320x240
    AL_kBGRColorSpace = 13
    captureDevice = videoDevice.subscribeCamera(
        "1te2121111st", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)
    captureDeviceDepth = videoDevice.subscribeCamera(
        "test1112112", 2, AL_kQVGA, AL_kBGRColorSpace, 10)

    key = 0
    no_images = 0
    t_time = 0
    saved_images = []
    while key != 27:
        t0 = time.time()

        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.
        naoImage = videoDevice.getImageRemote(captureDevice)
        depthImage = videoDevice.getImageRemote(captureDeviceDepth)
        if naoImage != None and depthImage != None:
            t1 = time.time()

            # Time the image transfer.
            t_time += t1 - t0
            no_images += 1

            # Now we work with the image returned and save it as a PNG  using ImageDraw
            # package.

            # Get the image size and pixel array.
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_string = str(bytearray(array))

            dWidth = depthImage[0]
            dHeight = depthImage[1]
            d_array = depthImage[6]

            d_int_array = []
            for i in range(0, len(d_array) - 1, 2):
                y = d_array[i] + d_array[i+1]
                b = np.frombuffer(y, dtype=np.uint16)
                d_int_array.append(b[0])

            d_np_array = numpy.array(d_int_array, dtype=np.uint16)
            depth_image = d_np_array.reshape((dHeight, dWidth, 1))
            depth_image = (depth_image.astype(np.float) / 9100).astype(np.float)

            # Create a PIL Image from our pixel array.
            im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)
            open_cv_image = numpy.array(im)

            # Save the image.
            cv2.imshow("RGB", open_cv_image)
            cv2.imshow("Depth", depth_image)
            saved_images.append({
                "rgb": open_cv_image,
                "depth": depth_image
            })
            if(len(saved_images) % 250 == 0):
                now = datetime.datetime.now()
                print("No Images...", no_images)
                print("Dumping...")
                print("Average time...", t_time / no_images)
                print("-" * 100)
                dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
                pickle.dump(saved_images, open(dt_string + "_recording_pepper.p", "wb"))
                saved_images = []

            key = cv2.waitKey(1)

    now = datetime.datetime.now()
    dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
    pickle.dump(saved_images, open(dt_string + "_recording_pepper.p", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.115",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    main(args.ip, args.port)
