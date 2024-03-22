import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
from draw import Draw
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class Vertices:

    def __init__(self, width, height,results):
        self.image_width = width
        self.image_height = height

        self.point_a = self.PointA()
        self.point_b = self.PointB()
        
        self.point_a.x = (results.pose_landmarks.landmark[24].x + results.pose_landmarks.landmark[23].x) / 2
        self.point_a.y = (results.pose_landmarks.landmark[24].y + results.pose_landmarks.landmark[23].y) / 2
        print('this')
        self.point_b.x = (results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) / 2
        self.point_b.y = (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2

    def get_vertices(self, index, results):

        if index == 'A':
            return self.point_a

        elif index == 'B':
            return self.point_b

        else:
            return results.pose_landmarks.landmark[index]

    def calculate_distance(self, x1, y1, x2, y2):
        return int((math.sqrt(((x2*self.image_width) - (x1*self.image_width)) ** 2 + ((y2*self.image_height) - (y1*self.image_height)) ** 2)) / 2.5)

    class PointA:
        def __int__(self):
            self.x = None
            self.y = None

    class PointB:
        def __int__(self):
            self.x = None
            self.y = None

index = 0



class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
    def transform(self, frame):
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                image = frame.to_ndarray(format="bgr24")
                image = cv2.resize(image, (800, 600))
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image_height, image_width, _ = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                results = pose.process(image)
                if results.pose_landmarks:
                    # ---------------- GENERATE IMAGE -----------------
                    trace_leg = [32, 30, 28, 26, 'A', 25, 27, 29, 31]
                    trace_body = ['A', 'B']
                    trace_hand = [20, 16, 14, 'B', 13, 15, 19]
                    trace_head = ['B', 0]
                    # init canvas
                    draw_img = Draw([image_height, image_width], [255, 255, 255, 0], (0, 0, 0, 255))
                    # print(image_height,image_width)
                    vertices = Vertices(image_width, image_height, results)
                    # draw leg
                    for i in range(len(trace_leg) - 1):
                        print(i)

                        start = (
                            int(vertices.get_vertices(trace_leg[i], results).x * image_width),
                            int(vertices.get_vertices(trace_leg[i], results).y * image_height)
                        )
                        
                        end = (
                            int(vertices.get_vertices(trace_leg[i + 1], results).x * image_width),
                            int(vertices.get_vertices(trace_leg[i + 1], results).y * image_height)
                        )
                        
                        draw_img.draw_line(start, end)
                        
                    # draw body
                    for i in range(len(trace_body) - 1):
                        start = (
                            int(vertices.get_vertices(trace_body[i], results).x * image_width),
                            int(vertices.get_vertices(trace_body[i], results).y * image_height)
                        )

                        end = (
                            int(vertices.get_vertices(trace_body[i + 1], results).x * image_width),
                            int(vertices.get_vertices(trace_body[i + 1], results).y * image_height)
                        )

                        draw_img.draw_line(start, end)
                    #getting length of the body
                        len_of_body = vertices.calculate_distance(vertices.get_vertices(trace_body[i], results).x, vertices.get_vertices(trace_body[i], results).y, vertices.get_vertices(trace_body[i + 1], results).x, vertices.get_vertices(trace_body[i + 1], results).y)
                        # print(len_of_body)
                        HEAD_RADIUS = int(len_of_body/1.5)
                    # draw hand
                    for i in range(len(trace_hand) - 1):
                        start = (
                            int(vertices.get_vertices(trace_hand[i], results).x * image_width),
                            int(vertices.get_vertices(trace_hand[i], results).y * image_height)
                        )

                        end = (
                            int(vertices.get_vertices(trace_hand[i + 1], results).x * image_width),
                            int(vertices.get_vertices(trace_hand[i + 1], results).y * image_height)
                        )

                        draw_img.draw_line(start, end)
                    
                    # draw head
                    for i in range(len(trace_head) - 1):
                        start = (
                            int(vertices.get_vertices(trace_head[i], results).x * image_width),
                            int(vertices.get_vertices(trace_head[i], results).y * image_height)
                        )

                        end = (
                            int(vertices.get_vertices(trace_head[i + 1], results).x * image_width),
                            int(vertices.get_vertices(trace_head[i + 1], results).y * image_height)
                        )

                        draw_img.draw_line(start, end)
                    
                    draw_img.draw_head(
                        (int(results.pose_landmarks.landmark[0].x * image_width), int(results.pose_landmarks.landmark[0].y * image_height)),
                        HEAD_RADIUS
                    )
                    
                    img = draw_img.generate()
                    img = img[:,:,:-1]
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # index += 1
                    
                    # cv2.imwrite(f'./frames/img{index}.png', img)
                    image = cv2.flip(image, 1)
                    cv2.imshow('Animation', img)
                    cv2.waitKey(1)
                    return img



ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer, video_frame_callback=False, audio_frame_callback=True)