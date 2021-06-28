import errno
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def video2frames(video_path, output_path):
    '''
    
    :param video_path: 비디오 경로
    :param output_path: 프레임을 저장할 경로
    :return: 
    '''
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret is True:
            OUTPUT_IMAGE_PATH = os.path.join(output_path, 'image_%09d.jpg' % (cnt))
            print("Now %d-th images being processed..." % (cnt))
            plt.imsave(OUTPUT_IMAGE_PATH, frame)
        else:
            break

        cnt += 1

    cap.release()

def frames2video(frames_path, output_path, video_file_name):
    '''
    
    :param frames_path: 프레임이 저장되어 있는 경로 
    :param output_path: 비디오를 저장할 경로
    :param video_file_name: 비디오 파일명
    :return: 
    '''
    try:
        if os.path.isdir(frames_path):
            for root, dirs, files in os.walk(frames_path, topdown=False):

                b_is_first = True
                for name in files:
                    cur_file = os.path.join(frames_path, name)
                    cur_img = cv2.imread(cur_file)

                    print("Currently %s being processed..." % (cur_file))

                    if type(cur_img) == np.ndarray:
                        if b_is_first:
                            frame_height = cur_img.shape[0]
                            frame_width = cur_img.shape[1]

                            video_file = os.path.join(output_path, video_file_name)
                            out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                                  (frame_width, frame_height))

                        out.write(cur_img)

                    b_is_first = False

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    out.release()
