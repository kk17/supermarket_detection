import cv2
import numpy as np
import mtcnn
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import sys
import os
import argparse
from stopwatch import Stopwatch
from facenet.architecture import *
from facenet.preprocessing import normalize,l2_normalizer
import queue
import threading

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

confidence_t=0.99
recognition_t=0.5
required_size = (160,160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def draw_bounding_box(img, pt_1, pt_2, name, distance):
    if name != 'unknown':
        cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
        cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 200, 200), 2)
    return img

def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    pt_1, pt_2, distance = None, None, None
    name = 'unknown'

    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
    return pt_1, pt_2, name, distance


class DetectThread(threading.Thread):
    def __init__(self, face_detector, face_encoder, encoding_dict, frame_queue,
                 result_queue):
        super().__init__()
        self.face_detector = face_detector
        self.face_encoder = face_encoder 
        self.encoding_dict = encoding_dict 
        self.frame_queue = frame_queue
        self.result_queue = result_queue 
        self.stop_event = threading.Event()

    def run(self):
        stopwatch = Stopwatch()
        while not self.stop_event.is_set():
            frame, n = self.frame_queue.get(timeout=1)
            if frame is None:
                logging.info(f'timeout waiting for frame')
                continue
            stopwatch.restart()
            logging.info(f'Start detect frame: {n}')
            pt_1, pt_2, name, distance  = detect(frame , self.face_detector , self.face_encoder , self.encoding_dict)
            logging.info(f'Finished detect frame: {n}\n\tUsed time: {stopwatch}')
            self.result_queue.put((pt_1, pt_2, name, distance, n))
        logging.info(f'Stoped detection')

    def stop(self):
        self.stop_event.set()
        logging.info(f'Stopping detection')

def main():
    face_encoder = InceptionResNetV2()
    path_m = "facenet/facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'facenet/encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputpath",
                        "-i",
                        type=str,
                        default='')
    parser.add_argument("--outputpath",
                        "-o",
                        type=str,
                        default='')
    parser.add_argument("--detection_interval",
                        "-d",
                        type=float,
                        default=0.5)
    parser.add_argument("--show",
                        "-s",
                        action="store_true")
    parser.add_argument("--multi_thread",
                        "-m",
                        action="store_true")
    args = parser.parse_args()

    stopwatch = Stopwatch()
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    logging.info(f'Using CV version {major_ver}.{minor_ver}.{subminor_ver}')

    logging.info('Opening video')
    if args.inputpath == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.inputpath)

    if int(major_ver) < 3:
        fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        logging.info("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".
              format(fps))
    else:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        logging.info(
            "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(
                fps))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f'Video size : {width} x {height}')

    writer = None

    if args.outputpath != '':
        writer = cv2.VideoWriter(args.outputpath, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

    detect_every_n_frame = round(fps * args.detection_interval)
    logging.info(f'Running face detection every {detect_every_n_frame} frames')

    logging.info('Initiating model')
    stopwatch.start()
    # make a detection using a fake image to initiate the model
    fake_iamge_np = np.zeros((height, width, 3)).astype('float32')
    pt_1, pt_2, name, distance = detect(fake_iamge_np , face_detector , face_encoder , encoding_dict)
    logging.info(f'model initiated time: {stopwatch}')
    stopwatch.stop()

    f = 0
    stopwatch_all = Stopwatch()
    stopwatch_all.start()

    if args.multi_thread:
        frame_queue = queue.Queue(maxsize=1)
        result_queue = queue.Queue(maxsize=1)
        detect_thread = DetectThread(face_detector , face_encoder , encoding_dict, frame_queue, result_queue)
        detect_thread.start()
        _, frame = cap.read()
        frame_queue.put((frame, f))
        f += 1
        pt_1, pt_2, name, distance = None, None, 'unknown', None

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        key = cv2.waitKey(1)

        if not ret:
            if args.multi_thread:
                detect_thread.stop()
            logging.info(f'Completed for all frames in the video. Total used time: {stopwatch_all}')
            break

        if args.multi_thread:
            if not result_queue.empty():
                pt_1, pt_2, name, distance, n = result_queue.get()
                logging.info(f'Got detection from frame: {n}, current frame: {f}, {f-n} frames left behind')
                frame_queue.put((frame, f))
            frame = draw_bounding_box(frame, pt_1, pt_2, name, distance)
        else:
            if f % detect_every_n_frame == 0:
                stopwatch.restart()
                logging.info(f'Start detect frame: {f}')
                pt_1, pt_2, name, distance = detect(frame , face_detector , face_encoder , encoding_dict)
                frame = draw_bounding_box(frame, pt_1, pt_2, name, distance)
                logging.info(f'Finished detect frame: {f}\n\tUsed time: {stopwatch}')
            else:
                frame = draw_bounding_box(frame, pt_1, pt_2, name, distance)

        if args.show:
            cv2.imshow('camera', frame)

        if writer is not None:
            writer.write(frame)
        # cv2.waitKey(int(1/fps*1000))
        if key & 0xFF == ord('q'):
            logging.info(f'face_detect.py terminated using \'q\' key. Total used time: {stopwatch_all}')
            break
        f+=1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()