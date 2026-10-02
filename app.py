"""
app.py - Real-time hand-gesture (sign alphabet) recognition from a webcam.

Show a hand sign inside the red box. The predicted letter and its confidence
are drawn on the video. If the same letter is predicted several times in a row
it is appended to the text shown in the "sequence" window.

Special labels:
    space   -> adds a space to the sequence
    del     -> deletes the last character
    nothing -> ignored

Keys:  ESC = quit    c = clear the sequence

Works with TensorFlow 1.14+ and TensorFlow 2.x.
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

INPUT_TENSOR = 'DecodeJpeg/contents:0'
OUTPUT_TENSOR = 'final_result:0'


def load_labels(path):
    with tf.io.gfile.GFile(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_graph(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def predict(sess, softmax_tensor, labels, image_bytes):
    """Return (best_label, score) for one JPEG-encoded image."""
    scores = sess.run(softmax_tensor, {INPUT_TENSOR: image_bytes})[0]
    best = int(np.argmax(scores))
    return labels[best], float(scores[best])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--graph', default='tf_files/retrained_graph.pb')
    p.add_argument('--labels', default='tf_files/retrained_labels.txt')
    p.add_argument('--camera', type=int, default=0, help='webcam index')
    p.add_argument('--predict_every', type=int, default=5,
                   help='run the model on every N-th frame')
    p.add_argument('--hold', type=int, default=2,
                   help='consecutive identical predictions needed to accept '
                        'a letter')
    p.add_argument('--roi', type=int, nargs=4, default=[100, 100, 350, 350],
                   metavar=('X1', 'Y1', 'X2', 'Y2'),
                   help='box in which the hand is shown')
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.graph, args.labels):
        if not os.path.exists(path):
            sys.exit('Missing "%s". Train a model first with retrain.py '
                     '(see README).' % path)

    labels = load_labels(args.labels)
    graph = load_graph(args.graph)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit('Could not open camera %d.' % args.camera)

    x1, y1, x2, y2 = args.roi
    res, score = '', 0.0
    frame_count = 0
    prev = ''
    consecutive = 0
    sequence = ''

    with tf.Session(graph=graph) as sess:
        softmax_tensor = sess.graph.get_tensor_by_name(OUTPUT_TENSOR)

        while True:
            ok, img = cap.read()
            if not ok:
                print('Camera frame not received - exiting.')
                break
            img = cv2.flip(img, 1)  # mirror view
            crop = img[y1:y2, x1:x2]
            image_bytes = cv2.imencode('.jpg', crop)[1].tobytes()

            key = cv2.waitKey(1) & 0xFF

            if frame_count == args.predict_every - 1:
                res, score = predict(sess, softmax_tensor, labels,
                                     image_bytes)
                frame_count = 0

                consecutive = consecutive + 1 if res == prev else 0
                prev = res

                if consecutive == args.hold and res != 'nothing':
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0
            else:
                frame_count += 1

            # Draw UI
            cv2.putText(img, res.upper(), (100, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
            cv2.putText(img, '( Accuracy score = %.5f)' % score, (100, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('img', img)

            seq_img = np.zeros((200, 1200, 3), np.uint8)
            cv2.putText(seq_img, sequence.upper(), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.imshow('sequence', seq_img)

            if key == 27:      # ESC
                break
            if key == ord('c'):
                sequence = ''

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
