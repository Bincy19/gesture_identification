"""
collect_data.py - Record your own training images from a webcam.

    python collect_data.py --label a

Show the sign inside the red box, then:
    SPACE = start / stop automatic capture (one image every few frames)
    s     = save a single image
    q/ESC = quit

Images are saved to dataset/<label>/. Record 'nothing' (empty box) and
'space' / 'del' signs too if you want those classes.
"""
import argparse
import os
import sys
import time

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--label', required=True, help='class name, e.g. a')
    p.add_argument('--out_dir', default='dataset')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--every', type=int, default=3,
                   help='auto-capture every N frames')
    p.add_argument('--roi', type=int, nargs=4, default=[100, 100, 350, 350],
                   metavar=('X1', 'Y1', 'X2', 'Y2'))
    args = p.parse_args()

    folder = os.path.join(args.out_dir, args.label)
    os.makedirs(folder, exist_ok=True)
    count = len([f for f in os.listdir(folder) if f.endswith('.jpg')])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit('Could not open camera %d.' % args.camera)

    x1, y1, x2, y2 = args.roi
    auto, frame_no = False, 0

    def save(crop):
        nonlocal count
        name = '%s_%d_%d.jpg' % (args.label, int(time.time()), count)
        cv2.imwrite(os.path.join(folder, name), crop)
        count += 1

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.flip(img, 1)
        crop = img[y1:y2, x1:x2].copy()
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            auto = not auto
        elif key == ord('s'):
            save(crop)
        elif key in (ord('q'), 27):
            break

        frame_no += 1
        if auto and frame_no % args.every == 0:
            save(crop)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        status = 'AUTO' if auto else 'paused'
        cv2.putText(img, '%s [%s]  saved: %d' % (args.label, status, count),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('collect', img)

    cap.release()
    cv2.destroyAllWindows()
    print('Saved %d images in %s' % (count, folder))


if __name__ == '__main__':
    main()
