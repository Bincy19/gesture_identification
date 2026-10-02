# Gesture Identification using Machine Learning

Real-time recognition of hand-sign alphabet gestures (A–Z, `space`, `del`, `nothing`) from a webcam, using **transfer learning on Inception v3** with **TensorFlow** and **OpenCV**.

B.Tech final-year project, Department of Computer Science and Engineering, Mount Zion Institute of Science and Technology (APJ Abdul Kalam Technological University), 2020.
By **Bincy Annamma Saji** and **N C Chanjal**. Guide: Ms. Ruhin Mary Saji.

---

## How it works

```
 Training                                   Live recognition
 --------                                   ----------------
 Image dataset (one folder per letter)      Webcam frame
        |                                         |
 Pretrained Inception v3 (frozen)           Crop the hand box (ROI)
        |  2048-d "bottleneck" features           |
 New softmax layer (trained)                Inception v3 + trained layer
        |                                         |
 retrained_graph.pb + retrained_labels.txt  Predicted letter + confidence
                                                  |
                                            Letters accepted when held
                                            steadily -> text sequence
```

1. **`retrain.py`** downloads Inception v3, extracts a 2048-value feature vector for every training image, and trains a new final softmax layer on those features. It writes a frozen graph (`retrained_graph.pb`) and the class list (`retrained_labels.txt`).
2. **`app.py`** reads the webcam, crops a fixed box, runs the model on every 5th frame, and shows the predicted letter with its score. A letter is appended to the on-screen text after it is predicted several times in a row; the `space` and `del` gestures add a space or delete the last character.

> **Note on scope:** the project report mentions an SSD stage. This implementation classifies the fixed cropped region directly with Inception v3; it does **not** run an SSD object detector. The hand must therefore be shown inside the red box.

## Repository layout

```
.
├── app.py              # live webcam recognition
├── retrain.py          # transfer-learning training script
├── collect_data.py     # record your own images from a webcam
├── train.sh            # quick-start: train (500 imgs/class) then run the demo
├── requirements.txt
├── tf_files/           # trained model and labels are written here
└── dataset/            # you create this (not committed)
```

## Setup

Requires Python 3.7+ and a webcam.

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The scripts run on both **TensorFlow 1.14** (the original environment: Python 3.7, Ubuntu) and **TensorFlow 2.x**, using `tensorflow.compat.v1`.

## Dataset

The report trains on an alphabet-gesture image dataset downloaded from Kaggle (about 3000 images per class; classes A–Z plus `del`, `nothing`, `space`). A dataset with this layout is the **ASL Alphabet** dataset on Kaggle. Download it and arrange the images like this (folder names are case-insensitive):

```
dataset/
├── A/ ... .jpg
├── B/ ... .jpg
├── ...
├── Z/
├── del/
├── nothing/
└── space/
```

Please check the dataset's own license and terms on Kaggle before redistributing it. This repo does not include the images.

**Or record your own** (works best when training and demo conditions match: same camera, background and lighting):

```bash
python collect_data.py --label a      # SPACE = auto-capture, s = one image, q = quit
python collect_data.py --label b
python collect_data.py --label nothing   # empty box
```

## Train

Quick run (about 500 images per class):

```bash
python retrain.py --image_dir dataset --max_images_per_class 500
```

Full run on the whole dataset (the report mentions roughly 8–10 hours on a CPU laptop, mostly for the one-time feature extraction):

```bash
python retrain.py --image_dir dataset
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--image_dir` | `dataset` | folder with one sub-folder per class |
| `--how_many_training_steps` | 4000 | training iterations |
| `--learning_rate` | 0.01 | learning rate of the new layer |
| `--max_images_per_class` | 0 (all) | cap images per class for faster runs |
| `--testing_percentage` / `--validation_percentage` | 10 / 10 | split sizes |
| `--model_dir` | `inception` | where Inception v3 is stored |

Feature vectors are cached in `tf_files/bottlenecks/`, so later runs are fast. Training ends by printing the **test accuracy** and saving:

```
tf_files/retrained_graph.pb
tf_files/retrained_labels.txt
```

If the automatic Inception download fails, download
`http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz`
manually, put it in `inception/`, and re-run.

## Run the live demo

```bash
python app.py
```

- Put your hand sign inside the **red box**.
- The big text is the predicted letter; the line below it is the confidence score.
- Hold a sign steadily to add it to the text in the **sequence** window.
- Keys: `ESC` quit, `c` clear the text.

Options: `--camera 1` (other webcam), `--hold 3` (hold longer before a letter is accepted), `--predict_every 3`, `--roi X1 Y1 X2 Y2`, `--graph`, `--labels`.

## Tips for better accuracy

- Use a plain background and good, even lighting.
- Keep the hand centred in the box and about the same size as in the training images.
- Train on images from your own camera (`collect_data.py`) mixed with the Kaggle set.
- Add a good number of `nothing` images (empty box, face, background) to reduce false letters.
- Similar signs (e.g. M/N, U/V) are the most commonly confused.

## Results

Add your measured numbers here after training, for example:

| Metric | Value |
|---|---|
| Classes | 29 |
| Training images | _fill in_ |
| Test accuracy | _fill in (printed by `retrain.py`)_ |

## Limitations

- Static single-hand signs only; no motion/dynamic gestures.
- Fixed region of interest, no hand detection or tracking.
- Accuracy depends on camera quality, background and lighting, and drops on conditions unlike the training data.
- Uses the 2015 Inception v3 graph and the TF1-style graph API, kept for compatibility with the original project.

## Future work

- Real hand detection (e.g. an SSD or other detector) so the hand can be anywhere in the frame.
- Data augmentation and fine-tuning of deeper layers.
- Dynamic gestures (video) and a cloud-hosted version.

## References

The project report cites ten papers on sEMG, accelerometer, RGB-D and CNN/RNN based gesture recognition; see the report for the full list.

## License

MIT. See `LICENSE`.
