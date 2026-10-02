"""
retrain.py - Train the final layer of Inception v3 on gesture images
(transfer learning), in the style of "TensorFlow for Poets".

Works with TensorFlow 1.14+ and TensorFlow 2.x (via tensorflow.compat.v1).

Expected dataset layout (one folder per class):
    dataset/
        a/ *.jpg
        b/ *.jpg
        ...
        space/ *.jpg
        del/ *.jpg
        nothing/ *.jpg

Outputs:
    tf_files/retrained_graph.pb      (frozen graph, output node: final_result)
    tf_files/retrained_labels.txt    (one label per line)

Example:
    python retrain.py --image_dir dataset --max_images_per_class 500
"""
import argparse
import hashlib
import os
import random
import re
import sys
import tarfile
import urllib.request

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

MODEL_URL = ('http://download.tensorflow.org/models/image/imagenet/'
             'inception-2015-12-05.tgz')
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.JPG', '.JPEG')

_bottleneck_memory = {}  # in-RAM cache so training steps don't re-read disk


# ----------------------------------------------------------------------------
# Model download / loading
# ----------------------------------------------------------------------------
def maybe_download_and_extract(model_dir):
    """Download and extract the pretrained Inception v3 model if missing."""
    graph_file = os.path.join(model_dir, 'classify_image_graph_def.pb')
    if os.path.exists(graph_file):
        return
    os.makedirs(model_dir, exist_ok=True)
    archive = os.path.join(model_dir, MODEL_URL.split('/')[-1])
    if not os.path.exists(archive):
        print('Downloading %s ...' % MODEL_URL)
        try:
            urllib.request.urlretrieve(MODEL_URL, archive)
        except Exception as e:  # noqa
            sys.exit('Download failed (%s).\nDownload the file manually from\n'
                     '  %s\nand place it in "%s/", then re-run.'
                     % (e, MODEL_URL, model_dir))
    print('Extracting %s ...' % archive)
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(model_dir)


def create_inception_graph(model_dir):
    """Load the pretrained graph; return graph, bottleneck and jpeg tensors."""
    with tf.Graph().as_default() as graph:
        path = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with tf.io.gfile.GFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        bottleneck, jpeg_data = tf.import_graph_def(
            graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    return graph, bottleneck, jpeg_data


# ----------------------------------------------------------------------------
# Dataset handling
# ----------------------------------------------------------------------------
def create_image_lists(image_dir, testing_pct, validation_pct, max_per_class):
    """Split the images in every sub-folder into train / test / validation."""
    if not os.path.isdir(image_dir):
        print('Image directory "%s" not found.' % image_dir)
        return None

    result = {}
    for folder_name in sorted(os.listdir(image_dir)):
        folder = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder)
                       if f.endswith(IMAGE_EXTENSIONS))
        if not files:
            continue
        if max_per_class > 0:
            files = files[:max_per_class]

        label = re.sub(r'[^a-z0-9]+', ' ', folder_name.lower()).strip()
        training, testing, validation = [], [], []
        for name in files:
            # Hash of the filename -> an image always stays in the same split
            hash_name = re.sub(r'_nohash_.*$', '', name)
            digest = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
            pct = ((int(digest, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                   (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if pct < validation_pct:
                validation.append(name)
            elif pct < (testing_pct + validation_pct):
                testing.append(name)
            else:
                training.append(name)
        result[label] = {'dir': folder_name, 'training': training,
                         'testing': testing, 'validation': validation}
    return result


def get_image_path(lists, image_dir, label, index, category):
    items = lists[label][category]
    return os.path.join(image_dir, lists[label]['dir'],
                        items[index % len(items)])


def get_bottleneck_path(lists, bottleneck_dir, label, index, category):
    items = lists[label][category]
    folder = os.path.join(bottleneck_dir, lists[label]['dir'])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, items[index % len(items)]) + '.npy'


def get_bottleneck(sess, lists, args, label, index, category,
                   jpeg_data, bottleneck):
    """Return the 2048-d feature vector for one image (cached on disk+RAM)."""
    path = get_bottleneck_path(lists, args.bottleneck_dir,
                               label, index, category)
    if path in _bottleneck_memory:
        return _bottleneck_memory[path]
    if os.path.exists(path):
        values = np.load(path)
    else:
        img_path = get_image_path(lists, args.image_dir, label, index,
                                  category)
        with tf.io.gfile.GFile(img_path, 'rb') as f:
            data = f.read()
        values = np.squeeze(sess.run(bottleneck, {jpeg_data: data}))
        np.save(path, values)
    _bottleneck_memory[path] = values
    return values


def cache_bottlenecks(sess, lists, args, jpeg_data, bottleneck):
    """Compute features for every image once (the slow part of training)."""
    total = sum(len(d[c]) for d in lists.values()
                for c in ('training', 'testing', 'validation'))
    done = 0
    for label, d in lists.items():
        for category in ('training', 'testing', 'validation'):
            for i in range(len(d[category])):
                get_bottleneck(sess, lists, args, label, i, category,
                               jpeg_data, bottleneck)
                done += 1
                if done % 200 == 0 or done == total:
                    print('  bottlenecks: %d / %d' % (done, total))


def get_batch(sess, lists, args, how_many, category, jpeg_data, bottleneck):
    """Random batch (how_many >= 0) or the whole category (how_many < 0)."""
    labels = list(lists.keys())
    class_count = len(labels)
    features, truths = [], []

    def add(label_index, image_index):
        features.append(get_bottleneck(sess, lists, args, labels[label_index],
                                       image_index, category,
                                       jpeg_data, bottleneck))
        t = np.zeros(class_count, dtype=np.float32)
        t[label_index] = 1.0
        truths.append(t)

    if how_many >= 0:
        for _ in range(how_many):
            add(random.randrange(class_count),
                random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1))
    else:
        for li, label in enumerate(labels):
            for ii in range(len(lists[label][category])):
                add(li, ii)
    return features, truths


# ----------------------------------------------------------------------------
# New final layer
# ----------------------------------------------------------------------------
def add_final_training_ops(class_count, learning_rate):
    bottleneck_input = tf.placeholder_with_default(
        tf.zeros([1, BOTTLENECK_SIZE]), shape=[None, BOTTLENECK_SIZE],
        name='BottleneckInputPlaceholder')
    truth_input = tf.placeholder(tf.float32, [None, class_count],
                                 name='GroundTruthInput')

    weights = tf.Variable(
        tf.truncated_normal([BOTTLENECK_SIZE, class_count], stddev=0.001),
        name='final_weights')
    biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    logits = tf.matmul(bottleneck_input, weights) + biases
    final_tensor = tf.nn.softmax(logits, name='final_result')

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=truth_input, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss)
    return train_step, loss, bottleneck_input, truth_input, final_tensor


def add_evaluation_step(final_tensor, truth_input):
    correct = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(truth_input, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    maybe_download_and_extract(args.model_dir)
    graph, bottleneck, jpeg_data = create_inception_graph(args.model_dir)

    lists = create_image_lists(args.image_dir, args.testing_percentage,
                               args.validation_percentage,
                               args.max_images_per_class)
    if not lists:
        sys.exit('No valid image folders found in "%s".' % args.image_dir)
    class_count = len(lists)
    if class_count == 1:
        sys.exit('Only one class found - multiple classes are required.')
    print('Found %d classes: %s' % (class_count, ', '.join(lists.keys())))

    with tf.Session(graph=graph) as sess:
        print('Computing bottleneck features (cached after the first run)...')
        cache_bottlenecks(sess, lists, args, jpeg_data, bottleneck)

        (train_step, loss, bottleneck_input, truth_input,
         final_tensor) = add_final_training_ops(class_count,
                                                args.learning_rate)
        accuracy = add_evaluation_step(final_tensor, truth_input)
        sess.run(tf.global_variables_initializer())

        for step in range(args.how_many_training_steps):
            feats, truths = get_batch(sess, lists, args, args.train_batch_size,
                                      'training', jpeg_data, bottleneck)
            sess.run(train_step, {bottleneck_input: feats,
                                  truth_input: truths})

            is_last = (step + 1 == args.how_many_training_steps)
            if step % args.eval_step_interval == 0 or is_last:
                tr_acc, tr_loss = sess.run(
                    [accuracy, loss],
                    {bottleneck_input: feats, truth_input: truths})
                vf, vt = get_batch(sess, lists, args,
                                   args.validation_batch_size, 'validation',
                                   jpeg_data, bottleneck)
                v_acc = sess.run(accuracy, {bottleneck_input: vf,
                                            truth_input: vt})
                print('Step %4d | train acc %5.1f%% | loss %.4f | '
                      'validation acc %5.1f%%'
                      % (step, tr_acc * 100, tr_loss, v_acc * 100))

        tf_feats, tf_truths = get_batch(sess, lists, args,
                                        args.test_batch_size, 'testing',
                                        jpeg_data, bottleneck)
        test_acc = sess.run(accuracy, {bottleneck_input: tf_feats,
                                       truth_input: tf_truths})
        print('Final test accuracy = %.1f%% (N = %d)'
              % (test_acc * 100, len(tf_feats)))

        out_dir = os.path.dirname(args.output_graph)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        frozen = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['final_result'])
        with tf.io.gfile.GFile(args.output_graph, 'wb') as f:
            f.write(frozen.SerializeToString())
        with tf.io.gfile.GFile(args.output_labels, 'w') as f:
            f.write('\n'.join(lists.keys()) + '\n')

    print('Saved %s and %s' % (args.output_graph, args.output_labels))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--image_dir', default='dataset',
                   help='folder with one sub-folder of images per class')
    p.add_argument('--output_graph', default='tf_files/retrained_graph.pb')
    p.add_argument('--output_labels', default='tf_files/retrained_labels.txt')
    p.add_argument('--bottleneck_dir', default='tf_files/bottlenecks')
    p.add_argument('--model_dir', default='inception')
    p.add_argument('--how_many_training_steps', type=int, default=4000)
    p.add_argument('--learning_rate', type=float, default=0.01)
    p.add_argument('--testing_percentage', type=int, default=10)
    p.add_argument('--validation_percentage', type=int, default=10)
    p.add_argument('--eval_step_interval', type=int, default=100)
    p.add_argument('--train_batch_size', type=int, default=100)
    p.add_argument('--test_batch_size', type=int, default=-1,
                   help='-1 = use the whole test split')
    p.add_argument('--validation_batch_size', type=int, default=100)
    p.add_argument('--max_images_per_class', type=int, default=0,
                   help='cap images per class for quick runs (0 = use all)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
