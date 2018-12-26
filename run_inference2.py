from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf
from PIL import Image
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

fout = open(r'data/mscoco/raw-data/val2014/picture_description.txt', 'a+')

# 检索所有图像
# filedir = "C:\\Users\\lsz95\\Desktop\\图像"
filedir = "data/mscoco/raw-data/val2014/"
picture_dir = []
for root, dirs, files in os.walk(filedir):
    for file in files:
        print(os.path.join(root, file))
        picture_dir.append(os.path.join(root, file))
print(picture_dir)

# tf.flags.DEFINE_string("input_files", "data/mscoco/raw-data/val2014/COCO_val2014_000000003832.jpg",
#                        "File pattern or comma-separated list of file patterns "
#                        "of image files.")
tf.flags.DEFINE_string("input_files", picture_dir,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("checkpoint_path", "data/mscoco/train",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/mscoco/raw-data/word_counts.txt", "Text file containing the vocabulary.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    #

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    filenames = []
    for file_pattern in FLAGS.input_files:
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), FLAGS.input_files)
    print(filenames)
    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        j = 0
        for filename in filenames:
            try:
                # picture_name= filename.split("\\")[-1].split(".")[0].split("_")[0][4:]
                # picture_name = filename.split("\\")[-1].split(".")[0].split("_")[0]
                picture_name = filename.split("\\")[-1].split(".")[0]
                j += 1
                with tf.gfile.FastGFile(filename, "rb") as f:
                    image = f.read()
                captions = generator.beam_search(sess, image)
                # print("Captions for image %s:" % os.path.basename(filename))
                for i, caption in enumerate(captions):
                    # Ignore begin and end words.
                    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                print("   %d)  %s  %s \n" % (j, picture_name, sentence))
                fout.write("%s||%s \n" % (picture_name, sentence))

                # print("   %d) %s %s  %s \n" % (j, picture_name,picture_name2,sentence))
                # fout.write("%s $|$ %s $|$ %s \n" % (picture_name,picture_name2, sentence))
            except:
                # fout.write("fail")
                print("------------")
        #     图像加描述图形显示
        #     img = Image.open(filename)
        #
        #     plt.subplot(2,2,j)
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.title(str(sentence))
        # plt.show()
        fout.close()


if __name__ == "__main__":
    tf.app.run()

