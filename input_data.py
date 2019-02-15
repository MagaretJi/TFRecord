import tensorflow as tf
import os
import sys
from PIL import Image
import numpy as np

def get_files(file_dir,is_random=True):
    image_list=[]
    label_list=[]
    dog_count=0
    cat_count=0
    for file in os.listdir(file_dir):
        name=file.split(sep='.')
        if(name[0]=='cat'):
            image_list.append(file_dir+file)
            label_list.append(0)
            cat_count+=1
        else:
            image_list.append(file_dir+file)
            label_list.append(1)
            dog_count+=1
    print('%d cats and %d dogs'%(cat_count,dog_count))

    image_list=np.asarray(image_list)
    label_list=np.asarray(label_list)

    if is_random:
        rnd_index=np.arange(len(image_list))
        np.random.shuffle(rnd_index)
        image_list=image_list[rnd_index]
        label_list=label_list[rnd_index]

    return image_list,label_list

file_dir='F:/train/train/'
image_list,label_list=get_files(file_dir,is_random=True)
print(image_list)
print(label_list)


def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def image_to_tfexample(image_data, label,size):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': int64_feature(label),
        'image_width':int64_feature(size[0]),
        'image_height':int64_feature(size[1])
    }))


def _convert_dataset(image_list, label_list, tfrecord_dir):
    """ Convert data to TFRecord format. """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if not os.path.exists(tfrecord_dir):
                os.makedirs(tfrecord_dir)
            output_filename = os.path.join(tfrecord_dir, "train.tfrecord")
            tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
            length = len(image_list)
            for i in range(length):
                # 图像数据
                image_data = Image.open(image_list[i],'r')

                size = image_data.size
                image_data = image_data.tobytes()
                label = label_list[i]
                example = image_to_tfexample(image_data, label,size)
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, length))
                sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()


#tfrecord_dir='F:/'
#_convert_dataset(image_list,label_list,tfrecord_dir)

def read_and_decode(tfrecord_path):
    data_files = tf.gfile.Glob(tfrecord_path)  #data_path为TFRecord格式数据的路径
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label':tf.FixedLenFeature([],tf.int64),
                                           'image':tf.FixedLenFeature([],tf.string),
                                           'image_width': tf.FixedLenFeature([],tf.int64),
                                           'image_height': tf.FixedLenFeature([],tf.int64),
                                       })

    image = tf.decode_raw(features['image'],tf.uint8)
    image_width = tf.cast(features['image_width'],tf.int32)
    image_height = tf.cast(features['image_height'],tf.int32)
    image = tf.reshape(image,[image_height,image_width,3])
    label = tf.cast(features['label'], tf.int32)
    return image,label

def batch(image,label):
    # Load training set.
    image = tf.image.resize_images(image, [128, 128])
    with tf.name_scope('input_train'):
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=30, capacity=2000,
                                                        min_after_dequeue=1500)
    return image_batch, label_batch

def train():
    tfrecord_path='F:/train.tfrecord'
    image,label=read_and_decode(tfrecord_path)

    image_batch, label_batch=batch(image,label)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for i in range (10):
            single, l = sess.run([image_batch,label_batch])

            print(single.shape, l)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    train()
