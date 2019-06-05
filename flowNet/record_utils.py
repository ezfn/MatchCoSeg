import tensorflow as tf
from flowNet.data_utils import MatlabBasedCosegDataset,PickleBasedCosegDataset
import math
import random
import cv2
import os

def create_example(sample):
    ret, enc = cv2.imencode('.png', sample['patches'][0:3,:,:].transpose([1,2,0]), (cv2.IMWRITE_PNG_COMPRESSION, 3))
    encoded_pixel_data_src = enc.tobytes()
    ret, enc = cv2.imencode('.png', sample['patches'][3:6, :, :].transpose([1, 2, 0]), (cv2.IMWRITE_PNG_COMPRESSION, 3))
    encoded_pixel_data_tgt = enc.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'patches/src': bytes_feature(encoded_pixel_data_src),
        'patches/tgt': bytes_feature(encoded_pixel_data_tgt),
        'flow/factor_map_025': float_list_feature(list(sample['factor_map_025'].flatten())),
        'flow/factor_map_0125': float_list_feature(list(sample['factor_map_0125'].flatten())),
        'flow/gt_flow_025': float_list_feature(list(sample['gt_flow_025'].flatten())),
        'flow/gt_flow_0125': float_list_feature(list(sample['gt_flow_0125'].flatten())),
        'meta/path': bytes_feature(bytes(sample['path'], 'utf-8')),
        'meta/sampled_idx': int64_list_feature([sample['sampled_idxs']]),
    }))
    return example


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_pickled_to_tf_record(root_dir, example_format, do_conf_factor=True, buffer_size=10000):
    record_writer = tf.python_io.TFRecordWriter(os.path.join(root_dir, 'all_data_tf.record'))

    dataset = PickleBasedCosegDataset(root_dir=root_dir, example_format=example_format, max_batch_size=math.inf,
                                      transform_list=[], do_normalize_image=False)
    sample = dataset[0]
    tmp_dict = dict.fromkeys(sample.keys(), 0)
    example_list = []
    ctr = 0
    for file_idx, sample in enumerate(dataset):
        N = len(sample['sampled_idxs'])
        for key in tmp_dict.keys():
            if key == 'path':
                tmp_dict[key] = [sample[key]] * N
            else:
                tmp_dict[key] = list(sample[key])
        for idx in range(N):
            single_sample = dict.fromkeys(tmp_dict.keys(), 0)
            for key in single_sample.keys():
                single_sample[key] = tmp_dict[key][idx]
            example_list.append(single_sample)

        N_total = len(example_list)
        num_to_draw = N_total - buffer_size
        if num_to_draw > 0:
            for k in range(num_to_draw):
                drawn_idx = random.randint(0,len(example_list)-1)
                single_sample = example_list[drawn_idx]
                example = create_example(single_sample)
                record_writer.write(example.SerializeToString())
                ctr += 1
                if not ctr % 1000:
                    print('recorded {} examples. {}/{} files'.format(ctr, file_idx, len(dataset)))
                del example_list[drawn_idx]  # clear from buffer

    # flush the rest to the end of the record file
    random.shuffle(example_list)
    for drawn_idx in range(len(example_list)):
        single_sample = example_list[drawn_idx]
        example = create_example(single_sample)
        record_writer.write(example.SerializeToString())
        ctr += 1
        if not ctr % 1000:
            print('recorded {} examples. {}/{} files'.format(ctr, file_idx, len(dataset)))
    record_writer.close()


if __name__ == '__main__':
    convert_pickled_to_tf_record(root_dir='/media/fastData/coSegDataPasses/SintelCleanPasses',
    example_format='affnet_*/*/*.pklz', buffer_size=2000)