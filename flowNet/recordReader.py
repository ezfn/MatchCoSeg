import tensorflow as tf
import numpy as np
import cv2


class recordReader():

    def __init__(self,filename,example_dict):
        self.tf_record_path = filename
        self.example_dict = example_dict
        self._open_record()

    def get_next_example_parsed_non_safe(self):
        se = next(self.tf_record_iterator)
        return self._serialized_example_to_image(se)

    def _open_record(self):
        self.tf_record_iterator = tf.python_io.tf_record_iterator(path=self.tf_record_path)

    def get_next_example_parsed(self):
        try:
            se = next(self.tf_record_iterator)
        except StopIteration:
            self._open_record()
            se = next(self.tf_record_iterator)
        except:
            self._open_record()
            se = next(self.tf_record_iterator)
        return self._serialized_example_to_image(se)

    def _serialized_example_to_image(self, serialized_example):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        parsed_example = dict.fromkeys(self.example_dict.keys(),0)
        for key in self.example_dict.keys():
            if key in ['src_patch','tgt_patch']:
                parsed_example[key] = (cv2.imdecode(np.fromstring(example.features.feature[self.example_dict[key]].bytes_list.value[0],
                                           np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float) - 127)/128
            elif key in ['gt_flow_0125','gt_flow_025']:
                floats = example.features.feature[self.example_dict[key]].float_list.value
                dim = int(np.sqrt(len(floats)/3))
                parsed_example[key] = np.reshape(np.array(floats), (3,dim,dim))
            elif key in ['factor_map_0125', 'factor_map_025']:
                floats = example.features.feature[self.example_dict[key]].float_list.value
                dim = int(np.sqrt(len(floats)))
                parsed_example[key] = np.reshape(np.array(floats), (dim, dim))
            elif key == 'path':
                parsed_example[key] = example.features.feature[self.example_dict[key]].bytes_list.value[0].decode("utf-8")
            elif key == 'sampled_idxs':
                parsed_example[key] = example.features.feature[self.example_dict[key]].int64_list.value[0]
        return parsed_example


# record_file = '/media/fastData/records/IAI/classification_data_3backTrueCA3Speed_maskedApproved_eval_F27404.record'
# viewer = record_viewer(filename=record_file)
# while True:
#     viewer.draw_next()