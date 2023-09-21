import imp
import cv2
from PIL import Image
from utils.data_utils import Transforms


class TestVideoData(object):

    def __init__(self, config):

        self.vcap = cv2.VideoCapture(config['testing']['test_video'])
        self.transform = Transforms(config)
        self.transform.create_transforms_from_list(config['testing']['preprocess'])
        self.transform = self.transform.compose_transforms()

    def __del__(self):
        self.vcap.release()

    def get_len(self):
        return int(self.vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_item(self):
        return_dict = {}
        _, frame = self.vcap.read()
        frame = Image.fromarray(frame[:,:,::-1]).convert('RGB')
        return_dict['test_A'], return_dict['test_B'] = self.transform(frame, frame)
        return_dict['test_A_path'], return_dict['test_B_path'] = 'A.jpg', 'B.jpg'
        return return_dict
