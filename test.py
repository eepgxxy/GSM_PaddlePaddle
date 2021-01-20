import sys
import logging
import argparse
import ast
import numpy as np
import paddle

from model import InceptionV3GSM

import datasets_video
from dataset import VideoDataset

from paddle.vision import transforms as T
from transforms import GroupNdarray, GroupCenterCrop

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='diving48',
        help='name of dataset to train.')
    parser.add_argument(
        '--seg_num',
        type=int,
        default=16,
        help='number of segments for video')
    parser.add_argument(
        '--seg_len',
        type=int,
        default=1,
        help='number of segments for video')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--modality',
        type=str,
        default='RGB',
        help='Video frame mode.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--model_arc',
        type=str,
        default='inception',
        help='model architecture used for training')

    args = parser.parse_args()
    return args


def eval(args):

    if args.dataset == 'something-v1':
        num_class = 174
        args.rgb_prefix = ''
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'diving48':
        num_class = 48
        args.rgb_prefix = 'frames'
        rgb_read_format = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.model_arc == 'inception':

        model = InceptionV3GSM.InceptionV3(class_dim=num_class, seg_num=args.seg_num, seglen=args.seg_len)
   
    if args.weights:
        model_dict = paddle.load(args.weights)
        model.set_state_dict(model_dict)
    else:
        print("model path must be specified")
        exit()
        
    args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset)
    val_loader = paddle.io.DataLoader(
    VideoDataset(args.root_path, args.val_list, num_segments=args.seg_num,
                new_length=data_length,
                modality=args.modality,
            #    num_clips=2,
                image_tmpl=args.rgb_prefix+rgb_read_format,
                random_shift=False,
                transform=T.Compose([
                    GroupCenterCrop(229),
                    GroupNdarray()
                ])),
                places=paddle.CUDAPlace(0),
    batch_size=1, shuffle=False)
    
    model.eval()

    val_acc_list = []
    for batch_id, data in enumerate(val_loader()):

        img = data[0]
        label = data[1]
        
        out, acc = model(img, label)
        if out is not None:
            val_acc_list.append(acc.numpy()[0])

    model.train()
    result = np.mean(val_acc_list)

    print("测试集准确率为:{}".format(result))        
            
if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    eval(args)
