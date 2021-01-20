import os
import argparse
import ast
import logging
import numpy as np
import paddle

from model import InceptionV3GSM
import datasets_video
from dataset import VideoDataset

from paddle.vision import transforms as T
from transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupNdarray, GroupCenterCrop

import time


logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
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
        '--batch_size',
        type=int,
        default=16,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--modality',
        type=str,
        default='RGB',
        help='Video frame mode.')
    parser.add_argument(
        '--pretrain',
        type=ast.literal_eval,
        default=False,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='best_model/best_diving_model',
        help='pretrained model weights')
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--model_arc',
        type=str,
        default='inceptionv3',
        help='model architecture used for training')
    parser.add_argument(
        '--best_model_save',
        type=str,
        default='best_model/best_model',
        help='dir for best model saving')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='result/data.npz',
        help='dir for mode training and evaluating data')
    args = parser.parse_args()
    return args

def get_augmentation(modality, target_transform, input_size):
    if modality == 'RGB':
        # return T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
        #                                         GroupRandomHorizontalFlip(is_flow=False,
        #                                                                     target_transform=target_transform)],
        #                                                                     )
        return T.Compose([GroupRandomHorizontalFlip(is_flow=False, target_transform=target_transform)],
                                                                            )
    elif modality == 'Flow':
        return T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                                GroupRandomHorizontalFlip(is_flow=True)])
    elif modality == 'RGBDiff':
        return T.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                                GroupRandomHorizontalFlip(is_flow=False)])

base_lr = 1e-3
wamup_steps = 2
def make_optimizer(epochs, parameters=None):
    momentum = 0.9
    weight_decay = 4e-5

    learning_rate= paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=epochs, verbose=False)

    learning_rate = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=wamup_steps,
        start_lr=base_lr / 10.,
        end_lr=base_lr,
        verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        parameters=parameters)
    return optimizer

def train(args):
    all_train_rewards=[]
    all_test_rewards=[]
    prev_result=0

    # config = parse_config(args.config)
    args = parse_args()
    # args = parser.parse_args()
    # check_rootfolders()

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

    if 'something' in args.dataset:
        # label transformation for left/right categories
        target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
        print('Target transformation is enabled....')
    else:
        target_transforms = None

    args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset)

    if args.model_arc == 'inception':

        model = InceptionV3GSM.InceptionV3(class_dim=num_class, seg_num=args.seg_num, seglen=args.seg_len)

    
    if args.pretrain:
        # 加载上一次训练的模型，继续训练
        model_dict = paddle.load(args.weights)

        model.set_state_dict(model_dict)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_augmentation = get_augmentation(args.modality, target_transform = target_transforms, input_size=229)

    train_dataset = VideoDataset(args.root_path, args.train_list, num_segments=args.seg_num,
                   new_length=data_length,
                   modality=args.modality,
                #    num_clips=2,
                   image_tmpl=args.rgb_prefix+rgb_read_format,
                   transform=paddle.vision.transforms.Compose([
                       train_augmentation,
                       GroupCenterCrop(229),
                       GroupNdarray()
                   ]))


    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=paddle.CUDAPlace(0),
        batch_size=args.batch_size, shuffle=True)

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

    epochs = args.epoch or model.epoch_num()

    opt = make_optimizer(epochs, model.parameters())

    # opt = paddle.optimizer.Momentum(0.001, 0.9, parameters=model.parameters())

    model.train()

    for i in range(epochs):

        for batch_id, data in enumerate(train_loader()):

            img = data[0]
            label = data[1]

            out, acc = model(img, label)

            if out is not None:
            
                loss = paddle.nn.functional.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)

                avg_loss.backward()

                opt.minimize(avg_loss)
                model.clear_gradients()
          
                if batch_id % 200 == 0:
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    paddle.save(model.state_dict(), args.save_dir + '/diving_model')
        all_train_rewards.append(acc.numpy())

        #### validate during training

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


        ### end of validating

        all_test_rewards.append(result)
        if result > prev_result:
            prev_result = result
            print('The best result is ' + str(result))
            paddle.save(model.state_dict(), args.best_model_save)#保存模型'best_model/best_diving_model_229_32'
    logger.info("Final loss: {}".format(avg_loss.numpy()))
    print("Final loss: {}".format(avg_loss.numpy()))
    np.savez(args.data_dir, all_train_rewards=all_train_rewards, all_test_rewards=all_test_rewards)
    # np.savez('result/final_diving_data__229_32.npz', all_train_rewards=all_train_rewards, all_test_rewards=all_test_rewards)

if __name__ == "__main__":
    args = parse_args()
    # logger.info(args)
    start_time = time.time()
    train(args)
    end_time = time.time()
    print('time duration is ', end_time - start_time)
