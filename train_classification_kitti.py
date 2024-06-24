"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import argparse
import datetime
import importlib
import os
import shutil
import sys
from pathlib import Path

import click
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import provider
from data_utils.SemKittiDataLoader import SemKittiDataloader

import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu',
                        type=bool,
                        default=False,
                        help='use cpu mode')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='specify gpu device')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='batch size in training')
    parser.add_argument('--model',
                        default='riconv2_cls',
                        help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category',
                        default=20,
                        type=int,
                        choices=[20, 28],
                        help='training on ModelNet10/40')
    parser.add_argument('--epoch',
                        default=50,
                        type=int,
                        help='number of epoch in training')
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float,
                        help='learning rate in training')
    parser.add_argument('--num_point',
                        type=int,
                        default=1024,
                        help='Point Number')
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='optimizer for training')
    parser.add_argument('--log_dir',
                        type=str,
                        default=None,
                        help='experiment root')
    parser.add_argument('--decay_rate',
                        type=float,
                        default=1e-4,
                        help='decay rate')
    parser.add_argument('--use_normals',
                        type=bool,
                        default=True,
                        help='use normals')
    parser.add_argument('--process_data',
                        type=bool,
                        default=False,
                        help='save data offline')
    parser.add_argument('--use_uniform_sample',
                        type=bool,
                        default=True,
                        help='use uniform sampiling')
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points = torch.tensor(
            provider.rotate_point_cloud_with_normal_so3(points))
        if not args.use_cpu:
            points, target = points.cuda().float(), target.cuda()
        pred, _ = classifier(points)
        if len(pred.shape) == 3:
            pred = pred.mean(dim=1)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(
                target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    click.echo(
        click.style(
            f">> Accuracy per class: {np.array2string(class_acc[:, 2], precision=3, separator=', ')}",
            fg='cyan'))
    class_acc_per_class = class_acc[:, 2]
    class_acc = np.mean(np.nan_to_num(class_acc[:, 2]))
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc, class_acc_per_class


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''CREATE DIR'''
    timestr = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))

    # create wandb instance
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(project='reloc-gnn-riconv-kitti', config=args, name=timestr)
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification_modelnet40')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    '''LOG'''
    args = parse_args()
    click.echo(click.style(f">> Options: {args}", fg='blue'))
    '''DATA LOADING'''
    data_path = '../data/kitti_clusters'

    train_dataset = SemKittiDataloader(root=data_path,
                                       args=args,
                                       split='train',
                                       process_data=args.process_data)
    label_weights = torch.tensor(train_dataset.label_weights).float()
    test_dataset = SemKittiDataloader(root=data_path,
                                      args=args,
                                      split='test',
                                      process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=10,
                                                  drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=10)
    '''MODEL LOADING'''
    num_class = train_dataset.num_classes
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils.py' % args.model.split('_')[0], str(exp_dir))
    shutil.copy('./train_classification_kitti.py', str(exp_dir))

    classifier = model.get_model(num_class, 2, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    click.echo(click.style("=" * 50, fg='green', bold=True))
    checkpoint = torch.load(
        './log/classification_modelnet40/pretrained/checkpoints/best_model.pth'
    )
    start_epoch = 0
    classifier_dict = classifier.state_dict()
    checkpoint_dict = checkpoint['model_state_dict']
    checkpoint_dict["fc3.weight"] = classifier_dict["fc3.weight"]
    checkpoint_dict["fc3.bias"] = classifier_dict["fc3.bias"]
    classifier.load_state_dict(checkpoint_dict)
    click.echo(
        click.style('Using pretrain model %s ...' % str(exp_dir),
                    fg='green',
                    bold=True))

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=0.01,
                                    momentum=0.9)
    """scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.7)"""
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    step = 0
    '''TRANING'''
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        classifier = classifier.train()

        #scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0),
                                               total=len(trainDataLoader),
                                               smoothing=0.9,
                                               colour='magenta',
                                               desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            points = points.data.numpy()
            #points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :,
                                                                         0:3])
            #points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # rotate pointcloud along z direction
            points = provider.rotate_point_cloud_by_angle_with_normal(points)
            points = torch.Tensor(points)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
                label_weights = label_weights.cuda()

            pred, _ = classifier(points)

            loss = criterion(pred, target.long(), label_weights)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            wandb.log(
                {
                    "Loss": loss.item(),
                    "Training accuracy": mean_correct[-1],
                },
                step=step)
            loss.backward()
            optimizer.step()
            global_step += 1
            step += 1

        train_instance_acc = np.mean(mean_correct)
        click.echo(
            click.style(f">> Train Instance Accuracy: {train_instance_acc}",
                        fg='magenta'))

        with torch.no_grad():
            instance_acc, class_acc, _ = test(classifier.eval(),
                                              testDataLoader,
                                              num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            click.echo(
                click.style(
                    f">> Test Instance Accuracy: {instance_acc}, Class Accuracy: {class_acc}",
                    fg='cyan'))
            click.echo(
                click.style(
                    f">> Best Instance Accuracy: {best_instance_acc}, Class Accuracy: {best_class_acc}",
                    fg='green'))
            wandb.log(
                {
                    "Test Instance Accuracy": instance_acc,
                    "Test Class Accuracy": class_acc,
                    "Best Instance Accuracy": best_instance_acc,
                    "Best Class Accuracy": best_class_acc,
                },
                step=step)

            if (instance_acc >= best_instance_acc):
                savepath = str(checkpoints_dir) + '/best_model.pth'
                click.echo(
                    click.style(f">> Saving best model at {savepath}...",
                                fg='green'))
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
