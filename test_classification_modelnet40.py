"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""
import torch
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint, ))
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='specify gpu device')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='batch size in training')
    parser.add_argument('--num_category',
                        default=40,
                        type=int,
                        help='training on ModelNet40')
    parser.add_argument('--num_point',
                        type=int,
                        default=1024,
                        help='Point Number')
    parser.add_argument('--log_dir',
                        type=str,
                        default='pretrained',
                        help='log root')
    parser.add_argument('--process_data',
                        type=bool,
                        default=True,
                        help='save data offline')
    parser.add_argument('--use_normals',
                        type=bool,
                        default=True,
                        help='use normals')
    parser.add_argument('--use_uniform_sample',
                        type=bool,
                        default=True,
                        help='use uniform sampiling')
    return parser.parse_args()


def test(model, loader, args, num_class=40):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # if not car in target, skip
        if 7 not in target:
            continue
        # remove all points which x is less than 0
        for i, pc in enumerate(points):
            if target[i] != 7:
                continue
            # get points
            new_pc = pc[pc[:, 0] > 0]
            # resample
            new_pc = farthest_point_sample(new_pc.numpy(), 1024)
            # renormalize
            #new_pc[:, 0:3] = pc_normalize(new_pc[:, 0:3])
            # add 1 dimension for batch
            new_pc = np.expand_dims(new_pc, axis=0)
            # predict
            pred, _ = classifier(torch.from_numpy(new_pc).cuda())
            # get the point cloud and normals
            points_3d = new_pc[:, :, :3][0]
            normals = new_pc[:, :, 3:][0]
            features = pred[0].cpu().detach().numpy().reshape(1, -1)
            # initialize the point cloud
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            fig = plt.figure(figsize=(12, 2.5))
            gs = gridspec.GridSpec(1,
                                   4,
                                   figure=fig,
                                   width_ratios=[1, 1, .1, 1.4],
                                   wspace=0.1)
            # subplot the point cloud on the left
            ax = fig.add_subplot(gs[0], projection='3d')
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
            plt.axis('equal')
            ax.set_xlim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_xlim(-1, 1)  # type: ignore
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.xaxis.set_tick_params(pad=0)
            ax.yaxis.set_tick_params(pad=0)
            ax.zaxis.set_tick_params(pad=0)  # type: ignore
            # subplot the point cloud on the left
            ax = fig.add_subplot(gs[1], projection='3d')
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
            ax.view_init(30, -30)  # type: ignore
            plt.axis('equal')
            ax.set_xlim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_xlim(-1, 1)  # type: ignore
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.xaxis.set_tick_params(pad=0)
            ax.yaxis.set_tick_params(pad=0)
            ax.zaxis.set_tick_params(pad=0)  # type: ignore
            # subplot the feature map on the right
            gs_subplot = gridspec.GridSpecFromSubplotSpec(2,
                                                          1,
                                                          subplot_spec=gs[3],
                                                          height_ratios=[5, 1],
                                                          hspace=0)
            ax = fig.add_subplot(gs_subplot[0])
            ax.set_ylabel('Feature Value')
            ax.set_xlim([0, features.shape[1]])  # type: ignore
            ax.plot(np.array(range(features.shape[1])) + 0.5,
                    features.T,
                    color='black',
                    linewidth=2)
            xticks = ax.get_xticks()
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            ax.grid()
            # subplot the feature heatmap on the right
            ax = fig.add_subplot(gs_subplot[1])
            from matplotlib.colors import LinearSegmentedColormap
            c = [
                "darkred", "red", "lightcoral", "white", "palegreen", "green",
                "darkgreen"
            ]
            v = [0, .15, .4, .5, 0.6, .9, 1.]
            l = list(zip(v, c))  # type: ignore
            cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            ax.imshow(features, cmap=cmap, aspect='auto')
            xlabels = [str(int(i)) for i in xticks]
            ax.set_xlim([0 - 0.5, features.shape[1] - 0.5])  # type: ignore
            ax.set_xticks(ticks=xticks - 0.5, labels=xlabels)
            ax.set_xlabel('Feature Index')
            # final adjustments
            plt.tight_layout()
            fig.suptitle(f"Batch {j}, Item {i}, Class {target[i]}")
            plt.savefig(f"tmp/batch_{j}_item_{i}_halfx_unnormalized.png",
                        bbox_inches='tight')
            plt.close()

        continue
        pred, feat = classifier(points.cuda())
        if len(pred.shape) == 3:
            print("Using max pooling")
            pred = pred.mean(dim=1)

        pred_choice = pred.data.max(1)[1].cpu()
        for cat in np.unique(target):
            classacc = pred_choice[target == cat].eq(
                target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = 'log/classification_modelnet40/' + args.log_dir
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../data/modelnet40_normal_resampled/'  # original data
    if not args.process_data:
        data_path = '../data/modelnet40_preprocessed/'  # preprocessed data

    test_dataset = ModelNetDataLoader(root=data_path,
                                      args=args,
                                      split='test',
                                      process_data=args.process_data)
    testDataLoader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=10)
    """counter = 0
    for batch_id, data in enumerate(testDataLoader):
        points, target = data
        points_3d = points.numpy()[:, :, :3][0]
        normals = points.numpy()[:, :, 3:][0]

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        ax.quiver(points_3d[:, 0],
                  points_3d[:, 1],
                  points_3d[:, 2],
                  normals[:, 0],
                  normals[:, 1],
                  normals[:, 2],
                  length=0.05)
        plt.savefig(f"point_clouds/{counter}.png")
        plt.axis('equal')
        plt.show()
        break
        if counter == 15:
            break
    exit()"""
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module('riconv2_cls')

    classifier = model.get_model(num_class, 2)
    classifier = classifier.cuda()

    checkpoint = torch.load(
        str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(),
                                       testDataLoader,
                                       args,
                                       num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' %
                   (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
