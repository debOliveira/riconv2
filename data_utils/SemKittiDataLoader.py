'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import pickle
import sys
import warnings
from pathlib import Path

import click
import joblib
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models.pointnet2 import pointnet2_utils
from models.riconv2_utils import compute_LRA, index_points

warnings.filterwarnings('ignore')


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


def process_submap(filename, uniform, npoints, use_normals):
    data = pickle.load(open(filename, 'rb'))
    processed_points = []
    for i in range(len(data['points'])):
        points = data['points'][i]
        if uniform:
            points = farthest_point_sample(points, npoints)
        points = pc_normalize(points)
        if use_normals:
            normals = compute_LRA(Tensor(np.expand_dims(points, 0)), True,
                                  32).squeeze().numpy()
            points = np.concatenate((points, normals), axis=1)
        processed_points.append(points)
    return {"points": processed_points, "labels": data['labels']}


class SemKittiDataloader(Dataset):

    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.points = []
        self.labels = []

        assert (split == 'train' or split == 'test')
        # check if dataset has been processed
        all_clusters_filename = os.path.join(root, f"all_clusters_{split}.pkl")
        if not os.path.exists(all_clusters_filename) or self.process_data:
            click.echo(
                click.style('Processing data %s ...' % split, fg='yellow'))
            data_list = joblib.Parallel(n_jobs=12, return_as="generator")(
                joblib.delayed(process_submap)(filename, self.uniform,
                                               self.npoints, self.use_normals)
                for filename in tqdm(sorted(
                    Path(f"{root}/{split}").rglob('*.pkl')),
                                     colour='yellow'))
            for data in data_list:
                assert data is not None
                self.points.extend(data['points'])
                self.labels.extend(data['labels'])
            # save processed data
            all_clusters = {'points': self.points, 'labels': self.labels}
            click.echo(
                click.style('Saving processed data %s ...' % split,
                            fg='green'))
            pickle.dump(all_clusters, open(all_clusters_filename, 'wb'))
        else:
            click.echo(
                click.style('Loading processed data %s ...' % split,
                            fg='green'))
            all_clusters = pickle.load(open(all_clusters_filename, 'rb'))
            self.points = all_clusters['points']
            self.labels = all_clusters['labels']

        assert len(self.points) == len(self.labels)

        # compute label weights
        unique_labels, n_unique_labels = np.unique(self.labels,
                                                   return_counts=True)
        self.num_classes = 20  # number maximum of classes in Cylinder3d
        self.label_weights = np.zeros(self.num_classes)
        for i in range(len(unique_labels)):
            self.label_weights[unique_labels[i]] = 1 / n_unique_labels[i]

        # print stats
        click.echo(
            click.style(
                f"Total number of {split} data: {len(self.points)}, Total number of classes: {self.num_classes}",
                fg='green'))
        click.echo(
            click.style("Labels in dataset: %s" %
                        str(unique_labels).replace('\n', ''),
                        fg='green'))
        click.echo(
            click.style("Number of points per class: %s" %
                        str(n_unique_labels).replace('\n', ''),
                        fg='green'))
        click.echo(
            click.style("Label weights: %s" %
                        str(self.label_weights).replace('\n', ''),
                        fg='green'))

    def __len__(self):
        return len(self.labels)

    def _get_item(self, index):
        return self.points[index], self.labels[index]

    def __getitem__(self, index):
        return self._get_item(index)
