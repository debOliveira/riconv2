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
import pandas as pd
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


def pc_normalize(pc, scale=0.0):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if not scale:
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
    else:
        pc = pc / scale
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


def furthest_distance_of_points(points):
    '''
    Get the furthest distance of points
    :param points: np.array, shape=(N, 3)
    :return: float, furthest distance
    '''
    N = points.shape[0]
    return np.max(
        np.sqrt(np.sum((points - np.expand_dims(points, 1))**2, axis=-1)))


def process_submap(filename, uniform, npoints, use_normals):
    # load data
    data = pickle.load(open(filename, 'rb'))
    processed_points = []
    for i in range(len(data['points'])):
        # for each cluster
        points = data['points'][i]
        # sample points
        if uniform:
            points = farthest_point_sample(points, npoints)
        # normalize points on unit sphere
        # points = pc_normalize(points)
        # compute normals
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
        self.labels: list[int] = []
        self.num_classes = 20  # number maximum of classes in Cylinder3d

        assert (split == 'train' or split == 'test')
        click.echo(click.style('=' * 50, fg='green', bold=True))
        # check if dataset has been processed
        points_filename = os.path.join(root, f"points_{split}.parquet")
        if not os.path.exists(points_filename) or self.process_data:
            # process data
            click.echo(
                click.style('Processing data %s ...' % split,
                            fg='yellow',
                            bold=True))
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
            # get norm constant as the 95% percentile per class
            norm_scales_per_class: dict[int, float] = {}
            click.echo(
                click.style('Computing normalization scales per class ...',
                            fg='yellow',
                            bold=True))
            for i in tqdm(
                    range(self.num_classes),
                    colour='yellow',
            ):
                # get index of points in class
                index_of_points_in_class = np.array([
                    j for j in range(len(self.points)) if self.labels[j] == i
                ])
                if len(index_of_points_in_class) == 0:
                    continue
                # get furthest distance of points in class TODO: check this function
                furthest_distance = [
                    furthest_distance_of_points(self.points[j])
                    for j in index_of_points_in_class
                ]
                # get 95% percentile
                norm_scales_per_class[i] = np.quantile(furthest_distance,
                                                       0.95).astype(float)
            click.echo(
                click.style(
                    f" >> Normalization scales per class: {np.array2string(np.array([norm_scales_per_class[i] for i in norm_scales_per_class.keys()]), precision=2, separator=', ')}",
                    fg='yellow'))
            # normalize points
            click.echo(
                click.style('Normalizing points ...', fg='yellow', bold=True))
            for i in tqdm(range(len(self.points)), colour='yellow'):
                self.points[i] = pc_normalize(self.points[i],
                                              scale=norm_scales_per_class[int(
                                                  self.labels[i])])
        else:
            # load processed data
            click.echo(
                click.style('Loading processed data %s ...' % split,
                            fg='green',
                            bold=True))
            df_points = pd.read_parquet(points_filename)
            df_labels = pd.read_parquet(
                points_filename.replace('points', 'labels'))
            self.labels = df_labels['labels'].tolist()
            self.points = np.split(
                df_points[["x", "y", "z"]].to_numpy(),
                np.cumsum(df_labels['n_points'].values.tolist())[:-1])

        # sanity check
        assert len(self.points) == len(self.labels)

        # compute label weights
        unique_labels, n_unique_labels = np.unique(self.labels,
                                                   return_counts=True)
        self.label_weights = np.zeros(self.num_classes)
        for i in range(len(unique_labels)):
            self.label_weights[unique_labels[i]] = 1 / n_unique_labels[i]

        # save processed data
        click.echo(
            click.style('Saving processed %s data...' % split,
                        fg='green',
                        bold=True))
        if not os.path.exists(points_filename) or self.process_data:
            flattened_points = np.concatenate(self.points, axis=0)
            df_points = pd.DataFrame({
                'x': flattened_points[:, 0],
                'y': flattened_points[:, 1],
                'z': flattened_points[:, 2],
            })
            df_labels = pd.DataFrame({
                'labels': self.labels,
                "n_points": [len(p) for p in self.points]
            })
            # save data
            df_points.to_parquet(points_filename)
            df_labels.to_parquet(points_filename.replace('points', 'labels'))
        # print stats
        click.echo(
            click.style(
                f"Loaded {split} dataset with {len(self.points)} samples",
                fg='green',
                bold=True))
        click.echo(
            click.style(
                f" >> Labels in dataset: {np.array2string(unique_labels, precision=0, separator=', ')}",
                fg='green'))
        click.echo(
            click.style(
                f" >> Number of points per class: {np.array2string(n_unique_labels, precision=0, separator=', ')}",
                fg='green'))

    def __len__(self):
        return len(self.labels)

    def _get_item(self, index):
        # TODO: add normalization
        return self.points[index], self.labels[index]

    def __getitem__(self, index):
        return self._get_item(index)
