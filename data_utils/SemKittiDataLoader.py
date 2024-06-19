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
    return np.max(
        np.sqrt(np.sum((points - np.expand_dims(points, 1))**2, axis=-1)))


def process_submap(filename: Path, uniform: bool, npoints: int,
                   use_normals: bool) -> None:
    """
    Process a submap
    :param filename: Path, file to process
    :param uniform: bool, whether to use uniform sampling
    :param npoints: int, number of points to sample
    :param use_normals: bool, whether to use normals
    """
    # load data
    data = pickle.load(open(filename, 'rb'))
    for i in range(len(data['points'])):
        # for each cluster
        points = data['points'][i]
        if len(points.shape) < 2:
            continue
        label = data['labels'][i]
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
        # create data
        data = {
            "points": points,
            "labels": label,
            "furthest_distance": furthest_distance_of_points(points)
        }
        # save pickle
        processed_filename = filename.parents[1] / 'processed' / str(
            filename).split('/')[-2] / filename.name.replace(
                '.pkl', f'_cluster{i}_label{label}.pkl')
        pickle.dump(data, open(processed_filename, 'wb'))


def read_furthest_distance(filename: Path) -> float:
    '''
    Read the furthest distance of points from a file
    :param filename: Path, file to read
    :return: float, furthest distance
    '''
    data = pickle.load(open(filename, 'rb'))
    return data['furthest_distance']


def normalize_points_in_pickle(filename: Path, scale: float) -> None:
    '''
    Normalize points in a pickle file
    :param filename: Path, file to process
    :param scale: float, scale to normalize points
    '''
    assert scale > 0
    data = pickle.load(open(filename, 'rb'))
    data['points'] = pc_normalize(data['points'], scale)
    pickle.dump(data, open(filename, 'wb'))


def return_number_of_points_per_label(filename: Path) -> int:
    '''
    Return the number of points per label
    :param filename: Path, file to process
    :return: int, number of points
    '''
    data = pickle.load(open(filename, 'rb'))
    return data['points'].shape[0]


class SemKittiDataloader(Dataset):

    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.path_list = []
        self.num_classes = 20  # number maximum of classes in Cylinder3d

        assert (split == 'train' or split == 'test')
        click.echo(click.style('=' * 50, fg='green', bold=True))
        # check if dataset has been processed
        preprocessed_output_folder = os.path.join(root, f"processed/{split}")
        if not os.path.exists(preprocessed_output_folder) or self.process_data:
            # create output folder
            os.makedirs(preprocessed_output_folder, exist_ok=True)
            # process data
            click.echo(
                click.style('Processing data %s ...' % split,
                            fg='yellow',
                            bold=True))
            joblib.Parallel(n_jobs=12)(
                joblib.delayed(process_submap)(filename, self.uniform,
                                               self.npoints, self.use_normals)
                for filename in tqdm(sorted(
                    Path(f"{root}/{split}").rglob('*.pkl')),
                                     colour='yellow'))
            # get norm constant as the 95% percentile per class
            norm_scales_per_class: dict[int, float] = {}
            click.echo(
                click.style('Computing normalization scales per class ...',
                            fg='yellow',
                            bold=True))
            for i in tqdm(range(self.num_classes), colour='yellow'):
                # read furthest distance per class
                furthest_distances_all_clusters = joblib.Parallel(
                    n_jobs=12, return_as="list")(
                        joblib.delayed(read_furthest_distance)(filename)
                        for filename in sorted(
                            Path(preprocessed_output_folder).rglob(
                                f'*label{i}.pkl')))
                if len(furthest_distances_all_clusters) == 0:  # type: ignore
                    continue
                norm_scales_per_class[i] = float(
                    np.percentile(np.array(furthest_distances_all_clusters),
                                  95))
            click.echo(
                click.style(
                    f" >> Normalization scales per class: {np.array2string(np.array([norm_scales_per_class[i] for i in norm_scales_per_class.keys()]), precision=2, separator=', ')}",
                    fg='yellow'))
            # normalize points
            click.echo(
                click.style('Normalizing points ...', fg='yellow', bold=True))
            joblib.Parallel(n_jobs=12)(
                joblib.delayed(normalize_points_in_pickle)(
                    filename, norm_scales_per_class[int(
                        str(filename.name).split('_')[-1].split('.')[0][5:])])
                for filename in tqdm(sorted(
                    Path(preprocessed_output_folder).rglob('*.pkl')),
                                     colour='yellow'))

        # load paths
        self.path_list = sorted(
            Path(preprocessed_output_folder).rglob('*.pkl'))  # type: ignore

        # get label names
        label_names = [
            int(str(p.name).split('_')[-1].split('.')[0][5:])
            for p in self.path_list
        ]

        # load total number of points per label
        total_points_per_label = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            generator = joblib.Parallel(n_jobs=12, return_as="generator")(
                joblib.delayed(return_number_of_points_per_label)(filename)
                for filename in sorted(
                    Path(preprocessed_output_folder).rglob(f'*label{i}.pkl')))
            total_points_per_label[i] = sum(list(generator))  # type: ignore

        # compute label weights
        unique_labels = np.unique(label_names)
        self.label_weights = np.zeros(self.num_classes)
        for i in range(len(unique_labels)):
            self.label_weights[unique_labels[i]] = sum(
                total_points_per_label) / total_points_per_label[
                    unique_labels[i]]

        # print stats
        click.echo(
            click.style(
                f"Loaded {split} dataset with {len(self.path_list)} clusters",
                fg='green',
                bold=True))
        click.echo(
            click.style(
                f" >> Labels in dataset: {np.array2string(unique_labels, precision=0, separator=', ')}",
                fg='green'))
        click.echo(
            click.style(
                f" >> Label weights: {np.array2string(self.label_weights, precision=3, separator=', ')}",
                fg='green'))
        click.echo(
            click.style(
                f" >> Number of points per class: {np.array2string(total_points_per_label, precision=0, separator=', ')}",
                fg='green'))

    def __len__(self):
        return len(self.path_list)

    def _get_item(self, index):
        # read data
        data = pickle.load(open(self.path_list[index], 'rb'))
        points = data['points']
        label = data['labels']
        return points, label

    def __getitem__(self, index):
        return self._get_item(index)
