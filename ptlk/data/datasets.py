""" datasets """

import numpy
import torch
import torch.utils.data
import pykitti.utils as pyutils
import numpy as np

from . import globset
from . import mesh
from .. import so3
from .. import se3
from .utils import sample_IR2, R_to_angle


class ModelNet(globset.Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)

class ShapeNet2(globset.Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt

class KITTIVelo(torch.utils.data.Dataset):
    def __init__(self, velo_files, poses, num_pt, partition):
        self.velo_files = velo_files
        self.poses = poses
        self.sample_np = num_pt
        self.partition = partition
        assert len(self.velo_files) == len(self.poses), "Lidar scan and pose does NOT match!"

    def __len__(self):
        return len(self.velo_files) - 1 # last pose change not counted

    def __getitem__(self, index):
        velo_p = pyutils.load_velo_scan(self.velo_files[index])
        velo_n = pyutils.load_velo_scan(self.velo_files[index + 1])
        pose_p = self.poses[index]
        pose_n = self.poses[index + 1]
        rel_pose_T = se3.rel_pose(pose_p, pose_n)

        # downsample velo_pts
        p_si = sample_IR2(velo_p, self.sample_np)
        n_si = sample_IR2(velo_n, self.sample_np)

        rel_pose = R_to_angle(rel_pose_T[:3,:], to_6=True)

        return velo_p[p_si], velo_n[n_si], rel_pose

    def split(self):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = partition * len(dataset),
            len(dataset2) = (1-partition) * len(dataset)
        """
        orig_size = len(self)
        select_size = int(orig_size * self.partition)
        unselect_size = orig_size - select_size

        d1_ids = np.random.choice(orig_size, select_size, replace=False)
        d2_ids = [i for i in range(orig_size) if i not in d1_ids]
        dataset1 = KITTIVelo(self.velo_files[d1_ids], self.poses[d1_ids], self.sample_np, self.partition)
        dataset2 = KITTIVelo(self.velo_files[d2_ids], self.poses[d2_ids], self.sample_np, self.partition)

        return dataset1, dataset2



#EOF
