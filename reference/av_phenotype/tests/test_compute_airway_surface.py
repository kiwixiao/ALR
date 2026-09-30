"""Tests for compute_airway_surface.mask_to_surface on synthetic masks.

A closed cylinder with a known world origin stands in for an airway mask, so
every expected value (extent, volume, file layout) is known exactly.
"""
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_airway_surface import mask_to_surface  # noqa: E402

SPACING = 0.625
ORIGIN = np.array([-10.0, 20.0, 1300.0])   # far from zero, like a real CT


@pytest.fixture
def cylinder_mask(tmp_path):
    """Binary cylinder along z, radius 6 voxels, fully inside the grid."""
    shape = (40, 40, 60)
    i, j, k = np.indices(shape)
    arr = (((i - 20) ** 2 + (j - 20) ** 2) <= 6 ** 2) & (k >= 10) & (k < 50)
    affine = np.diag([SPACING, SPACING, SPACING, 1.0])
    affine[:3, 3] = ORIGIN
    path = tmp_path / 'mask.nii.gz'
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), affine), path)
    return path, arr


def test_surface_sits_on_the_mask_in_world_mm(cylinder_mask, tmp_path):
    src, arr = cylinder_mask
    dst = tmp_path / 'surface.stl'
    mask_to_surface(src, dst)

    idx = np.argwhere(arr)
    lo = ORIGIN + (idx.min(axis=0) - 0.5) * SPACING
    hi = ORIGIN + (idx.max(axis=0) + 0.5) * SPACING
    b = np.array(pv.read(dst).bounds).reshape(3, 2)
    assert np.all(np.abs(b[:, 0] - lo) <= SPACING)
    assert np.all(np.abs(b[:, 1] - hi) <= SPACING)


def test_closed_surface_volume_matches_voxel_volume(cylinder_mask, tmp_path):
    src, arr = cylinder_mask
    dst = tmp_path / 'surface.stl'
    mask_to_surface(src, dst)

    mesh = pv.read(dst).clean()
    open_edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                            manifold_edges=False, non_manifold_edges=False)
    assert open_edges.n_cells == 0
    voxel_volume = arr.sum() * SPACING ** 3
    assert abs(mesh.volume / voxel_volume - 1.0) < 0.05


def test_writes_binary_stl(cylinder_mask, tmp_path):
    src, _ = cylinder_mask
    dst = tmp_path / 'surface.stl'
    n_tri = mask_to_surface(src, dst)

    assert n_tri > 0
    assert os.path.getsize(dst) == 84 + 50 * n_tri   # binary STL layout


def test_rejects_flipped_affine(tmp_path):
    affine = np.diag([-SPACING, SPACING, SPACING, 1.0])   # x axis flipped
    src = tmp_path / 'flipped.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), affine), src)

    with pytest.raises(ValueError, match='axis aligned'):
        mask_to_surface(src, tmp_path / 'surface.stl')
    assert not (tmp_path / 'surface.stl').exists()


def test_skips_when_surface_is_newer_than_mask(cylinder_mask, tmp_path):
    src, _ = cylinder_mask
    dst = tmp_path / 'surface.stl'
    mask_to_surface(src, dst)
    first = os.path.getmtime(dst)
    time.sleep(0.05)

    assert mask_to_surface(src, dst) is None
    assert os.path.getmtime(dst) == first
