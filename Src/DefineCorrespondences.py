"""Match points together."""

from pathlib import Path
from typing import Any

import numpy as np
from PtsCapture.cpselect import cpselect
from scipy.spatial import Delaunay


def CSVtoSplitLines(path: Path) -> Any:
    """Format the csv data."""
    return np.array([[float(y) for y in x.split(",")] for x in path.read_text().split("\n")])


def DefineCorrespondences(im1: Any, im2: Any, im1PtsPath: Path, im2PtsPath: Path) -> tuple:
    """
    Define correspondences between two images.

    Parameters
    ----------
    im1, im2: Input images.
    im1_pts_path, im2_pts_path: Paths to save/load the points.

    Returns
    -------
    im1_pts, im2_pts: Corresponding points in the images.
    tri: Triangulation structure.
    """
    if not im1PtsPath.exists() or not im2PtsPath.exists():
        results = cpselect(im1, im2)
    else:
        im1Pts = CSVtoSplitLines(im1PtsPath)
        im2Pts = CSVtoSplitLines(im2PtsPath)
        results = cpselect(im1, im2, im1Pts, im2Pts)  # pyright: ignore[reportArgumentType]

    im1Pts = np.array([[x["img1_x"], x["img1_y"]] for x in results])
    im2Pts = np.array([[x["img2_x"], x["img2_y"]] for x in results])
    # Append four corners to cover the entire image with triangles
    im1Pts = np.vstack(
        [im1Pts, [1, 1], [im1.shape[1], 1], [im1.shape[1], im1.shape[0]], [1, im1.shape[0]]],
    )
    im2Pts = np.vstack(
        [im2Pts, [1, 1], [im2.shape[1], 1], [im2.shape[1], im2.shape[0]], [1, im2.shape[0]]],
    )
    im1PtsPath.write_text("\n".join([f"{x[0]},{x[1]}" for x in im1Pts]))
    im2PtsPath.write_text("\n".join([f"{x[0]},{x[1]}" for x in im2Pts]))

    pts1Out = []
    pts2Out = []
    # Mean of the two point sets
    for pt1, pt2 in zip(im1Pts, im2Pts, strict=True):
        xDis = pow(pt2[1] - pt1[1], 2)
        yDis = pow(pt2[0] - pt1[0], 2)
        pts1Out.append(pt1)
        if pow(xDis + yDis, 0.5) < (pt1[0] * 0.01):
            pts2Out.append(pt1)
        else:
            pts2Out.append(pt2)
    pts1OutNP = np.array(pts1Out)
    pts2OutNP = np.array(pts2Out)
    ptsMean = (pts1OutNP + pts2OutNP) / 2
    tri = Delaunay(ptsMean)

    return pts1OutNP, pts2OutNP, tri
