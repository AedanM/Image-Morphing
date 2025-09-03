"""Perform affine warp."""

from typing import Any

import cv2
import numpy as np


def AffineWarp(img: Any, fromPts: Any, toPts: Any, tri: Any) -> Any:
    """Perform affine warp."""
    warped = np.zeros_like(img)

    for i in range(tri.shape[0]):
        srcTri = fromPts[tri[i]]
        dstTri = toPts[tri[i]]

        # Get bounding box for each triangle
        r1 = cv2.boundingRect(np.float32([srcTri]))  # pyright: ignore[reportArgumentType]
        r2 = cv2.boundingRect(np.float32([dstTri]))  # pyright: ignore[reportArgumentType]
        x1, x2, y1, y2 = (r2[1] - 1, r2[1] + r2[3], r2[0] - 1, r2[0] + r2[2])
        # Offset points by left top corner of the respective rectangles
        srcTriRect = []
        dstTriRect = []

        for j in range(3):
            srcTriRect.append(((srcTri[j][0] - r1[0]), (srcTri[j][1] - r1[1])))
            dstTriRect.append(((dstTri[j][0] - r2[0]), (dstTri[j][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros_like(warped[x1:x2, y1:y2], dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dstTriRect), (1.0, 1.0, 1.0), 16, 0)  # pyright: ignore[reportArgumentType]

        # Apply warp to the triangle
        imgRect = img[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
        warpMap = cv2.getAffineTransform(
            np.float32(srcTriRect),  # pyright: ignore[reportArgumentType]
            np.float32(dstTriRect),  # pyright: ignore[reportArgumentType]
        )
        warpedImgRect = cv2.warpAffine(
            imgRect,
            warpMap,
            (mask.shape[1], mask.shape[0]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Copy triangular region of the rectangular patch to the output image
        warped[x1:x2, y1:y2] = warped[x1:x2, y1:y2] * (1 - mask) + warpedImgRect * mask

    return warped
