from Src.OpenCVWarp import AffineWarp


def morph(im1, im2, im1Pts, im2Pts, tri, warpFrac, dissolveFrac):
    """
    Morph image.

    Parameters:
    im1, im2: Input images.
    im1_pts, im2_pts: Corresponding points in the images (n-by-2 matrices of (x,y) locations).
    tri: Triangulation structure.
    warp_frac: Fraction for shape warping (0 to 1).
    dissolve_frac: Fraction for cross-dissolve (0 to 1).

    Returns:
    morphed_im: The morphed image.
    """
    # Compute intermediate shape points
    warp1 = im1Pts * (1 - warpFrac)
    warp2 = im2Pts * warpFrac
    interShapePts = warp1 + warp2
    im1Warp = AffineWarp(im1, im1Pts, interShapePts, tri.simplices)
    im2Warp = AffineWarp(im2, im2Pts, interShapePts, tri.simplices)
    # Cross-dissolve the warped images
    warpedIm = dissolveFrac * im2Warp + (1 - dissolveFrac) * im1Warp

    return warpedIm
