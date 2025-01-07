import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from progress.bar import Bar
from Src.DefineCorrespondences import DefineCorrespondences
from Src.MakeVideo import CreateVideoFromFrames
from Src.Morph import morph


def CheckInputs(startIMG: Path, endIMG: Path, outputDir: Path):

    assert startIMG.is_file() and startIMG.exists(), f"Path {startIMG} of image 1 not found."
    assert endIMG.is_file() and endIMG.exists(), f"Path {endIMG} of image 2 not found."
    if not Path(outputDir).exists():
        Path(outputDir).mkdir(parents=True)
    # pylint: disable=E1101
    startIM = cv2.imread(str(startIMG))
    endIm = cv2.imread(str(endIMG))

    assert startIM.shape == endIm.shape, "Image 1 and image 2 should be of the same size!"
    return (startIM, endIm)


def Main(
    image1: Path = Path(sys.argv[1]),
    image2: Path = Path(sys.argv[2]),
    outDir: Path = Path(sys.argv[3]),
    frameCount: int = int(sys.argv[4]) if len(sys.argv) > 4 else 30,
    vidLength: float = int(sys.argv[5]) if len(sys.argv) > 5 else 1,
    saveFrames: bool = bool(sys.argv[6]) if len(sys.argv) > 6 else True,
):
    startImage, endImage = CheckInputs(image1, image2, outDir)
    dstFolder = outDir / image1.stem
    dstFolder.mkdir(exist_ok=True)
    (dstFolder / "Points").mkdir(exist_ok=True)
    (dstFolder / "Frames").mkdir(exist_ok=True)
    # Defining Correspondences

    im1PtsPath = dstFolder / "Points" / f"{image1.stem}Pts.csv"
    im2PtsPath = dstFolder / "Points" / f"{image2.stem}Pts.csv"
    im1Pts, im2Pts, tri = DefineCorrespondences(startImage, endImage, im1PtsPath, im2PtsPath)

    # The Morph Sequence
    # pylint: disable=E1101
    frame = 0
    with Bar(
        "Generating Frames...", max=frameCount * 2, suffix=r"%(index)d/%(max)d - %(eta)ds"
    ) as progBar:
        for i in range(frameCount // 2):
            cv2.imwrite(str(dstFolder / "Frames" / f"{i:03d}{image1.suffix}"), startImage)
            frame += 1
            progBar.next()
        for frac in np.linspace(0, 1, frameCount):
            morphResult = morph(startImage, endImage, im1Pts, im2Pts, tri, frac, frac)
            frameDst = dstFolder / "Frames" / f"{frame:03d}{image1.suffix}"
            cv2.imwrite(str(frameDst), morphResult)
            frame += 1
            progBar.next()
        for i in range(frameCount // 2):
            cv2.imwrite(str(dstFolder / "Frames" / f"{frame:03d}{image1.suffix}"), endImage)
            frame += 1
            progBar.next()
    CreateVideoFromFrames(dstFolder, frameCount / vidLength)
    if not saveFrames:
        shutil.rmtree(dstFolder / "Frames")


if __name__ == "__main__":
    Main()
