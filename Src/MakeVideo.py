from pathlib import Path

import cv2


def CreateVideoFromFrames(folder: Path, fps: float = 30.0):
    # pylint: disable=E1101
    # Get list of image files in the input folder
    frameFolder = folder / "Frames"
    videoDst = folder / f"{folder.stem}.mp4"
    images = sorted(
        [
            f
            for f in Path(frameFolder).iterdir()
            if f.is_file() and f.suffix in [".png", ".jpg", ".jpeg"]
        ]
    )

    # Read the first image to get the dimensions
    frame = cv2.imread(str(images[0]))
    height, width, _layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type:ignore
    video = cv2.VideoWriter(str(videoDst), fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(str(image))
        video.write(frame)

    video.release()
