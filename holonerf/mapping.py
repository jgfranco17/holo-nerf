import logging
import os
from time import perf_counter
from typing import List, Tuple

import cv2
import numpy as np
from imutils import resize
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DepthMap:
    def __init__(
        self, directory: str, color: str = "hot", export: bool = False
    ) -> None:
        # Set map style
        self.map_style = {
            "autumn": cv2.COLORMAP_AUTUMN,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "bone": cv2.COLORMAP_BONE,
            "hsv": cv2.COLORMAP_HSV,
            "ocean": cv2.COLORMAP_OCEAN,
            "hot": cv2.COLORMAP_HOT,
        }
        self.map_color = self.map_style.get(color, cv2.COLORMAP_HOT)

        # Compile, preprocess, and set the map image
        self.directory = os.path.join(os.getcwd(), directory)
        self.raw_array_data, self.array_compiled = self.__get_np_data(self.directory)
        self.image = self.preprocess(self.array_compiled)

        # Prepare video writer
        self.writer = None
        self.video_file_path = None
        self.export = export

    @property
    def color(self):
        return self.map_color

    @color.setter
    def set_map_color(self, color: str):
        if color not in self.map_style.keys():
            raise ValueError(f'Invalid color key "{color}" selected.')
        self.map_color = self.map_style[color]
        logger.debug(f'Color map set to "{color}"')

    @staticmethod
    def __get_np_data(source: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load the raw Numpy datae

        Args:
            source (str): Directory filepath to load the data from

        Returns:
            tuple: list of NumPy arrays, compiled array data
        """
        # Initialize blank array
        np_array_list = []
        np_array_data = np.zeros((1024, 1024), dtype=float)

        # Go through all files in target directory
        try:
            logger.info(f"Reading from: {source}")
            storage_dir = os.listdir(source)
            count = 0
            for filename in storage_dir:
                # Store data if file is an NPY file
                if filename.endswith(".npy"):
                    file = str(os.path.join(source, filename))
                    array = np.load(file)
                    np_array_data += array
                    np_array_list.append(array)
                    count += 1

            logger.info(f"Loaded in {count} array files.")
            return (
                np_array_list,
                np.around((np_array_data / count), decimals=0).astype(int),
            )

        except Exception as e:
            logger.info(f"Failed to load NPY data: {e}")

    @staticmethod
    def __normalize(image: np.ndarray, bits: int) -> np.ndarray:
        """Normalize the given map for OpenCV.

        Args:
            image (np.ndarray): Frame image
            bits (int): image bits

        Returns:
            np.ndarray: Normalized depth map
        """
        depth_min = image.min()
        depth_max = image.max()
        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (image - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(image.shape, dtype=image.type)

        if bits == 1:
            return out.astype("uint8")
        return out.astype("uint16")

    @staticmethod
    def __resize(image, factor: float = 1.0) -> np.ndarray:
        """Scale an image evenly by a given factor.

        Args:
            image (np.ndarray): Image to scale
            factor (float, optional): Scaling factor, defaults to 1

        Returns:
            np.ndarray: Resized image
        """
        height, width, _ = image.shape
        new_dimensions = (round(width * factor), round(height * factor))
        logger.info(f'Scaled image {"up" if factor > 1 else "down"} to {factor*100}%')
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    @staticmethod
    def __invert(array: np.ndarray) -> np.ndarray:
        """Invert the given array.

        Args:
            array (np.ndarray): Original array

        Returns:
            np.ndarray: Inverted array
        """
        inverted_array = np.zeros(array.shape)
        rows, cols = array.shape
        local_max = array.max()
        for row in range(rows):
            for col in range(cols):
                inverted_array[row, col] = local_max - array[row, col] + 1

        return inverted_array

    def colormap(self, image: np.ndarray) -> np.ndarray:
        """Recolor the depth map from grayscale to colored.

        Args:
            image (np.ndarray): Grayscale image

        Returns:
            np.ndarray: Colored map image
        """
        depth_map = self.__normalize(image, bits=2)
        depth_map = (depth_map / 256).astype(np.uint8)
        return cv2.applyColorMap(depth_map, self.map_color)

    def preprocess(self, image) -> np.ndarray:
        """Preprocess the depth map from raw array to image layout.

        Args:
            image (np.ndarray): Raw image array

        Returns:
            np.ndarray: Colored depth map
        """
        base_array = self.__invert(image)
        return self.colormap(base_array)

    def display(self, scale: float = 1.0):
        """Display the array data as an image.

        Args:
            scale (float, optional): Sizing scale factor, defaults to 1.0
        """
        try:
            image = self.__resize(self.image, factor=scale)
            x_dim, y_dim, _ = self.image.shape
            logger.info("Displaying compiled depth map")
            cv2.imshow(f"NeRF Depth Map, {x_dim}x{y_dim}", image)
            cv2.waitKey(0)

        except Exception as e:
            logger.error(f"Failed to display depth map: {e}")

        finally:
            cv2.destroyAllWindows()
            logger.info("All windows closed")

    def export_views(self, fps: int = 15):
        """Export the compiled depth maps into video format.

        Args:
            fps (int, optional): Exported video frame rate, defaults to 15
        """
        # Set export video attributes
        video_file_name = f"depthmap_nerf_{len(self.raw_array_data)}imgs_{fps}fps.avi"
        self.video_file_path = os.path.join(os.getcwd(), "videos", video_file_name)
        if os.path.exists(self.video_file_path):
            raise FileExistsError(f"Video file already exists: {self.video_file_path}")

        try:
            # Set video writer
            os.makedirs(os.path.join(os.getcwd(), "videos"), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            height, width, _ = self.image.shape
            self.writer = cv2.VideoWriter(
                self.video_file_path, fourcc, fps, (width, height)
            )

            # Compile all arrays into image format and export
            image_count = len(self.raw_array_data)
            logger.info(f"Compiling {image_count} images...")
            export_start_time = perf_counter()
            for image in tqdm(self.raw_array_data, desc="Writing images to video"):
                processed_image = self.preprocess(image)
                self.writer.write(processed_image)

            cv2.destroyAllWindows()
            self.writer.release()
            export_end_time = perf_counter()
            export_runtime = export_end_time - export_start_time
            logger.info(
                f"Elapsed compilation time: {int(export_runtime//60)}m {round(export_runtime%60)}s"
            )
            logger.info(
                f'Wrote {image_count} images to video output "{video_file_name}"'
            )

        except Exception as e:
            logger.error(f"Failed to compile images to video: {e}")

    def render_video(self, fps: int = 15):
        """Render the image sequences as a video.

        Args:
            fps (int, optional): Video FPS; defaults to 15.
        """
        video = cv2.VideoCapture(self.video_file_path)
        video.set(cv2.CAP_PROP_FPS, fps)
        logger.info(f"Loading video from source: {self.video_file_path}")
        try:
            while video.isOpened():
                # Grab frame from video
                ret, frame = video.read()
                if not ret:
                    break
                frame = resize(frame, height=720)
                cv2.imshow("NeRF Depth Render", frame)

                key = cv2.waitKey(10)
                if key in (27, 32):
                    logger.info("Closing video...")
                    break

        except Exception as e:
            logger.error(f"Error during video playback: {e}")

        finally:
            video.release()
            cv2.destroyAllWindows()
            logger.info("Video session ended.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fps", "-f", type=int, default=15, help="Output video frame rate"
    )
    parser.add_argument(
        "--color", "-c", type=str, default="hot", help="Colormap styling"
    )
    args = parser.parse_args()

    depthmap = DepthMap(directory="samples/depthdata", color=args.color)
    depthmap.export_views(fps=args.fps)
    depthmap.render_video()
