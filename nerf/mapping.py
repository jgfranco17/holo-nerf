import os
import cv2
import numpy as np
from tqdm import tqdm
from time import perf_counter


class DepthMap(object):
    def __init__(self, directory:str, color:str="hot", export:bool=False) -> None:
        # Set map style
        self.map_style = {
            "autumn": cv2.COLORMAP_AUTUMN,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "bone": cv2.COLORMAP_BONE,
            "hsv": cv2.COLORMAP_HSV,
            "ocean": cv2.COLORMAP_OCEAN,
            "hot": cv2.COLORMAP_HOT
        }
        self.map_color = self.map_style.get(color, cv2.COLORMAP_HOT)
        
        # Compile, preprocess, and set the map image
        self.directory = os.path.join(os.getcwd(), directory)
        self.raw_array_data, self.array_compiled = self.__get_np_data(self.directory)
        self.image = self.preprocess(self.array_compiled)
        self.writer = None
        self.export = export
        
    @property
    def color(self):
        return self.map_color
    
    @color.setter
    def set_map_color(self, color):
        if color not in self.map_style.keys():
            raise ValueError(f'Invalid color key \"{color}\" selected.')
        self.map_color = self.map_style[color]
        print(f'Color map set to \"{color}\".')
        
    @staticmethod
    def __get_np_data(source:str) -> tuple:
        """_summary_

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
            print(f'Reading from: {source}')
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
        
            print(f'Loaded in {count} array files.')
            return np_array_list, np.around((np_array_data / count), decimals=0).astype(int)
        
        except Exception as e:
            print(f'Failed to load NPY data: {e}')
    
    @staticmethod
    def __normalize(image:np.ndarray, bits:int) -> np.ndarray:
        """
        Normalize the given map for OpenCV.

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
    def __resize(image, factor:float=1.0) -> np.ndarray:
        """
        Scale an image evenly by a given factor.

        Args:
            image (np.ndarray): Image to scale
            factor (float, optional): Scaling factor, defaults to 1

        Returns:
            np.ndarray: Resized image
        """
        height, width, _ = image.shape
        new_dimensions = (round(width * factor), round(height * factor))
        print(f'Scaled image {"up" if factor > 1 else "down"} to {factor*100}%')
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def __invert(array:np.ndarray) -> np.ndarray:
        """
        Invert the given array.

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
    
    def colormap(self, image:np.ndarray) -> np.ndarray:
        """
        Recolor the depth map from grayscale to colored.

        Args:
            image (np.ndarray): Grayscale image

        Returns:
            np.ndarray: Colored map image
        """
        depth_map = self.__normalize(image, bits=2)
        depth_map = (depth_map/256).astype(np.uint8)
        return cv2.applyColorMap(depth_map, self.map_color)
    
    def preprocess(self, image) -> np.ndarray:
        """
        Preprocess the depth map from raw array to image layout.

        Args:
            image (np.ndarray): Raw image array

        Returns:
            np.ndarray: Colored depth map
        """
        base_array = self.__invert(image)
        return self.colormap(base_array)
    
    def display(self, scale:float=1.0):
        """
        Display the array data as an image.

        Args:
            scale (float, optional): Sizing scale factor, defaults to 1.0
        """
        try:
            image = self.__resize(self.image, factor=scale)
            x_dim, y_dim, _ = self.image.shape
            print("Displaying compiled depth map...")
            cv2.imshow(f'NeRF Depth Map, {x_dim}x{y_dim}', image)
            cv2.waitKey(0) 
            
        except Exception as e:
            print(f'Failed to display depth map: {e}')
            
        finally:
            cv2.destroyAllWindows()
            
    def export_views(self, fps:int=15):
        """
        Export the compiled depth maps into video format.

        Args:
            fps (int, optional): Exported video frame rate, defaults to 15
        """
        # Set export video attributes
        video_file_name = f'depthmap_nerf_{len(self.raw_array_data)}_fps{fps}.avi'
        video_file_path = os.path.join(os.getcwd(), "videos", video_file_name)
        if os.path.exists(video_file_path):
            print(f'Video file already exists: {video_file_path}')
            
        else:
            try:
                # Set video writer
                os.makedirs(os.path.join(os.getcwd(), "videos"), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"DIVX")
                height, width, _ = self.image.shape
                self.writer = cv2.VideoWriter(video_file_path, fourcc, fps, (width, height))
                
                # Compile all arrays into image format and export
                image_count = len(self.raw_array_data)
                print(f'Compiling {image_count} images...')
                export_start_time = perf_counter()
                for image in tqdm(self.raw_array_data, desc="Writing images to video"):
                    processed_image = self.preprocess(image)
                    self.writer.write(processed_image)
                
                cv2.destroyAllWindows()
                self.writer.release()
                export_end_time = perf_counter()
                export_runtime = export_end_time - export_start_time
                print(f'Elapsed compilation time: {int(export_runtime//60)}m {round(export_runtime%60)}s')
                print(f'Wrote {image_count} images to video output \"{video_file_name}\"')
                
            except Exception as e:
                print(f'Failed to compile images to video: {e}')
        
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", "-f",
                        type=int,  
                        default=15, 
                        help="Output video frame rate")
    parser.add_argument("--color", "-c",
                        type=str, 
                        default="hot", 
                        help="Colormap styling")
    args = parser.parse_args()
    
    depthmap = DepthMap(directory="samples/depthdata", color=args.color)
    # depthmap.display(scale=0.8)
    depthmap.export_views(fps=args.fps)