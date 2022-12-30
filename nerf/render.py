import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from pointcloud import PointCloudVisualizer


class NeRFCloud(object):
    def __init__(self, depthdir:str, colordir:str, xfactor:float=1.0, yfactor:float=1.0, zfactor:float=1.0):
        """
        Model architecture for the NeRF point cloud visualizer.

        Args:
            depthdir (str): Directory containing the depth arrays
            colordir (str): Directory containing the color arrays
            xfactor (float, optional): x-axis scaling factor. Defaults to 1.0.
            yfactor (float, optional): y-axis scaling factor. Defaults to 1.0.
            zfactor (float, optional): z-axis scaling factor. Defaults to 1.0.
        """
        self.depth_files = os.path.join(os.getcwd(), depthdir)
        self.color_files = os.path.join(os.getcwd(), colordir)
        self.visualizer = PointCloudVisualizer()
        self.depthmap = None
        self.xfactor = xfactor
        self.yfactor = yfactor
        self.zfactor = zfactor
        
        # Load NPY files
        self.depths, self.compiled_depth = self.__get_np_data(self.depth_files, size=2)
        self.colors, self.compiled_color = self.__get_np_data(self.color_files, size=3)

    @staticmethod
    def __get_np_data(source:str, size:int=2) -> tuple:
        """
        Compile NPY data from directory.

        Args:
            source (str): Directory filepath to load the data from

        Returns:
            tuple: list of NumPy arrays, compiled array data
        """
        # Initialize blank array
        np_array_list = []
        array_size = (1024, 1024) if size == 2 else (1024, 1024, 3)
        np_array_data = np.zeros(array_size, dtype=float)
        
        # Go through all files in target directory
        try:
            print(f'Reading from: {source}')
            storage_dir = os.listdir(source)
            count = 0
            for filename in tqdm(storage_dir, desc="Loading files"):
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

    def get_image_data(self):
        preprocessed_array = self.__invert(np.around(self.compiled_depth, decimals=0).astype(int))
        preprocessed_colors = np.around((self.compiled_color / 256), decimals=3)
        raw_depth_map = []
        raw_color_map = []
        depth_rows, depth_cols = preprocessed_array.shape
        color_rows, color_cols, _ = preprocessed_colors.shape
        
        for row in tqdm(range(depth_rows), desc="Processing depth matrix"):
            for col in range(depth_cols):
                z = int(preprocessed_array[row, col])
                raw_depth_map.append([row, col, z])
        processed_depth_map = np.array(raw_depth_map)
        
        for row in tqdm(range(color_rows), desc="Processing color matrix"):
            for col in range(color_cols):
                raw_color_map.append(list(preprocessed_colors[row, col]))
        processed_color_map = np.array(raw_color_map)
        
        return processed_depth_map, processed_color_map
    
    def export_scene_data(self):
        export_data = []
        coordinates, color = self.get_image_data()
        color = np.around(color * 256).astype(int)
        
        count, dimension = coordinates.shape
        for point in tqdm(range(count), desc="Preparing dataframe"):
            export_data.append(list(coordinates[point]+1) + list(color[point]))
           
        output_directory = os.path.join(os.getcwd(), "exports") 
        filename = f'nerf_scene_{"x".join([str(num) for num in self.compiled_depth.shape])}.csv'
        os.makedirs(output_directory, exist_ok=True)
        exported_file = os.path.join(output_directory, filename)
        headers = ["X", "Y", "Z", "R", "G", "B"]
        print(f'Exporting data to file \"{filename}\"')
        df = pd.DataFrame(export_data, columns=headers)
        df.to_csv(exported_file, index=False)
        print(f'Exported {len(coordinates):,} scene points to: \"{exported_file}\"')
        
    def render_scene(self):
        """
        Render the point cloud.
        """
        # Render map as point cloud
        depth, color = self.get_image_data()
        self.visualizer.draw_cloud(array=depth, colors=color)


if __name__ == "__main__":
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            "--source", "-s",
            type=str, required=True,
            help="The source directory to get array data from"
        )
        parser.add_argument(
            "--colors", "-c",
            type=str, required=False,
            help="The source directory to get color data from"
        )
        parser.add_argument(
            "--xfactor", "-x",
            type=float, required=False,
            default=1.0,
            help="Scale factor for x-axis"
        )
        parser.add_argument(
            "--yfactor", "-y",
            type=float, required=False,
            default=1.0,
            help="Scale factor for y-axis"
        )
        parser.add_argument(
            "--zfactor", "-z",
            type=float, required=False,
            default=1.0,
            help="Scale factor for z-axis"
        )
        return parser

    parser = get_parser()
    args = parser.parse_args()
    
    cloud_data = NeRFCloud(depthdir=args.source, 
                           colordir=args.colors,
                           xfactor=args.xfactor,
                           yfactor=args.yfactor,
                           zfactor=args.zfactor)
    cloud_data.render_scene()
    # cloud_data.export_scene_data()
    