import numpy as np
from typing import Tuple

def normalize(x, axis):
    norm = np.linalg.norm(x, axis=axis)
    x /= norm[:,:,np.newaxis]

class Camera:
    def __init__(self, aspact_ratio: float, resolution: Tuple[int, int], z: float):
        """
        :param: aspect_ratio - width/height
        :param: resolution - (width, height)
        """
        self.aspact_ratio = aspact_ratio
        self.resolution = resolution
        self.shape_3 = [resolution[0], resolution[1], 3]
        self.shape_1 = [resolution[0], resolution[1], 1]

        # (width, height, component)
        self.position_buffer = np.array([ [ [w, h, 0.0] for h in np.arange(resolution[1]) ] for w in np.arange(resolution[0])])
        self.position_buffer[:,:,0] *= aspact_ratio / resolution[0]
        self.position_buffer[:,:,1] /= resolution[1]
        self.position_buffer -= np.array([[[aspact_ratio/2, 0.5, 0.0]]])

        self.direction_buffer = self.position_buffer.copy()
        self.direction_buffer[:, :, 2] = z
        normalize(self.direction_buffer, axis=2)

        self.distance_buffer = np.zeros(self.shape_1)


    def calc_distances(self, position_buffer):
        sphere_pos = np.array([[[0.0, 0.0, 3.0]]])
        return np.linalg.norm(position_buffer - sphere_pos, axis=2) - 1.0

    def calc_normals(self) -> np.ndarray:
        normals = np.zeros(self.shape_3)
        offset = 0.01
        offset_vec_x = np.array([[[offset, 0, 0]]])
        offset_vec_y = np.array([[[0, offset, 0]]])
        offset_vec_z = np.array([[[0, 0, offset]]])

        normals[:,:,0] = self.distance_buffer[:,:,0] - np.linalg.norm(self.position_buffer + offset_vec_x, axis=2)
        normals[:,:,1] = self.distance_buffer[:,:,0] - np.linalg.norm(self.position_buffer + offset_vec_y, axis=2)
        normals[:,:,2] = self.distance_buffer[:,:,0] - np.linalg.norm(self.position_buffer + offset_vec_z, axis=2)

        normalize(normals, axis=2)

        return normals


    def run(self, n):
        for i in range(n):
            self.distance_buffer = self.calc_distances(self.position_buffer)[:,:,np.newaxis]
            self.position_buffer += self.direction_buffer * self.distance_buffer

        light_vec = np.array([[[-1.,0.5,1.]]])
        normalize(light_vec, axis=2)

        hit_mask = self.distance_buffer < 0.1
        normals = self.calc_normals()
        diffuse = np.sum(normals*light_vec, axis=2)

        return diffuse


        
if __name__ == "__main__":
    c = Camera(16/9, (16,9), 1)
    c.run(10)
    