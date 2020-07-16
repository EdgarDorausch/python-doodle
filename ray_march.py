# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def normalize(x, axis=0):
    norm = np.linalg.norm(x,2, axis=axis)
    return x / norm[np.newaxis,:,:]


def split_input(fn):
    def position_component_splitter(self, pos):
        x = pos[0,:,:]
        y = pos[1,:,:]
        z = pos[2,:,:]

        return fn(self,x,y,z)
    return position_component_splitter

def sq_norm(*components):
    return sum(el**2 for el in components)

def norm(*components):
    return np.sqrt(sq_norm(*components))



class RenderObject:
    def calc_distances(self, x):
        raise NotImplementedError()

    def __add__(self, other):
        s = self
        class LambdaRenderObject(RenderObject):
            def calc_distances(self, x):
                m = np.zeros([2,x.shape[1], x.shape[2]])
                m[0,:,:] = s.calc_distances(x)
                m[1,:,:] = other.calc_distances(x)

                return np.min(m, axis=0)
        return LambdaRenderObject()

class Torus(RenderObject):
    def __init__(self):
        pass

    @split_input
    def calc_distances(self, x, y, z):
        z_ = z - 3.
        return norm((norm(x, y) - 1.), z_) - 0.1

class Plane(RenderObject):
    def __init__(self, normal, ppos):
        self.normal = np.array(normal) / norm(*normal)
        self.ppos = np.array(ppos).reshape([3,1,1])

    def calc_distances(self, pos):
        return np.einsum('ijk,i->jk', pos-self.ppos, self.normal)

class Sphere(RenderObject):
    def __init__(self, radius, position):
        self.radius = radius
        self.position = np.array(position).reshape([3,1,1])

    def calc_distances(self, p):
        return np.linalg.norm(p - self.position, axis=0) - self.radius



class Camera:
    def __init__(self, aspact_ratio: float, resolution: Tuple[int, int], z: float, scene: RenderObject):
        """
        :param: aspect_ratio - width/height
        :param: resolution - (width, height)
        """
        self.aspact_ratio = aspact_ratio
        self.scene = scene
        self.resolution = resolution
        self.shape_3 = [3, resolution[0], resolution[1]]
        self.shape_1 = [1, resolution[0], resolution[1]]

        # ([x, y, z], width, height)
        self.position_buffer = np.zeros(self.shape_3)
        self.position_buffer[0,:,:] = np.linspace(-aspact_ratio/2, aspact_ratio/2, num=resolution[0])[:, np.newaxis]
        self.position_buffer[1,:,:] = np.linspace(-.5, .5, num=resolution[1])[:, np.newaxis].T
        
        self.direction_buffer = self.position_buffer.copy()
        self.direction_buffer[2, :, :] = z
        self.direction_buffer = normalize(self.direction_buffer, axis=0)

        self.distance_buffer = np.zeros(self.shape_1)


    def calc_distances(self, position_buffer):
        return self.scene.calc_distances(position_buffer)

    def calc_normals(self) -> np.ndarray:
        normals = np.zeros(self.shape_3)
        offset = 0.01
        offset_vec_x = np.array([offset, 0, 0]).reshape([3,1,1])
        offset_vec_y = np.array([0, offset, 0]).reshape([3,1,1])
        offset_vec_z = np.array([0, 0, offset]).reshape([3,1,1])

        normals[0,:,:] = self.distance_buffer[0,:,:] - self.calc_distances(self.position_buffer + offset_vec_x)
        normals[1,:,:] = self.distance_buffer[0,:,:] - self.calc_distances(self.position_buffer + offset_vec_y)
        normals[2,:,:] = self.distance_buffer[0,:,:] - self.calc_distances(self.position_buffer + offset_vec_z)

        normals = normalize(normals, axis=0)

        return normals

    def run(self, n):
        for i in range(n):
            self.distance_buffer = self.calc_distances(self.position_buffer)[np.newaxis,:,:]
            self.position_buffer += self.direction_buffer * self.distance_buffer

        light_vec = np.array([0.7, 0.3, 1.]).reshape([3,1,1])
        light_vec = normalize(light_vec, axis=0)

        hit_mask = self.distance_buffer[0,:,:] < 0.005

        normals = self.calc_normals()
        # print(normals)
        diffuse = np.sum(normals*light_vec, axis=0)
        diffuse = np.clip(diffuse, 0., 1.)
        diffuse[~hit_mask] = 0.0

        r = 2*np.sum(self.direction_buffer*normals, axis=0)*normals - self.direction_buffer
        specular = np.sum(r*light_vec, axis=0)
        specular = np.power(specular.clip(0., 1.), 15)
        specular[~hit_mask] = 0.0
        print(specular.shape)

        ambient = np.zeros(specular.shape)
        ambient[hit_mask] = 1.

        return (0.5*diffuse + 0.3*specular + 0.2*ambient).clip(0., 1.)

     
scene = Torus() + Sphere( .5, [0.,0.,3.]) + Plane([0.,0.,-1.], [0.,0.,5.])

n=200
c = Camera(4/3, (4*n,3*n), 1, scene)
d = c.run(30)


# %%
from PIL import Image
from matplotlib import cm

im = Image.fromarray(np.uint8(cm.cividis(d.T)*255))
im.show()
