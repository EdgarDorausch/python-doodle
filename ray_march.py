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



class RayTracer:
    def __init__(self, position_buffer: np.ndarray, direction_buffer: np.ndarray, scene):
        self.scene = scene

        self.shape_3 = position_buffer.shape
        self.shape_1 = list(self.shape_3)
        self.shape_1[0] = 1
        self.shape_1 = list(self.shape_1)

        if direction_buffer.shape != self.shape_3:
            raise Exception()

        self.position_buffer = position_buffer
        self.direction_buffer = direction_buffer

        self.distance_buffer = np.zeros(self.shape_1)
        self.depth_buffer = np.zeros(self.shape_1)

    def run(self, iterations):
        for i in range(iterations):
            self.distance_buffer = self.scene.calc_distances(self.position_buffer).reshape(self.shape_1)
            self.depth_buffer += self.distance_buffer 
            self.position_buffer += self.direction_buffer * self.distance_buffer

    def calc_normals(self) -> np.ndarray:
        normals = np.zeros(self.shape_3)
        offset = 0.01
        offset_vec_x = np.array([offset, 0, 0]).reshape([3,1,1])
        offset_vec_y = np.array([0, offset, 0]).reshape([3,1,1])
        offset_vec_z = np.array([0, 0, offset]).reshape([3,1,1])

        normals[0,:,:] = self.distance_buffer[0,:,:] - self.scene.calc_distances(self.position_buffer + offset_vec_x)
        normals[1,:,:] = self.distance_buffer[0,:,:] - self.scene.calc_distances(self.position_buffer + offset_vec_y)
        normals[2,:,:] = self.distance_buffer[0,:,:] - self.scene.calc_distances(self.position_buffer + offset_vec_z)

        normals = normalize(normals, axis=0)

        return normals

    def calc_hitmask(self):
        return self.distance_buffer[0,:,:] < 0.8

class Renderer:
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
        self.shape_n = lambda n: (n, resolution[0], resolution[1])

        # ([x, y, z], width, height)
        position_buffer = np.zeros(self.shape_3)
        position_buffer[0,:,:] = np.linspace(-aspact_ratio/2, aspact_ratio/2, num=resolution[0])[:, np.newaxis]
        position_buffer[1,:,:] = np.linspace(-.5, .5, num=resolution[1])[:, np.newaxis].T
        
        direction_buffer = position_buffer.copy()
        direction_buffer[2, :, :] = z
        direction_buffer = normalize(direction_buffer, axis=0)

        self.direct_raytrace = RayTracer(position_buffer, direction_buffer, scene)


    def calc_distances(self, position_buffer):
        return self.scene.calc_distances(position_buffer)


    def calc_diffuse(self, light_vec, normals, hitmask):
        diffuse = np.sum(normals*light_vec, axis=0)
        diffuse = np.clip(diffuse, 0., 1.)
        diffuse[~hitmask] = 0.0
        return diffuse

    def calc_specular(self, light_vec, normals, hitmask, direction_buffer):
        r = 2*np.sum(direction_buffer*normals, axis=0)*normals - direction_buffer
        specular = np.sum(r*light_vec, axis=0)
        specular = np.power(specular.clip(0., 1.), 15)
        specular[~hitmask] = 0.0
        return specular

    def calc_ambient(self, hitmask):
        ambient = np.zeros(hitmask.shape)
        ambient[hitmask] = 1.
        return ambient

    def calc_fresnel(self, normals, hitmask, direction_buffer):
        fresnel = 1. - np.sum(direction_buffer*normals, axis=0).clip(0.,1.)
        fresnel[~hitmask] = 0.0
        return fresnel

    def calc_image(self, layer, weights):
        return np.einsum('ijk,i->jk', layer, np.array(weights)).clip(0., 1.)

    def run(self, iterations):
        self.direct_raytrace.run(iterations)

        light_vec = np.array([0.7, 0.3, 1.]).reshape([3,1,1])
        light_vec = normalize(light_vec, axis=0)

        hitmask = self.direct_raytrace.calc_hitmask()
        normals = self.direct_raytrace.calc_normals()

        layers = np.zeros(self.shape_n(4))

        layers[0,:,:] = self.calc_diffuse(light_vec, normals, hitmask)
        layers[1,:,:]  = self.calc_specular(light_vec, normals, hitmask, self.direct_raytrace.direction_buffer)
        layers[2,:,:]  = self.calc_ambient(hitmask)
        layers[3,:,:]  = self.calc_fresnel(normals, hitmask, self.direct_raytrace.direction_buffer)

        

        return self.calc_image(layers, [.5, .2, .1, .2])

     
scene = Torus() + Sphere( .5, [0.,0.,3.]) + Plane([0.,0.,-1.], [0.,0.,6.])

n=100
c = Renderer(4/3, (4*n,3*n), 1, scene)
d = c.run(150)


# %%
from PIL import Image
from matplotlib import cm

im = Image.fromarray(np.uint8(cm.gray(d.T)*255))
im.show()
