# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple

from torch import device, inverse

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Divice: ', dev)


def normalize(x):
    norm = torch.norm(x, dim=0)
    return x / norm[np.newaxis,:]


def split_input(fn):
    def position_component_splitter(self, pos):
        x = pos[0,:]
        y = pos[1,:]
        z = pos[2,:]

        return fn(self,x,y,z)
    return position_component_splitter

def apply_transformations(fn):
    def transformation_applicator(self, pos):
        res = pos - self.trans_vec.reshape([3,1])
        res = torch.einsum('ji,ik->jk', torch.inverse(self.trans_mat), res)
        return fn(self, res)

    return transformation_applicator

def sq_norm(*components):
    return sum(el**2 for el in components)

def norm(*components):
    return torch.sqrt(sq_norm(*components))



class RenderObject:

    def __init__(self):
        self.trans_mat = torch.eye(3, device=dev)
        self.trans_vec = torch.zeros(3, device=dev).reshape([3,1,1])

    def calc_distances(self, x):
        raise NotImplementedError()

    def __add__(self, other):
        s = self
        class LambdaRenderObject(RenderObject):
            def calc_distances(self, x):
                m = torch.zeros([2,x.shape[1]], device=dev)
                return torch.min(s.calc_distances(x), other.calc_distances(x))
        return LambdaRenderObject()

class Torus(RenderObject):
    def __init__(self):
        super(Torus, self).__init__()

    @apply_transformations
    @split_input
    def calc_distances(self, x, y, z):
        return norm((norm(x, y) - 1.), z) - 0.2

class Plane(RenderObject):
    def __init__(self, normal, ppos):
        super(Plane, self).__init__()
        self.normal = torch.tensor(normal, device=dev) / norm(*torch.tensor(normal))
        self.ppos = torch.tensor(ppos, device=dev).reshape([3,1])

    @apply_transformations
    def calc_distances(self, pos):
        return torch.einsum('ij,i->j', pos-self.ppos, self.normal)

class Sphere(RenderObject):
    def __init__(self, radius):
        super(Sphere, self).__init__()
        self.radius = radius

    @apply_transformations
    def calc_distances(self, p):
        return torch.norm(p, dim=0) - self.radius



class RayTracer:
    def __init__(self, position_buffer: np.ndarray, direction_buffer: np.ndarray, scene, start_depth=0.0):
        self.scene = scene
        self.start_depth = start_depth

        self.shape_3 = position_buffer.shape
        self.shape_1 = list(self.shape_3)
        self.shape_1[0] = 1
        self.shape_1 = list(self.shape_1)

        if direction_buffer.shape != self.shape_3:
            raise Exception()

        self.direction_buffer = direction_buffer
        self.position_buffer = position_buffer + start_depth*direction_buffer

        self.distance_buffer = torch.zeros(self.shape_1, device=dev)
        self.depth_buffer = torch.zeros(self.shape_1, device=dev) + start_depth

        self.mindist_buffer = torch.zeros(self.shape_1, device=dev) + np.infty

    def run(self, iterations):
        for i in range(iterations):
            k=16.0
            self.distance_buffer = self.scene.calc_distances(self.position_buffer).reshape(self.shape_1)
            self.depth_buffer += self.distance_buffer 
            self.position_buffer += self.direction_buffer * self.distance_buffer
            self.mindist_buffer = torch.min(self.mindist_buffer, k*self.distance_buffer/self.depth_buffer)
            print(f'{100*i/iterations:2.1f} %', end='\r')

    def calc_normals(self) -> np.ndarray:
        normals = torch.zeros(self.shape_3, device=dev)
        offset = 0.01
        offset_vec_x = torch.tensor([offset, 0, 0], device=dev).reshape([3,1])
        offset_vec_y = torch.tensor([0, offset, 0], device=dev).reshape([3,1])
        offset_vec_z = torch.tensor([0, 0, offset], device=dev).reshape([3,1])

        normals[0,:] = self.distance_buffer[0,:] - self.scene.calc_distances(self.position_buffer + offset_vec_x)
        normals[1,:] = self.distance_buffer[0,:] - self.scene.calc_distances(self.position_buffer + offset_vec_y)
        normals[2,:] = self.distance_buffer[0,:] - self.scene.calc_distances(self.position_buffer + offset_vec_z)

        normals = normalize(normals)

        return normals

    def calc_hitmask(self, thershold=0.05):
        return self.distance_buffer < thershold

class Renderer:
    def __init__(self, scene: RenderObject, position_buffer, direction_buffer):
        """
        :param: aspect_ratio - width/height
        :param: resolution - (width, height)
        """

        self.shape_n = lambda n: (n, position_buffer.shape[1])
        self.shape_3 = self.shape_n(3)
        self.shape_1 = self.shape_n(1)

        self.scene = scene
        self.direct_raytrace = RayTracer(position_buffer, direction_buffer, scene)


    def calc_distances(self, position_buffer):
        return self.scene.calc_distances(position_buffer)

    def calc_reflection_directions(self, direction_buffer, normals):
        return 2*torch.sum(direction_buffer*normals, dim=0)*normals - direction_buffer

    def calc_diffuse(self, light_vec, normals, hitmask, color, shadows):
        diffuse = torch.sum(normals*light_vec, dim=0).reshape(self.shape_1)
        diffuse = torch.clamp(diffuse, 0., 1.)*shadows
        diffuse[~hitmask] = 0.0
        return color * diffuse

    def calc_specular(self, light_vec, refl_directions, hitmask, slope, color, shadows):
        specular = torch.sum(refl_directions*light_vec, dim=0).reshape(self.shape_1)
        specular = torch.pow(specular.clamp(0., 1.), slope) * shadows

        specular[~hitmask] = 0.0
        return specular * color 
    def calc_ambient(self, hitmask, color):
        ambient = torch.zeros(self.shape_1, device=dev)
        ambient[hitmask] = 1.
        return ambient * color

    def calc_fresnel(self, normals, hitmask, direction_buffer, color):
        fresnel = 1. - torch.sum(direction_buffer*normals, dim=0).clamp(0.,1.).reshape(self.shape_1)
        fresnel[~hitmask] = 0.0
        return fresnel * color

    def calc_shadow(self, light_vec, iterations):
        shadow_directions = torch.zeros(self.shape_3, device=dev)
        shadow_directions[:,:] = -light_vec
        shadow_raytrace = RayTracer(self.direct_raytrace.position_buffer.clone(), shadow_directions, self.scene, 0.1)
        shadow_raytrace.run(iterations)
        shadow_mask = shadow_raytrace.calc_hitmask(0.01)
        shadows = shadow_raytrace.mindist_buffer.clamp(0.,1.0)
        shadows[shadow_mask] = 0.0
        shadows[torch.isnan(shadows)] = 1.0
        return shadows

    def calc_space_texture(self):
        k = 1.0
        res = torch.sign(torch.remainder(self.direct_raytrace.position_buffer[0,:], k)-k/2)
        res *= torch.sign(torch.remainder(self.direct_raytrace.position_buffer[1,:], k)-k/2)
        res *= torch.sign(torch.remainder(self.direct_raytrace.position_buffer[2,:], k)-k/2)
        res += 1.0
        res *= 0.5
        return res

    def calc_image(self, layer, weights, texture):
        return torch.einsum('ij,i,j->j', layer, torch.tensor(weights, device=dev), texture).clamp(0., 1.)

    def run(self, iterations, reflection_depth):
        self.direct_raytrace.run(iterations)

        light_vec = torch.tensor([0.1, 0.5, .2], device=dev).reshape([3,1])
        light_vec = normalize(light_vec)

        hitmask = self.direct_raytrace.calc_hitmask(0.01)
        invhitmask_float = torch.zeros(hitmask.shape, device=dev)
        invhitmask_float[~hitmask] = 1.0

        normals = self.direct_raytrace.calc_normals()
        refl_directions = self.calc_reflection_directions(normals, self.direct_raytrace.direction_buffer)

        colors = torch.zeros(self.shape_3, device=dev)


        if reflection_depth != 0:
            reflection_renderer = Renderer(self.scene, self.direct_raytrace.position_buffer.clone() + 0.1*refl_directions, refl_directions)
            reflection_color = reflection_renderer.run(100, reflection_depth-1)*0.2
            reflection_color[:, ~hitmask[0,:]] = 0.0

            colors += reflection_color

        diffuse, specular, ambient, fresnel =  [.4, .2, .2, .1,]

        white = red = torch.tensor([1.,1.,1.], device=dev).reshape([3,1])
        red = torch.tensor([1.,0.,0.], device=dev).reshape([3,1])
        cornflower_blue = red = torch.tensor([100.,149.,237.], device=dev).reshape([3,1])/255.

        shadows = self.calc_shadow(light_vec, 100)
        texture = 0.1*self.calc_space_texture()+0.9
        texture = white *texture

        background = torch.zeros(self.shape_1, device=dev)
        background[~hitmask] = 1.0

        colors += background * cornflower_blue
        colors += diffuse * self.calc_diffuse(light_vec, normals, hitmask, texture, shadows)
        colors += specular * self.calc_specular(light_vec, refl_directions , hitmask, 100., texture, shadows) * shadows
        colors += ambient * self.calc_ambient(hitmask, white)
        colors += fresnel * self.calc_fresnel(normals, hitmask, self.direct_raytrace.direction_buffer, red)

        
        return colors

class Camera:
    def __init__(self, aspact_ratio: float, resolution: Tuple[int, int], z: float, scene: RenderObject):
        self.aspact_ratio = aspact_ratio
        self.scene = scene
        self.resolution = resolution
        self.shape_3 = [3, resolution[0]*resolution[1]]
        self.shape_1 = [1, resolution[0]*resolution[1]]
        self.shape_n = lambda n: (n, resolution[0]*resolution[1])

        # ([x, y, z], width, height)
        position_buffer = torch.zeros([3, resolution[0],resolution[1]], device=dev)
        position_buffer[0,:,:] = torch.linspace(-aspact_ratio/2, aspact_ratio/2, steps=resolution[0])[:, np.newaxis]
        position_buffer[1,:,:] = torch.linspace(-.5, .5, steps=resolution[1])[:, np.newaxis].T
        position_buffer = position_buffer.reshape(self.shape_3)
        
        direction_buffer = position_buffer.clone()
        direction_buffer[2, :] = z
        direction_buffer = normalize(direction_buffer)

        self.renderer = Renderer(scene, position_buffer, direction_buffer)

    def run(self):
        return self.renderer.run(120, 1).clamp(0.,1.).reshape([3,*self.resolution]).cpu()


s = Sphere( 0.9)
s.trans_vec = torch.tensor([-0.3, 0.0, 4.5], device=dev)

t = Torus()
t.trans_vec = torch.tensor([0., -1.2, 3.], device=dev)
t.trans_mat = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]],device=dev, dtype=torch.float32)

p = Plane([0.,-1.,-0.1], [0.,1.2,0.])
scene = t + s + p

n=1
r=(1920,1080)
c = Camera(r[0]/r[1], (int(r[0]*n),int(r[1]*n)), 1, scene)
d = c.run()

plt.imshow(d.T)
plt.show()
