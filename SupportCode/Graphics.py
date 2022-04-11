import numpy as np
import itertools
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch, Arc


class cube():
    def __init__(self, length, width, heigth, equal_scale=False, facecolors='k', linewidths=1, edgecolors='k', alpha=0.25, scale_mult=2, rotation_angles=(0, 0, 0)):
        self.length = length
        self.width = width
        self.heigth = heigth
        self.fc = facecolors
        self.lw = linewidths
        self.ec = edgecolors
        self.a = alpha
        self.angles = rotation_angles
        self.xlim = (-scale_mult * length - length / 2, scale_mult * length + length / 2)
        self.ylim = (-scale_mult * width - width / 2, scale_mult * width + width / 2)
        self.zlim = (-scale_mult * heigth - heigth / 2, scale_mult * heigth + heigth / 2)
        if equal_scale:
            x_range = np.abs(self.xlim[1] - self.xlim[0])
            x_middle = np.mean(self.xlim)
            y_range = np.abs(self.ylim[1] - self.ylim[0])
            y_middle = np.mean(self.ylim)
            z_range = np.abs(self.zlim[1] - self.zlim[0])
            z_middle = np.mean(self.zlim)
            plot_radius = 0.5*max([x_range, y_range, z_range])
            self.xlim = (x_middle - plot_radius, x_middle + plot_radius)
            self.ylim = (y_middle - plot_radius, y_middle + plot_radius)
            self.zlim = (z_middle - plot_radius, z_middle + plot_radius)
        pass

    def rotation_matrix(self):
        a, b, g = self.angles
        m1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        m2 = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        m3 = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
        return m3.dot(m2).dot(m1)

    def corners(self):
        rot_matrix = self.rotation_matrix()
        a1 = rot_matrix.dot(np.array([self.length / 2, -self.width / 2, -self.heigth / 2]))
        b1 = rot_matrix.dot(np.array([self.length / 2, self.width / 2, -self.heigth / 2]))
        c1 = rot_matrix.dot(np.array([-self.length / 2, self.width / 2, -self.heigth / 2]))
        d1 = rot_matrix.dot(np.array([-self.length / 2, -self.width / 2, -self.heigth / 2]))
        a2 = rot_matrix.dot(np.array([self.length / 2, -self.width / 2, self.heigth / 2]))
        b2 = rot_matrix.dot(np.array([self.length / 2, self.width / 2, self.heigth / 2]))
        c2 = rot_matrix.dot(np.array([-self.length / 2, self.width / 2, self.heigth / 2]))
        d2 = rot_matrix.dot(np.array([-self.length / 2, -self.width / 2, self.heigth / 2]))
        return [a1, b1, c1, d1, a2, b2, c2, d2]

    def verts(self, Z):
        return [[Z[0] ,Z[1] ,Z[2] ,Z[3]], [Z[4], Z[5], Z[6], Z[7]],
                [Z[0], Z[1], Z[5], Z[4]], [Z[2], Z[3], Z[7], Z[6]],
                [Z[1], Z[2], Z[6], Z[5]], [Z[4], Z[7], Z[3], Z[0]]]

    def collection(self):
        return Poly3DCollection(self.verts(self.corners()), facecolors=self.fc, linewidths=self.lw, edgecolors=self.ec, alpha=self.a)

    pass


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, rotation_angles=(0, 0, 0), mutation_scale=10, lw=2, arrowstyle='-|>', color='k', **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, mutation_scale=mutation_scale, lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
        self.angles = rotation_angles
        self.verts3d = xs, ys, zs
        pass

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self.verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)
    
    def verts(self):
        rot_matrix = self.rotation_matrix()
        return zip(*[rot_matrix.dot(np.array([i, j, k])) for i, j, k in zip(*self.verts3d)])
    
    def rotation_matrix(self):
        a, b, g = self.angles
        m1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        m2 = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        m3 = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
        return m3.dot(m2).dot(m1)
    
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self.verts()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        pass

    pass


class plane():
    def __init__(self, normal, length=1, equal_scale=False, facecolors='k', linewidths=1, edgecolors='k', alpha=0.25, num_err=1e-8):
        self.normal = normal
        self.length = length
        self.fc = facecolors
        self.lw = linewidths
        self.ec = edgecolors
        self.a = alpha
        self.num_err = num_err
        self.verts3d = self.verts()
        xx, yy, zz = [*zip(*self.verts3d)]
        self.xlim = (min(xx), max(xx))
        self.ylim = (min(yy), max(yy))
        self.zlim = (min(zz), max(zz))
        if equal_scale:
            x_range = abs(self.xlim[1] - self.xlim[0])
            x_middle = np.mean(self.xlim)
            y_range = abs(self.ylim[1] - self.ylim[0])
            y_middle = np.mean(self.ylim)
            z_range = abs(self.zlim[1] - self.zlim[0])
            z_middle = np.mean(self.zlim)
            plot_radius = 0.5*max([x_range, y_range, z_range])
            self.xlim = (x_middle - plot_radius, x_middle + plot_radius)
            self.ylim = (y_middle - plot_radius, y_middle + plot_radius)
            self.zlim = (z_middle - plot_radius, z_middle + plot_radius)
    
    def init_vector(self):
        for vec in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            if not abs(np.inner(self.normal, vec)) / (np.linalg.norm(self.normal) * np.linalg.norm(vec)) > 1 - self.num_err:
                return vec
    
    def verts(self):
        w = np.cross(self.init_vector(), self.normal)
        w = (w / np.linalg.norm(w))
        u = np.cross(w, self.normal)
        u = (u / np.linalg.norm(u))
        c0 = self.length / 2 * (u + w)
        c1 = self.length / 2 * (u - w)
        c2 = self.length / 2 * (- u + w)
        c3 = self.length / 2 * (- u - w)
        return [c0, c1, c3, c2]

    def collection(self):
        return Poly3DCollection([self.verts3d], facecolors=self.fc, linewidths=self.lw, edgecolors=self.ec, alpha=self.a)

    pass


class cube_plane():
    def __init__(self, length, width, heigth, equal_scale=False, facecolors='k', linewidths=1, edgecolors='k', alpha=0.25, scale_mult=2, rotation_angles=(0, 0, 0), num_err=1e-8):
        self.length = length
        self.width = width
        self.heigth = heigth
        self.fc = facecolors
        self.lw = linewidths
        self.ec = edgecolors
        self.a = alpha
        self.angles = rotation_angles
        self.num_err = num_err
        self.vertices = self.cube_vertices()
        self.planes = self.cube_planes()
        self.xlim = (-scale_mult * length - length / 2, scale_mult * length + length / 2)
        self.ylim = (-scale_mult * width - width / 2, scale_mult * width + width / 2)
        self.zlim = (-scale_mult * heigth - heigth / 2, scale_mult * heigth + heigth / 2)
        if equal_scale:
            x_range = abs(self.xlim[1] - self.xlim[0])
            x_middle = np.mean(self.xlim)
            y_range = abs(self.ylim[1] - self.ylim[0])
            y_middle = np.mean(self.ylim)
            z_range = abs(self.zlim[1] - self.zlim[0])
            z_middle = np.mean(self.zlim)
            plot_radius = 0.5*max([x_range, y_range, z_range])
            self.xlim = (x_middle - plot_radius, x_middle + plot_radius)
            self.ylim = (y_middle - plot_radius, y_middle + plot_radius)
            self.zlim = (z_middle - plot_radius, z_middle + plot_radius)
        pass

    def rotation_matrix(self):
        a, b, g = self.angles
        m1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        m2 = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        m3 = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
        return m3.dot(m2).dot(m1)

    def cube_vertices(self):
        rot_matrix = self.rotation_matrix()
        a1 = rot_matrix.dot(np.array([self.length / 2, -self.width / 2, -self.heigth / 2]))
        b1 = rot_matrix.dot(np.array([self.length / 2, self.width / 2, -self.heigth / 2]))
        c1 = rot_matrix.dot(np.array([-self.length / 2, self.width / 2, -self.heigth / 2]))
        d1 = rot_matrix.dot(np.array([-self.length / 2, -self.width / 2, -self.heigth / 2]))
        a2 = rot_matrix.dot(np.array([self.length / 2, -self.width / 2, self.heigth / 2]))
        b2 = rot_matrix.dot(np.array([self.length / 2, self.width / 2, self.heigth / 2]))
        c2 = rot_matrix.dot(np.array([-self.length / 2, self.width / 2, self.heigth / 2]))
        d2 = rot_matrix.dot(np.array([-self.length / 2, -self.width / 2, self.heigth / 2]))
        return [a1, b1, c1, d1, a2, b2, c2, d2]

    def cube_planes(self):
        return [[self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]],
                [self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]],
                [self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4]],
                [self.vertices[2], self.vertices[3], self.vertices[7], self.vertices[6]],
                [self.vertices[1], self.vertices[2], self.vertices[6], self.vertices[5]],
                [self.vertices[4], self.vertices[7], self.vertices[3], self.vertices[0]]]

    def cube_collection(self):
        return Poly3DCollection(self.planes, facecolors=self.fc, linewidths=self.lw, edgecolors=self.ec, alpha=self.a)

    def init_vector(self):
        for vec in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            if not abs(np.inner(self.normal, vec)) / (np.linalg.norm(self.normal) * np.linalg.norm(vec)) > 1 - self.num_err:
                return vec

    def plane_equation(self, poly):
        v1 = poly[1] - poly[0]
        v2 = poly[2] - poly[1]
        normal = np.cross(v1, v2)
        d = -np.array(poly[0]).dot(normal)
        return np.append(normal, d)

    def plane_intersection(self, a, b):
        a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
        aXb_vec = np.cross(a_vec, b_vec)
        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a[3], -b[3], 0.]).reshape(3,1)
        if abs(np.linalg.det(A)) > self.num_err:
            p_inter = np.linalg.solve(A, d).T
            return p_inter[0], (p_inter + aXb_vec)[0]
        else:
            return None, None

    def new_basis(self):
        vec0 = self.init_vector()
        vec0 = vec0 / np.linalg.norm(vec0)
        vec1 = np.cross(self.normal, vec0)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = np.cross(self.normal, vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return np.linalg.inv(np.array([vec1, vec2, self.normal]).T)

    def clockwiseangle_sort(self, point):
        point = point[0]
        origin = [0, 0]
        refvec = [1, 0]
        vector = np.array([point[0] - origin[0], point[1] - origin[1]])
        lenvector = np.linalg.norm(vector)
        if lenvector == 0:
            return -np.pi, 0
        normalized = vector / lenvector
        angle = np.arctan2(refvec[1] * normalized[0] - refvec[0] * normalized[1], np.inner(normalized, refvec))
        if angle < 0:
            return 2 * np.pi + angle, lenvector
        return angle, lenvector

    def inter_collection(self, normal=(1, 1, 1)):
        self.normal = np.array(normal)
        polygon = []
        sorting = []
        LTM = self.new_basis()
        for plane in self.planes:
            a, b = self.plane_intersection(np.append(self.normal, 0), self.plane_equation(plane))
            if a is not None:
                for c, d in zip([plane[0], plane[1], plane[2], plane[3]], [plane[1], plane[2], plane[3], plane[0]]):
                    for i, j in itertools.combinations([0, 1, 2], 2):
                        M = np.array([[b[i] - a[i], c[i] - d[i]], [b[j] - a[j], c[j] - d[j]]])
                        B = np.array([[c[i] - a[i]], [c[j] - a[j]]])
                        if abs(np.linalg.det(M)) > self.num_err:
                            params = np.linalg.inv(M).dot(B)
                            x, y, z = params[0] * (b - a) + a
                            if all([abs(2 * x) <= self.length + self.num_err,
                                    abs(2 * y) <= self.width + self.num_err,
                                    abs(2 * z) <= self.heigth + self.num_err]):
                                polygon.append((x, y, z))
                                vecB = LTM.dot(np.array([x, y, z]).T)
                                sorting.append(vecB[:2])
                            break
        sorting, polygon = zip(*sorted(zip(sorting, polygon), key=self.clockwiseangle_sort))
        polygon = list(polygon)
        polygon.append(polygon[0])
        return Poly3DCollection([polygon], facecolors='gold', linewidths=1, edgecolors='gold', alpha=0.5)

    pass


def plot_angle_arc(axis, x0, y0, theta1, theta2, rad, num=1, inc=0.1, color='k'):
    inc *= rad
    for r in np.linspace(rad, rad + (num - 1) * inc, num):
        axis.add_patch(Arc((x0, y0), width=r, height=r, theta1=theta1, theta2=theta2, color=color))
    pass

