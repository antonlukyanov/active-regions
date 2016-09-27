import math
import os
import os.path as path
from hashlib import md5
import pickle
from typing import Union, List, Tuple, Dict

from skimage import io
from skimage import img_as_float
from skimage.filters import gaussian
from skimage.color import rgb2gray
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import networkx as nx
from networkx.linalg import normalized_laplacian_matrix
from networkx.linalg import laplacian_spectrum
from networkx.linalg import adjacency_spectrum
from networkx.classes.function import density

import utils as utl

nb8 = []
for i in [1, -1, 0]:
    for j in [1, -1, 0]:
        if i != 0 or j != 0:
            nb8.append((i, j))

nb4 = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)


class Keypoint:
    def __init__(self, x, y, laplacian):
        self.x = x
        self.y = y
        self.max = laplacian > 0
        self.min = laplacian < 0
        self.laplacian = laplacian
        self.abs_laplacian = abs(laplacian)

    def pt(self):
        return self.x, self.y

    def __gt__(self, other):
        return self.laplacian > other.laplacian

    def __lt__(self, other):
        return self.laplacian < other.laplacian

    def __ge__(self, other):
        return self.laplacian >= other.laplacian

    def __le__(self, other):
        return self.laplacian <= other.laplacian

    def __repr__(self):
        return '(%d, %d)' % (self.x, self.y)

    def __str__(self):
        return '(%d, %d, %f)' % (self.x, self.y, self.laplacian)


class CNet:
    """
    This class represents a sequence of critical nets computed for a sequence of images.
    Each critical net is represented with a networkx.Graph.
    """
    def __init__(self, *, cache: bool = True, spectrum: str = 'normalized_laplacian', ss_sigma: int = 1.6,
                 logging=True):
        """
        :param cache:
            If True, then graphs and their spectra are cached.
        :param spectrum:
            One of: normalized_laplacian, laplacian, adjacency.
        """
        self._cache = cache
        self._spectrum = spectrum
        self._ss_sigma = ss_sigma
        self._logging = logging

        self._cache_prefix = '.ar-cache'
        self._cache_dir = None
        self._images_dir = None
        self._image_paths = None
        self._images_num = None
        self._figtitle = None
        self._dates = None
        self._flares = None
        self._stable_scales = None

        # The following properties are cached.

        # List of stable scales and betas for each image.
        self._scales_and_betas = None
        # Selected stable levels for each image.
        self._selected_levels = None
        # List of edges for each image. Edge is represented by Keypoint instance.
        self._edges = None
        # List of graphs for each image. Each graph is a critical net.
        self._graphs = None
        # List of spectrum for each image.
        self._spectra = None

    @property
    def imagepaths(self):
        return self._image_paths

    @property
    def images_num(self):
        return self._images_num

    @property
    def scales_and_betas(self):
        return self._scales_and_betas

    @property
    def selected_levels(self):
        return self._selected_levels

    @property
    def edges(self):
        return self._edges

    @property
    def graphs(self):
        return self._graphs

    @property
    def spectra(self):
        return self._spectra
    
    def _log(self, *args, **kwargs):
        if self._logging:
            print(*args, **kwargs)

    def _md5(self, string):
        m = md5()
        m.update(string.encode('utf-8'))
        return m.hexdigest()

    def proc(self, images_dir: str, *, scale: Union[int, None] = None):
        self._edges = []
        self._graphs = []
        self._scales_and_betas = []
        self._selected_levels = []

        self._images_dir = images_dir
        self._figtitle = images_dir.split('/')[-1]
        self._image_paths, self._images_num = utl.find_images(images_dir)
        self._dates = utl.parse_dates(self._image_paths)
        self._images_num = len(self._image_paths)
        self._flares = utl.parse_flares(path.join(images_dir, 'flares.txt'))

        self._log('\nFlares:')
        for date, flares in self._flares.items():
            for (flcl, _, _) in flares:
                self._log(date, flcl)
        self._log()

        # Which stable level we try to find.
        beta_level = 6

        # Checking cache first.
        md5hash = self._md5('%f%d' % (self._ss_sigma, scale if scale is not None else -1))
        self._cache_dir = path.join(
            self._cache_prefix,
            images_dir.replace('/', '_') + '__%s' % md5hash
        )

        if self._cache and not path.isdir(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)

        edges_fname = path.join(self._cache_dir, 'edges.pickle')
        graphs_fname = path.join(self._cache_dir, 'graphs.pickle')
        scales_and_betas_fname = path.join(self._cache_dir, 'scales_and_betas.pickle')
        selected_levels_fname = path.join(self._cache_dir, 'selected_levels.pickle')

        if not path.isfile(edges_fname) or not self._cache:
            self._log('Processing images')
            for i, filepath in enumerate(self._image_paths):
                self._log('%d/%d' % (i + 1, self._images_num), end=' ')

                if scale is None:
                    log, components, beta_scales, betas =\
                        cpt_stable_log(filepath, ss_sigma=self._ss_sigma)
                    self._scales_and_betas.append(list(zip(beta_scales, betas)))
                    idx = select_level(beta_level, betas)
                    self._selected_levels.append((beta_scales[idx], betas[idx]))
                    lap = log[idx]
                else:
                    lap = cpt_log(filepath, scale=scale, ss_sigma=self._ss_sigma)

                minima, maxima = cpt_keypoints(lap, replace_clusters=True)
                edges = cpt_criticalnet(lap, minima, maxima)
                self._edges.append(edges)

                graph = mk_graph(edges)
                self._graphs.append(graph)
            self._log('\n')

            if self._cache:
                with open(edges_fname, 'wb') as f:
                    pickle.dump(self._edges, f, pickle.HIGHEST_PROTOCOL)
                with open(graphs_fname, 'wb') as f:
                    pickle.dump(self._graphs, f, pickle.HIGHEST_PROTOCOL)
                with open(scales_and_betas_fname, 'wb') as f:
                    pickle.dump(self._scales_and_betas, f, pickle.HIGHEST_PROTOCOL)
                with open(selected_levels_fname, 'wb') as f:
                    pickle.dump(self._selected_levels, f, pickle.HIGHEST_PROTOCOL)
        else:
            self._log('Loading edges from cache')
            with open(edges_fname, 'rb') as f:
                self._edges = pickle.load(f)
            self._log('Loading graphs from cache')
            with open(graphs_fname, 'rb') as f:
                self._graphs = pickle.load(f)
            self._log('Loading stable levels from cache')
            with open(scales_and_betas_fname, 'rb') as f:
                self._scales_and_betas = pickle.load(f)
            self._log('Loading selected levels from cache')
            with open(selected_levels_fname, 'rb') as f:
                self._selected_levels = pickle.load(f)

        self.cpt_spectra(self._spectrum)

    def cpt_spectra(self, spectrum_tp):
        self._spectra = []
        spectra_fname = path.join(self._cache_dir, 'spectra_%s.pickle' % spectrum_tp)
        if self._cache and path.isfile(spectra_fname):
            self._log('Loading spectra from cache')
            with open(spectra_fname, 'rb') as f:
                self._spectra = pickle.load(f)
        else:
            self._log('Computing spectra')
            for graph in self._graphs:
                if spectrum_tp == 'normalized_laplacian':
                    spectrum = cpt_normalized_laplacian_spectrum(graph)
                elif spectrum_tp == 'laplacian':
                    spectrum = laplacian_spectrum(graph)
                elif spectrum_tp == 'adjacency':
                    spectrum = adjacency_spectrum(graph)
                else:
                    spectrum = None
                self._spectra.append(spectrum)
            if self._cache:
                with open(spectra_fname, 'wb') as f:
                    pickle.dump(self._spectra, f, pickle.HIGHEST_PROTOCOL)

    def describe_spectra(self):
        eps = 1e-10
        min_vals = []
        max_vals = []
        for sp in self._spectra:
            sp = sp[np.abs(sp) >= eps]
            min_vals.append(sp.min())
            max_vals.append(sp.max())
        print('Min spectra value varies from %f to %f' % (min(min_vals), max(min_vals)))
        print('Max spectra value varies from %f to %f' % (min(max_vals), max(max_vals)))

    def stable_scales(self, beta):
        if self._stable_scales is None and self._scales_and_betas:
            stable_scales = {}
            for _, pairs in enumerate(self._scales_and_betas):
                for s, b in pairs:
                    if b not in stable_scales:
                        stable_scales[b] = []
                    stable_scales[b].append(s)
            self._stable_scales = stable_scales
        return self._stable_scales[beta] if beta in self._stable_scales else None

    def density_nodes_corrcoef(self):
        nn_vals = []
        density_vals = []
        for graph in self._graphs:
            nn_vals.append(graph.number_of_nodes())
            density_vals.append(density(graph))
        print(np.corrcoef(nn_vals, density_vals)[0, 1])

    def plot_stable_scales_hist(self):
        plt.figure()
        plt.subplot()
        plt.xlabel('Scale', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.hist(np.array([x for x, _ in self._selected_levels]), 20)
        plt.show()

    def plot_stable_levels_hist(self):
        plt.figure()
        plt.subplot()
        plt.xlabel(r'$\beta$', fontsize=18)
        plt.ylabel('Count', fontsize=14)
        plt.hist(np.array([x for _, x in self._selected_levels]), 10)
        plt.show()

    def plot_spectral_max(self, log_scale: bool = False, log_base: float = 2):
        vals = []
        for gsp in self._spectra:
            val = gsp.max()
            if log_scale:
                val = math.log(val, log_base)
            vals.append(val)

        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        ax.set_xlabel('Image index', fontsize=14)
        ax.set_ylabel('Absolute maximum spectra value', fontsize=12)
        vals_abs = np.abs(vals)
        ax.plot(vals_abs, 'b-')
        ax.set_ylim(0, max(vals_abs) + 1)
        ax.set_xlim(0, len(vals) + 1)

        plot_dates(ax, self._dates, self._flares)
        plt.show()

    def plot_spectral_gap(self, i: int = 0, j: int = 1, *, remove_zeroes: bool = False,
                          log_scale: bool = False, log_base: float = 2, **kwargs):
        eps = 1e-10
        gaps = []
        for gsp in self._spectra:
            vals = [x for x in sorted(np.abs(gsp)) if x > eps]
            # Turn very small values into zeroes.
            if not remove_zeroes:
                zeroes_num = len(gsp) - len(vals)
                vals = [0] * zeroes_num + vals
            val1 = vals[i]
            val2 = val1
            # Considering multiplicity.
            jj = j
            while abs(val1 - val2) <= eps and abs(jj) != len(vals) - 1:
                val2 = vals[jj]
                jj += 1 if j > 0 else -1
            gap = abs(val1 - val2)
            if log_scale:
                gap = math.log(gap, log_base)
            gaps.append(gap)

        gaps = utl.apply_smoothing(gaps, **kwargs)

        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        ymin = min(gaps)
        ymin = 0 if ymin >= 0 else ymin
        ax.set_ylim(ymin, max(gaps))
        ax.set_xlim(-1, len(gaps) + 1)
        ax.set_xlabel('Image index', fontsize=14)
        ax.set_ylabel('Spectral gap', fontsize=14)
        ax.plot(gaps, 'b-')

        plot_dates(ax, self._dates, self._flares)
        plt.show()

    def plot_density(self, log_scale: bool = False, log_base: float = 2):
        vals = []
        for graph in self._graphs:
            val = density(graph)
            if log_scale:
                val = math.log(val, log_base)
            vals.append(val)

        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        ax.set_xlim(0, len(self._graphs))
        ax.set_ylim(0, max(vals))
        ax.set_xlabel('Image index', fontsize=14)
        ax.set_ylabel('Graph density', fontsize=14)
        ax.plot(vals)

        plot_dates(ax, self._dates, self._flares)
        plt.show()

    def plot_number_of_nodes(self, log_scale: bool = False, log_base: float = 2):
        vals = []
        for graph in self._graphs:
            val = graph.number_of_nodes()
            if log_scale:
                val = math.log(val, log_base)
            vals.append(val)

        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        ax.set_ylim(0, max(vals))
        ax.set_xlim(0, len(self._graphs))
        ax.set_xlabel('Image index', fontsize=14)
        ax.set_ylabel('Number of nodes', fontsize=14)
        ax.plot(vals)

        plot_dates(ax, self._dates, self._flares)
        plt.show()

    def plot_algebraic_connectivity(self, log_scale: bool = False, log_base: float = 2) -> Axes:
        eps = 1e-10
        vals = []
        for gsp in self._spectra:
            # Checking multiplicity.
            agsp = [x for x in sorted(np.abs(gsp)) if x > eps]
            # We search for the next eigenvalue different from the first eigenvalue.
            val = agsp[0]
            for i in range(1, len(agsp)):
                if abs(val - agsp[i]) > eps:
                    val = agsp[i]
                    break
            if log_scale:
                val = math.log(val, log_base)
            vals.append(val)

        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        ax.set_xlabel('Image index', fontsize=14)
        ax.set_ylabel('Algebraic connectivity', fontsize=14)
        ax.plot(vals, 'b-')
        ax.set_ylim(0, max(vals))
        ax.set_xlim(0, len(vals))

        plot_dates(ax, self._dates, self._flares)
        plt.show()


def is_regional_extremum(component: List[Tuple[int, int]], img: np.ndarray, visited: np.ndarray,
                         is_max: bool) -> Union[bool, int]:
    """
    Finds connected component with constant intensity and checks if it is extremum.
    """
    ly, lx = img.shape
    stack = component[1:]
    y, x = component[0]
    value = img[y, x]
    try:
        while True:
            y, x = stack.pop()
            for dy, dx in nb8:
                yy = y + dy
                xx = x + dx
                if 0 <= xx < lx and 0 <= yy < ly and not visited[yy, xx]:
                    l = img[yy, xx]
                    if l != 0:
                        if l == value:
                            visited[yy, xx] = True
                            stack.append((yy, xx))
                            component.append((yy, xx))
                        elif is_max and value < l or not is_max and value > l:
                            return False
    except IndexError:
        return 1 if is_max is True else -1


def search_extrema(img: np.ndarray) -> List[Keypoint]:
    keypoints = []
    ly, lx = img.shape
    visited = np.zeros(img.shape, dtype=np.int8)

    for y in range(ly):
        for x in range(lx):
            z = img[y, x]
            if z != 0 and not visited[y, x]:
                visited[y, x] = True
                prev = None
                is_extremum = True
                component = [(y, x)]

                # Checking neighbourhood of a point.
                for dy, dx in nb8:
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < lx and 0 <= yy < ly:
                        zz = img[yy, xx]
                        if zz:
                            if z > zz:
                                # 1 means maximum.
                                cur = 1
                            elif z < zz:
                                # -1 means minimum.
                                cur = -1
                            else:
                                cur = 0
                                component.append((yy, xx))

                    if prev is not None and prev != cur and cur:
                        is_extremum = False
                        break

                    prev = cur

                # Neighbourhood contained intensities equal to value that means this is a connected
                # components with constant intensity.
                if is_extremum and len(component) > 1:
                    # print('checking component')
                    # We need to find the whole component and check if it is extremum.
                    cur = is_regional_extremum(component, img, visited, prev == 1)
                    # As coordinates we choose a center of mass of the connected component.
                    if cur is not False and prev == cur:
                        y = round(sum([yy for yy, xx in component]) / len(component))
                        x = round(sum([xx for yy, xx in component]) / len(component))
                        # print('found component')
                    else:
                        is_extremum = False

                if is_extremum:
                    keypoints.append(Keypoint(x, y, z))
    return keypoints


def remove_weak_keypoints(kpts: List[Keypoint], threshold: float) -> List[Keypoint]:
    """
    Removes points whose absolute laplacian value is more than 90% different from maximal
    absolute laplacian value.

    :param kpts:
    :param threshold:
    :return:
    """
    kpts_ = []
    max_abs_lap = max(kpts, key=lambda z: z.abs_laplacian).abs_laplacian
    for kp in kpts:
        ratio = abs(kp.laplacian) / max_abs_lap
        if ratio > threshold:
            kpts_.append(kp)
    return kpts_


def remove_border_keypoints(kpts: List[Keypoint], lx: int, ly: int) -> List[Keypoint]:
    """
    Removes points lying on the border.

    :param kpts:
    :param lx:
    :param ly:
    :return:
    """
    kpts_ = []
    for kp in kpts:
        if not (kp.x == 0 or kp.x == lx - 1 or kp.y == 0 or kp.y == ly - 1):
            kpts_.append(kp)
    return kpts_


def replace_clusters_with_centroids(kpts: List[Keypoint]) -> List[Keypoint]:
    new_kpts = []
    clf = DBSCAN(10, 2)
    clf.fit([[kp.x, kp.y] for kp in kpts])
    kpts_np = np.asarray(kpts)
    labels = clf.labels_
    unique_labels = set(labels) - {-1}

    for label in unique_labels:
        cluster = kpts_np[labels == label]
        centroid = Keypoint(
            int(round(np.mean([kp.x for kp in cluster]))),
            int(round(np.mean([kp.y for kp in cluster]))),
            np.mean([kp.laplacian for kp in cluster])
        )
        new_kpts.append(centroid)

    for pt in kpts_np[labels == -1]:
        new_kpts.append(pt)

    return new_kpts


def select_level(beta: int, betas: List[int]) -> int:
    closest = 0
    idx = None
    for i, b in enumerate(betas):
        if beta - b == 0:
            return i
        elif closest < b < beta:
            closest = b
            idx = i
    return idx


def cpt_log(filepath: str, scale: int, ss_sigma: float = 1.67) -> List[np.ndarray]:
    img = rgb2gray(img_as_float(io.imread(filepath)))
    # Laplacian of gaussians.
    return gaussian(img, sigma=math.sqrt(scale + 1) * ss_sigma, mode='reflect') - \
        gaussian(img, sigma=math.sqrt(scale) * ss_sigma, mode='reflect')


def cpt_stable_log(filepath: str, beta: float = None, ss_size: int = 100, ss_sigma: float = 1.67):
    nb8_elem = [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]

    img = rgb2gray(img_as_float(io.imread(filepath)))
    prev_img = gaussian(img, sigma=ss_sigma, mode='reflect')

    log = []
    components = []

    for i in range(ss_size):
        # Computing laplacian of gaussians.
        next_img = gaussian(prev_img, sigma=ss_sigma, mode='reflect')
        lap = next_img - prev_img
        prev_img = next_img

        # Computing number of connected components in each binarized laplacian (maximally convex
        # areas).
        lap_bin = np.zeros(img.shape, dtype=np.int8)
        lap_bin[lap > 0] = 1
        _, components_num = label(lap_bin, nb8_elem)
        components.append(components_num)

        # Searching for all stable levels.
        if beta is None:
            log.append(lap)
        # Attempt to find specified beta-stable level.
        elif i+1 > beta and len(np.unique(components[i - beta: i])) == 1:
            return lap, i, beta

    # Could not find stable level, otherwise work would be finished earlier.
    if beta is not None:
        return False

    # Computing betas.
    beta = 0
    betas = [0] * len(components)
    for idx, c in enumerate(components):
        if idx:
            if prev_img == c:
                beta += 1
                betas[idx] = beta
            else:
                beta = 0
        prev_img = c

    beta_scales = []
    betas_ = []
    l = len(betas)
    for i in range(l - 1):
        if i == 0 and betas[i] > betas[i + 1] or betas[i - 1] < betas[i] > betas[i + 1]:
            beta_scales.append(i)
            betas_.append(betas[i] + 1)
    if betas[l - 2] < betas[l - 1]:
        beta_scales.append(l - 1)
        betas_.append(betas[l - 1] + 1)
    betas = betas_

    return log, components, beta_scales, betas


def cpt_keypoints(lap: np.ndarray, threshold: float = 0.1, replace_clusters: bool = True)\
      -> Tuple[List[Keypoint], List[Keypoint]]:
    ly, lx = lap.shape
    keypoints = search_extrema(lap)
    minima = sorted([k for k in keypoints if k.min])
    maxima = sorted([k for k in keypoints if k.max])
    minima = remove_weak_keypoints(minima, threshold)
    maxima = remove_weak_keypoints(maxima, threshold)
    minima = remove_border_keypoints(minima, lx, ly)
    maxima = remove_border_keypoints(maxima, lx, ly)

    if replace_clusters:
        minima = replace_clusters_with_centroids(minima)
        maxima = replace_clusters_with_centroids(maxima)

    return minima, maxima


Edges = List[Tuple[Keypoint, Keypoint]]


def cpt_criticalnet(lap: np.ndarray, minima: List[Keypoint], maxima: List[Keypoint]) -> Edges:
    max_map = {}
    for m in maxima:
        if m.y not in max_map:
            max_map[m.y] = {}
        max_map[m.y][m.x] = m

    edges = []
    ly, lx = lap.shape

    for m in minima:
        visited = np.zeros(lap.shape, dtype=bool)
        visited[m.y, m.x] = True
        queue = [(m.x, m.y)]
        queue_sz = 1
        idx = 0
        while idx < queue_sz:
            x, y = queue[idx]
            idx += 1

            if x == 0 or x == lx - 1 or y == 0 or y == ly - 1:
                continue

            if y in max_map and x in max_map[y]:
                edges.append((
                    m,
                    max_map[y][x]
                ))
                continue

            for dy, dx in nb4:
                xx = x + dx
                yy = y + dy
                if 0 <= xx < lx and 0 <= yy < ly:
                    z = lap[y, x]
                    if not visited[yy, xx] and lap[yy, xx] >= z:
                        visited[yy, xx] = True
                        queue.append((xx, yy))
                        queue_sz += 1

    return edges


def cpt_adjacency_matrix(edges: Edges) -> np.ndarray:
    gr_minima = sorted(list({m for m, M in edges}))
    gr_maxima = sorted(list({M for m, M in edges}))
    min_len = len(gr_minima)
    max_len = len(gr_maxima)
    minima_idx = {}
    maxima_idx = {}

    for i, m in enumerate(gr_minima):
        minima_idx[m] = i
    for i, M in enumerate(gr_maxima):
        maxima_idx[M] = min_len + i

    vert_num = min_len + max_len
    adj_mat = np.zeros((vert_num, vert_num), dtype=np.int32)

    for m, M in edges:
        idx_min = minima_idx[m]
        idx_max = maxima_idx[M]
        adj_mat[idx_min, idx_max] = 1
        adj_mat[idx_max, idx_min] = 1

    return adj_mat


def cpt_laplacian_spectrum(edges: Edges) -> Union[np.ndarray, bool]:
    adj_mat = cpt_adjacency_matrix(edges)
    adj_mat = -adj_mat
    for i, s in enumerate(adj_mat.sum(axis=0)):
        adj_mat[i, i] = -s

    try:
        e, _ = np.linalg.eig(adj_mat)
        return e
    except np.linalg.LinAlgError:
        return False


def cpt_normalized_laplacian_spectrum(graph: nx.Graph) -> Union[np.ndarray, bool]:
    from scipy.linalg import eigvalsh
    return eigvalsh(normalized_laplacian_matrix(graph).todense())


def mk_graph(edges: Edges) -> nx.Graph:
    """
    Returns networkx.Graph object constructed from edges computed by cpt_criticalnet() function.

    :param edges:
    """

    gr_minima = sorted(list({m for m, M in edges}))
    gr_maxima = sorted(list({M for m, M in edges}))
    min_len = len(gr_minima)
    minima_idx = {}
    maxima_idx = {}

    for i, m in enumerate(gr_minima):
        minima_idx[m] = i
    for i, M in enumerate(gr_maxima):
        maxima_idx[M] = min_len + i

    graph = nx.Graph()
    for m, M in edges:
        idx_min = minima_idx[m]
        idx_max = maxima_idx[M]
        graph.add_edge(idx_min, idx_max)

    return graph


def plot_keypoints(img: np.ndarray, minima: List[Keypoint], maxima: List[Keypoint]) -> Axes:
    ly, lx = img.shape[:2]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_xlim(0, lx - 1)
    ax.set_ylim(ly - 1, 0)
    ax.plot([kp.x for kp in maxima], [kp.y for kp in maxima], 'ro')
    ax.plot([kp.x for kp in minima], [kp.y for kp in minima], 'bo')
    return ax


def plot_criticalnet(img: np.ndarray, edges: Edges, newfigure: bool = True) -> Axes:
    min_x, min_y = [], []
    max_x, max_y = [], []

    ly, lx = img.shape[:2]
    if newfigure:
        fig = plt.figure(figsize=(12, 12))
    else:
        fig = plt.gcf()
        fig.clf()

    ax = fig.add_subplot(111)
    ax.xaxis.tick_top()
    ax.imshow(img, cmap='gray')
    ax.set_xlim(0, lx - 1)
    ax.set_ylim(ly - 1, 0)

    for e in edges:
        fr, to = e
        min_x.append(fr.x)
        max_x.append(to.x)
        min_y.append(fr.y)
        max_y.append(to.y)
        ax.plot([fr.x, to.x], [fr.y, to.y], 'b-')

    ax.plot(max_x, max_y, 'ro')
    ax.plot(min_x, min_y, 'bo')

    return ax


def plot_components(components: List[int]) -> Axes:
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.set_xlabel('Scale', fontsize=12)
    ax.set_ylabel('Components', fontsize=12)
    ax.plot(components)

    return ax


def plot_betas(beta_scales: List[int], betas: List[int]) -> Axes:
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.set_xlabel('$k$', fontsize=14)
    ax.set_ylabel('$\\beta$', fontsize=14)
    ax.set_xlim(0, max(beta_scales) + 1)
    ax.set_ylim(0, max(betas) + 1)
    ax.plot(beta_scales, betas, 'b')
    ax.plot(beta_scales, betas, 'ro')
    return ax


# Maps date to list of corresponding image indexes. Mostly used for drawing vertical lines.
Dates = Dict[str, List[int]]
Flares = Dict[str, Tuple[str, int, int]]


def plot_dates(ax: Axes, dates: Dates, flares: Flares, color: str = 'g'):
    ticks = []
    ticklabels = []

    y = ax.get_ylim()[1] * 0.95
    for date, idxs in dates.items():
        x = max(idxs)
        ticks.append(max(idxs))
        ticklabels.append(date)
        ax.axvline(x, color=color, linestyle='solid')
        if date in flares:
            for (flcl, idx, _) in flares[date]:
                flpos = min(idxs) + idx - 1
                flcolor = 'r' if flcl[0] in ['M', 'X'] else '#BD761A'
                ax.axvline(flpos, color=flcolor, linestyle='dashed')
                ax.text(flpos - 7, y, flcl, color=flcolor)

    twin = ax.twiny()
    twin.set_xlim(ax.get_xlim())
    twin.set_xticks(ticks)
    twin.set_xticklabels(ticklabels, rotation=45, ha='left', fontsize=10)
