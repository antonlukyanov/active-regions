from skimage import io
from skimage import img_as_float
from skimage.filters import gaussian
from skimage.color import rgb2gray
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import numpy as np


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
        return '(%d, %d, %f)' % (self.x, self.y, self.laplacian)


def is_regional_extremum(component, img, visited, is_max):
    """
    Довыделяет компоненту связности с постоянной яркостью и проверяет является ли она экстремумом.
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


def search_extrema(img):
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

                # Проверяем окрестность точки.
                for dy, dx in nb8:
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < lx and 0 <= yy < ly:
                        zz = img[yy, xx]
                        if zz:
                            if z > zz:
                                # 1 значит максимум.
                                cur = 1
                            elif z < zz:
                                # -1 значит минимум.
                                cur = -1
                            else:
                                cur = 0
                                component.append((yy, xx))

                    if prev is not None and prev != cur and cur:
                        is_extremum = False
                        break

                    prev = cur

                # В окрестности были значения равные value, значит это компонента
                # связности с постоянной яркостью.
                if is_extremum and len(component) > 1:
                    print('checking component')
                    # Необходимо найти всю компоненту связности с постоянной яркостью
                    # и проверить является ли она экстремумом.
                    cur = is_regional_extremum(component, img, visited, prev == 1)
                    # В качестве координат особой точки выбираем центр масс компоненты связности.
                    if cur is not False and prev == cur:
                        y = round(sum([yy for yy, xx in component]) / len(component))
                        x = round(sum([xx for yy, xx in component]) / len(component))
                        print('found component')
                    else:
                        is_extremum = False

                if is_extremum:
                    keypoints.append(Keypoint(x, y, z))
    return keypoints


def remove_weak_keypoints(kpts, threshold):
    """
    Выкидывает точки, у которых абсолютное значение лапласиана более чем на 90% отличается от
    максимального абсолютного значения.

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


def remove_border_keypoints(kpts, lx, ly):
    """
    Выкидывает точки, которые лежат на границе.

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


def replace_clusters_with_centroids(kpts):
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


def cpt_stable_log(filepath, beta=None, ss_size=100, ss_sigma=1.67):
    nb8_elem = [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]

    img = rgb2gray(img_as_float(io.imread(filepath)))
    prev_img = gaussian(img, sigma=ss_sigma, mode='reflect')

    ss_log = []
    ss_components = []

    for i in range(ss_size):
        # Считаем лапласиан гауссиан.
        next_img = gaussian(prev_img, sigma=ss_sigma, mode='reflect')
        lap = next_img - prev_img
        prev_img = next_img

        # Считаем количество связных компонент в каждом бинаризованном лапласиане (максимально
        # выпуклые области).
        lap_bin = np.zeros(img.shape, dtype=np.int8)
        lap_bin[lap > 0] = 1
        _, components_num = label(lap_bin, nb8_elem)
        ss_components.append(components_num)

        # Поиск всех устойчивых масштабов.
        if beta is None:
            ss_log.append(lap)
        # Попытка поиска beta-устойчивого масштаба.
        elif i+1 > beta and len(np.unique(ss_components[i - beta: i])) == 1:
            return lap, i, beta

    # Устойчивый масштаб не найден, иначе бы работа завершилась раньше.
    if beta is not None:
        return False

    # Считаем значения beta.
    beta = 0
    betas = [0] * len(ss_components)
    for idx, c in enumerate(ss_components):
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

    return ss_log, ss_components, beta_scales, betas


def cpt_keypoints(lap, threshold=0.1, replace_clusters=True):
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


def cpt_criticalnet(lap, minima, maxima):
    # Todo: наверное лучше заменить на словарь, а то слишком жирно.
    max_map = np.zeros(lap.shape, dtype=bool)
    for m in maxima:
        max_map[m.y, m.x] = True

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

            if max_map[y, x]:
                edges.append((
                    (m.x, m.y),
                    (x, y)
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


def plot_keypoints(img, minima, maxima):
    ly, lx = img.shape[:2]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_xlim(0, lx - 1)
    ax.set_ylim(ly - 1, 0)
    ax.plot([kp.x for kp in maxima], [kp.y for kp in maxima], 'ro')
    ax.plot([kp.x for kp in minima], [kp.y for kp in minima], 'bo')
    return ax


def plot_criticalnet(img, edges):
    min_x, min_y = [], []
    max_x, max_y = [], []

    ly, lx = img.shape[:2]
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.xaxis.tick_top()
    ax.imshow(img, cmap='gray')
    plt.xlim(0, lx - 1)
    plt.ylim(ly - 1, 0)

    for e in edges:
        fr, to = e
        x1, y1 = fr
        x2, y2 = to
        min_x.append(x1)
        max_x.append(x2)
        min_y.append(y1)
        max_y.append(y2)
        ax.plot([x1, x2], [y1, y2], 'b-')

    ax.plot(max_x, max_y, 'ro')
    ax.plot(min_x, min_y, 'bo')

    return ax


def plot_components(components):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.set_xlabel('Scale', fontsize=12)
    ax.set_ylabel('Components', fontsize=12)
    ax.plot(components)

    return ax


def plot_betas(beta_scales, betas):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.set_xlabel('$k$', fontsize=14)
    ax.set_ylabel('$\\beta$', fontsize=14)
    ax.set_xlim(0, max(beta_scales) + 1)
    ax.set_ylim(0, max(betas) + 1)
    ax.plot(beta_scales, betas, 'b')
    ax.plot(beta_scales, betas, 'ro')
    return ax
