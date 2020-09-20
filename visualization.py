import datasets
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler


VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'

ALPHA = 0.9
BACKGROUND_COLOR = '#EEEEEE'
BORDER_COLOR = '#161921'
COLOR_ANCHOR = '#F2E85C'
COLOR_MAJORITY = '#C44E52'
COLOR_MINORITY = '#4C72B0'
COLOR_NEUTRAL = '#F2F2F2'
FIGURE_SIZE = (6, 6)
LINE_WIDTH = 1.0
MARGIN = 0.05
MARKER_SIZE = 75
MARKER_SYMBOL = 'o'
ORIGINAL_EDGE_COLOR = '#F2F2F2'
OVERSAMPLED_EDGE_COLOR = '#262223'
POTENTIAL_GRID_N = 150


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def rbf_score(x, points, gamma, p_norm=2):
    result = 0.0

    for point in points:
        result += rbf(distance(x, point, p_norm), gamma)

    return result


def visualize(X, y, appended=None, anchors=None, gamma=None,
              potential_type='majority', file_name=None, lim=None):
    assert len(np.unique(y)) == 2
    assert X.shape[1] == 2
    assert potential_type in ['majority', 'minority']

    if appended is not None:
        assert appended.shape[1] == 2

    plt.style.use('ggplot')

    classes = np.unique(y)
    sizes = [sum(y == c) for c in classes]
    minority_class = classes[np.argmin(sizes)]

    minority_points = X[y == minority_class].copy()
    majority_points = X[y != minority_class].copy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.set_xticks([])
    ax.set_yticks([])

    for key in ax.spines.keys():
        ax.spines[key].set_color(BORDER_COLOR)

    ax.tick_params(colors=BORDER_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    x_limits = [np.min(X[:, 0]), np.max(X[:, 0])]
    y_limits = [np.min(X[:, 1]), np.max(X[:, 1])]

    x_spread = np.abs(x_limits[1] - x_limits[0])
    y_spread = np.abs(y_limits[1] - y_limits[0])

    x_limits[0] = x_limits[0] - MARGIN * x_spread
    y_limits[0] = y_limits[0] - MARGIN * y_spread
    x_limits[1] = x_limits[1] + MARGIN * x_spread
    y_limits[1] = y_limits[1] + MARGIN * y_spread

    if lim is None:
        plt.xlim(x_limits)
        plt.ylim(y_limits)
    else:
        plt.xlim(lim)
        plt.ylim(lim)

    plt.scatter(
        majority_points[:, 0], majority_points[:, 1],
        s=MARKER_SIZE, c=COLOR_MAJORITY, linewidths=LINE_WIDTH,
        alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=ORIGINAL_EDGE_COLOR
    )

    plt.scatter(
        minority_points[:, 0], minority_points[:, 1],
        s=MARKER_SIZE, c=COLOR_MINORITY, linewidths=LINE_WIDTH,
        alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=ORIGINAL_EDGE_COLOR
    )

    if appended is not None:
        plt.scatter(
            appended[:, 0], appended[:, 1],
            s=MARKER_SIZE, c=COLOR_MINORITY, linewidths=LINE_WIDTH,
            alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=OVERSAMPLED_EDGE_COLOR
        )

    if anchors is not None:
        plt.scatter(
            anchors[:, 0], anchors[:, 1],
            s=MARKER_SIZE, c=COLOR_ANCHOR, linewidths=LINE_WIDTH,
            alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=OVERSAMPLED_EDGE_COLOR
        )

    if gamma is not None:
        x_cont = np.linspace(x_limits[0], x_limits[1], POTENTIAL_GRID_N + 1)
        y_cont = np.linspace(y_limits[0], y_limits[1], POTENTIAL_GRID_N + 1)

        X_cont, Y_cont = np.meshgrid(x_cont, y_cont)

        Z = np.zeros(X_cont.shape)

        for i, x1 in enumerate(x_cont):
            for j, x2 in enumerate(y_cont):
                Z[j][i] = rbf_score(
                    np.array([x1, x2]),
                    majority_points if potential_type == 'majority' else minority_points,
                    gamma
                )

        plt.contour(X_cont, Y_cont, Z)

    if file_name is not None:
        VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

        plt.savefig(VISUALIZATIONS_PATH / f'{file_name}.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def prepare_data(dataset_name, n_minority_samples=20, scaler='MinMax'):
    dataset = datasets.load(dataset_name)
    (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])

    minority_class = Counter(y).most_common()[1][0]
    majority_class = Counter(y).most_common()[0][0]

    n_minority = Counter(y).most_common()[1][1]
    n_majority = Counter(y).most_common()[0][1]

    X, y = RandomUnderSampler(
        sampling_strategy={
            minority_class: np.min([n_minority, n_minority_samples]),
            majority_class: n_majority
        },
        random_state=42,
    ).fit_sample(X, y)

    X = TSNE(n_components=2, random_state=42).fit_transform(X)

    if scaler == 'MinMax':
        X = MinMaxScaler().fit_transform(X)
    elif scaler == 'Standard':
        X = StandardScaler().fit_transform(X)
    else:
        raise NotImplementedError

    return X, y
