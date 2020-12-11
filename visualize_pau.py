from algorithms import *
from visualization import *


if __name__ == '__main__':
    dataset_name = 'pima'
    gamma = 0.1

    X, y = prepare_data(dataset_name, scaler='MinMax', n_minority_samples=None)
    minority_class = Counter(y).most_common()[1][0]
    pau = PAU(gamma=gamma, random_state=42)
    X_, y_ = pau.sample(X, y)

    visualize(
        X, y, gamma=gamma,
        potential_type='majority',
        file_name=f'example_pau_{dataset_name}_original',
        minority_class=minority_class,
        lim=(-0.05, 1.05)
    )

    visualize(
        X_, y_, gamma=gamma,
        replaced=pau._prototypes,
        anchors=pau._anchors,
        potential_type='majority',
        file_name=f'example_pau_{dataset_name}_resampled',
        minority_class=minority_class,
        lim=(-0.05, 1.05)
    )
