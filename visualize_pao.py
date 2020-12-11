from algorithms import *
from visualization import *


if __name__ == '__main__':
    dataset_name = 'pima'
    gamma = 0.1

    X, y = prepare_data(dataset_name, scaler='MinMax', n_minority_samples=None)
    minority_class = Counter(y).most_common()[1][0]
    pao = PAO(gamma=gamma, lambd=0.0, random_state=42)
    pao.sample(X, y)

    visualize(
        X, y, gamma=gamma,
        potential_type='minority',
        file_name=f'example_pao_{dataset_name}_original',
        minority_class=minority_class,
        lim=(-0.05, 1.05)
    )

    visualize(
        X, y, gamma=gamma,
        appended=pao._prototypes,
        anchors=pao._anchors,
        potential_type='appended',
        file_name=f'example_pao_{dataset_name}_resampled',
        minority_class=minority_class,
        lim=(-0.05, 1.05)
    )
