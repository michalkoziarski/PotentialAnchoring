from algorithms import *
from visualization import *


if __name__ == '__main__':
    dataset_name = 'yeast3'
    gamma = 0.5

    X, y = prepare_data(dataset_name, scaler='Standard', n_minority_samples=None)
    minority_class = Counter(y).most_common()[1][0]

    for lambd in [0.0, 0.001, 10.0]:
        pao = PAO(gamma=gamma, lambd=lambd, random_state=42)
        pao.sample(X, y)

        if lambd == 0.0:
            visualize(
                X, y, gamma=gamma,
                anchors=pao._anchors,
                potential_type='minority',
                file_name=f'example_pao_{dataset_name}_original',
                minority_class=minority_class
            )

        visualize(
            X, y, gamma=gamma,
            appended=pao._prototypes,
            anchors=pao._anchors,
            potential_type='appended',
            file_name=f'example_pao_{dataset_name}_resampled_lambda_{lambd}',
            minority_class=minority_class
        )
