from algorithms import *
from visualization import *


if __name__ == '__main__':
    for dataset_name in ['yeast1', 'vehicle3', 'abalone-17_vs_7-8-9-10', 'page-blocks0']:
        X, y = prepare_data(dataset_name, scaler='Standard', n_minority_samples=None)
        minority_class = Counter(y).most_common()[1][0]
        pa = PA(random_state=42)
        pa.sample(X, y)

        visualize(
            X, y,
            file_name=f'example_pa_{dataset_name}_original',
            minority_class=minority_class
        )

        visualize(
            X, y,
            appended=pa.pao._prototypes,
            replaced=pa.pau._prototypes,
            file_name=f'example_pa_{dataset_name}_resampled',
            minority_class=minority_class
        )
