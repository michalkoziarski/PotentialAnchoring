import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['KNN', 'CART', 'L-SVM', 'R-SVM', 'P-SVM', 'LR', 'R-MLP']:
            command = f'sbatch run.sh run_preliminary_cpa.py -fold {fold} -classifier_name {classifier_name}'

            os.system(command)
