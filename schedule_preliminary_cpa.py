import os


if __name__ == '__main__':
    for fold in range(10):
        for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            command = f'sbatch run.sh run_preliminary_cpa.py -fold {fold} -ratio {ratio}'

            os.system(command)
