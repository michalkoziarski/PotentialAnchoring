import os


if __name__ == '__main__':
    for fold in range(10):
        for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
            command = f'sbatch run.sh run_gamma.py -fold {fold} -gamma {gamma}'

            os.system(command)
