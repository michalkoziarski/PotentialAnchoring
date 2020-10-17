import os


if __name__ == '__main__':
    for fold in range(10):
        for gamma in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            command = f'sbatch run.sh run_preliminary_gamma.py -fold {fold} -gamma {gamma}'

            os.system(command)
