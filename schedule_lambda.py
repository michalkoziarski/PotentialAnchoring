import os


if __name__ == '__main__':
    for fold in range(10):
        for lambd in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]:
            command = f'sbatch run.sh run_lambda.py -fold {fold} -lambd {lambd}'

            os.system(command)
