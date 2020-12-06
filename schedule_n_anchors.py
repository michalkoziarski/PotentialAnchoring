import os


if __name__ == '__main__':
    for fold in range(10):
        for n_anchors in [3, 6, 9, 12, 15, 18, 21]:
            command = f'sbatch run.sh run_n_anchors.py -fold {fold} -n_anchors {n_anchors}'

            os.system(command)
