import os


RESAMPLERS = [
    'SMOTE', 'polynom-fit-SMOTE', 'Lee', 'SMOBD',
    'G-SMOTE', 'LVQ-SMOTE', 'Assembled-SMOTE',
    'SMOTE-TomekLinks', 'RBO', 'PA'
]


if __name__ == '__main__':
    for fold in range(10):
        for resampler_name in RESAMPLERS:
            command = f'sbatch run.sh run_final.py -fold {fold} -resampler_name {resampler_name}'

            os.system(command)
