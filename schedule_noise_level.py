import os


RESAMPLERS = [
    'SMOTE', 'polynom-fit-SMOTE', 'Lee', 'SMOBD',
    'G-SMOTE', 'LVQ-SMOTE', 'Assembled-SMOTE',
    'SMOTE-TomekLinks', 'RBO', 'PA'
]


if __name__ == '__main__':
    for fold in range(10):
        for level in [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]:
            for resampler_name in RESAMPLERS:
                command = f'sbatch run.sh run_noise_level.py -fold {fold} -resampler_name {resampler_name} -level {level}'

                os.system(command)
