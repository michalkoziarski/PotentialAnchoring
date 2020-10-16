import os


RESAMPLERS = [
    'SMOTE', 'polynom-fit-SMOTE', 'ProWSyn',
    'SMOTE-IPF', 'Lee', 'SMOBD', 'G-SMOTE',
    'CCR', 'LVQ-SMOTE', 'Assembled-SMOTE',
    'SMOTE-TomekLinks', 'PAO', 'PAU',
    'CPA.1', 'CPA.9'
]


if __name__ == '__main__':
    for fold in range(10):
        for resampler_name in RESAMPLERS:
            command = f'sbatch run.sh run_final.py -fold {fold} -resampler_name {resampler_name}'

            os.system(command)
