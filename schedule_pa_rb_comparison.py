import os


RESAMPLERS = [
    'RBO', 'RBU', 'PAO', 'PAU'
]


if __name__ == '__main__':
    for fold in range(10):
        for resampler_name in RESAMPLERS:
            command = f'sbatch run.sh run_pa_rb_comparison.py -fold {fold} -resampler_name {resampler_name}'

            os.system(command)
