import argparse
import os
import subprocess

from itertools import product
import numpy as np

parser = argparse.ArgumentParser(
    description='generate shell scripts.')
parser.add_argument('--script-dir', default='cifar', type=str,
                    help='scripts save dir')
parser.add_argument('--name', default='CF10', type=str,
                    help='experiment name')
parser.add_argument('--trial', default=3, type=int, metavar='N',
                    help='number of experiment trial (default: 3)')

args = parser.parse_args()

def main():
    os.makedirs(args.script_dir, exist_ok=True)

    datasets = ['cifar10']
    models = ['vcnn']
    optims = ['gd']
    schedulers= ['none']

    perturb_types = ['random']
    # perturb_eps = [0.0] [0.0001, 0.001, 0.01]
    # perturb_eps = [0.0, 0.001, 0.005, 0.01, 0.05]
    perturb_eps = [0.005, 0.05]
    seeds = np.arange(1, args.trial + 1)

    configs = [datasets, models, optims, schedulers, perturb_types, perturb_eps, seeds]

    for n, (data, model, optim, scheduler, perturb_type, perturb_ep, seed) in enumerate(product(*configs)):
        script = '''\
#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o outputs/CIFAR10_PGD_random/%x.out
#SBATCH -e outputs/CIFAR10_PGD_random/%x.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -D /pds/pds1/slurm_workspace/tardis/CAMP

__conda_setup="$('/pds/pds1/slurm_workspace/tardis/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate base
python -u main.py \\
  --seed {seed} \\
  --data {data} \\
  --noise-rate 0.0 \\
  --save-path results/{data}/{model}_m0/PGD_random_{scheduler}/eps_{perturb_ep}/seed_{seed} \\
  --model {model} \\
  --perturb-type {perturb_type} \\
  --tot-epoch 1500 \\
  --optim {optim} \\
  --optim-scheduler {scheduler} \\
  --lr 5e-2 \\
  --momentum 0.0 \\
  --batch-size 4096 \\
  --best-metric acc \\
  --perturb-eps {perturb_ep} \\
  '''.format(jobname='%s-%d' % (args.name, n), seed = seed, data=data, model=model, optim=optim, scheduler=scheduler,
             perturb_type = perturb_type, perturb_ep=perturb_ep, 
             )
        file_path = os.path.join(args.script_dir, 'script_%d.sh' %n)
        with open(file_path, 'wt') as rsh:
            rsh.write(script)
        subprocess.call("sbatch %s" % file_path, shell=True)

if __name__=='__main__':
    main()


