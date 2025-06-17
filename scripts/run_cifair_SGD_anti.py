import argparse
import os
import subprocess

from itertools import product
import numpy as np

parser = argparse.ArgumentParser(
    description='generate shell scripts.')
parser.add_argument('--script-dir', default='cifair', type=str,
                    help='scripts save dir')
parser.add_argument('--name', default='CIFAIR', type=str,
                    help='experiment name')
parser.add_argument('--trial', default=3, type=int, metavar='N',
                    help='number of experiment trial (default: 5)')

args = parser.parse_args()

def main():
    os.makedirs(args.script_dir, exist_ok=True)

    datasets = ['cifair']
    noise_rates = [0.0]
    models = ['vcnn']
    optims = ['sgd']

    batch_sizes = [64] # less than 500 for cifair

    perturb_types = ['anti']
    perturb_eps = [0.005, 0.01, 0.05]
    # perturb_eps = [0.003, 0.03]
    perturb_taus = [5e-2, 5e-1, 5, 50]
    # perturb_taus += [3e-2, 1e-1, 1, 10, 100]
    seeds = np.arange(1, args.trial + 1)

    configs = [datasets, noise_rates, models, optims, batch_sizes, perturb_types, perturb_eps, perturb_taus, seeds]

    for n, (data, noise_rate, model, optim, batch_size, perturb_type, perturb_ep, perturb_tau, seed) in enumerate(product(*configs)):
        script = '''\
#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o outputs/cifair_SGD_anti/%x.out
#SBATCH -e outputs/cifair_SGD_anti/%x.err
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
  --noise-rate {noise_rate} \\
  --save-path results/{data}_nr{noise_rate}/{model}_m0/{optim}_anti/bs_{batch_size}/eps_{perturb_ep}/tau_{perturb_tau}/seed_{seed} \\
  --model {model} \\
  --perturb-type {perturb_type} \\
  --tot-epoch 1500 \\
  --optim {optim} \\
  --lr 5e-2 \\
  --batch-size {batch_size} \\
  --momentum 0.0 \\
  --best-metric acc \\
  --perturb-eps0 0.0 \\
  --perturb-eps {perturb_ep} \\
  --perturb-tau {perturb_tau} \\
  '''.format(jobname='%s-%d' % (args.name, n), seed = seed, data=data, noise_rate=noise_rate, model=model, optim=optim, batch_size=batch_size,
             perturb_type = perturb_type, perturb_ep=perturb_ep, perturb_tau=perturb_tau
             )
        file_path = os.path.join(args.script_dir, 'script_%d.sh' %n)
        with open(file_path, 'wt') as rsh:
            rsh.write(script)
        subprocess.call("sbatch %s" % file_path, shell=True)

if __name__=='__main__':
    main()


