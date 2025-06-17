import argparse
import os
import subprocess

from itertools import product
import numpy as np

parser = argparse.ArgumentParser(
    description='generate shell scripts.')
parser.add_argument('--script-dir', default='MS_rho', type=str,
                    help='scripts save dir')
parser.add_argument('--name', default='MSrho', type=str,
                    help='experiment name')
parser.add_argument('--trial', default=3, type=int, metavar='N',
                    help='number of experiment trial (default: 5)')

args = parser.parse_args()

def main():
    os.makedirs(args.script_dir, exist_ok=True)
    output_dir = args.script_dir
    os.makedirs('/pds/pds1/slurm_workspace/tardis/NoiseDNN/outputs/' + output_dir, exist_ok=True)

    batch_rates = [0.1, 1]

    n_dims = [20]

    sigmas = [0.05, 0.1]
    rhos = [0.5, 0.9, 0.95, 0.99]

    seeds = np.arange(1, args.trial + 1)

    configs = [batch_rates, n_dims, sigmas, rhos, seeds]
    for n, (batch_rate, n_dim, sigma, rho, seed) in enumerate(product(*configs)):
        script = '''\
#!/bin/bash
#SBATCH -J {jobname}
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o outputs/{output_dir}/%x.out
#SBATCH -e outputs/{output_dir}/%x.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -D /pds/pds1/slurm_workspace/tardis/NoiseDNN

__conda_setup="$('/pds/pds1/slurm_workspace/tardis/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate base
python -u main_MatSen.py \\
  --seed {seed} \\
  --save-path results/MatSen/SGD_b{batch_rate}/n_{n_dim}/m_{m_train}/sigma_{sigma}/rho_{rho}/seed_{seed} \\
  --tot-iter {tot_iter} \\
  --log-iter {log_iter} \\
  --batch-size {batch_size} \\
  --n-dim {n_dim} \\
  --m-samples {m_sample} \\
  --m-train {m_train} \\
  --lr 5e-2 \\
  --sigma {sigma} \\
  --rho {rho} \\
  '''.format(jobname='%s-%d' % (args.name, n), output_dir=output_dir,
             tot_iter=10**7, log_iter=10**3, seed = seed, 
             batch_rate = batch_rate, batch_size=int(batch_rate * n_dim * 5),
             n_dim=n_dim, m_sample=5 * n_dim * 5, m_train=n_dim * 5, sigma=sigma, rho=rho
             )
        file_path = os.path.join(args.script_dir, 'script_%d.sh' %n)
        with open(file_path, 'wt') as rsh:
            rsh.write(script)
        subprocess.call("sbatch %s" % file_path, shell=True)

if __name__=='__main__':
    main()


