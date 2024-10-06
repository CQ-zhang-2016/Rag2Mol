import subprocess
from glob import glob 
import os.path as osp
import time
import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--srcdir', type=str, default='../data/test_set')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='./results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for filename in os.listdir(args.srcdir):
        if '.sdf' in filename:
            basename = filename.split('.sdf')[0]
            pdbname = basename +'.pdb'
            chain_id = basename[-1]

            print(basename)
            pdb_file = osp.join(args.srcdir, pdbname)
            lig_file = osp.join(args.srcdir, filename)
            
            command = f'taskset -c {args.channel} python sample4pdb.py --pdb_file {pdb_file} --sdf_file {lig_file} --outdir {args.outdir} --pdb_chain {chain_id}'

            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            # Check if fpocket command was successful
            if result.returncode == 0:
                print('executed successfully.')
                print('Output:')
                print(result.stdout)
            else:
                print('execution failed.')
                print('Error:')
                print(result.stderr)


