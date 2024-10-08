import argparse
from pathlib import Path
from ruamel.yaml import YAML
import pickle as pkl
import os
import shutil
import multiprocessing
from joblib import Parallel, delayed
from optimize_single_basin import run_single_basin as run_basin

#def batch_run_experiment(config_file, max_model_runs, out_dir, use_cores_frac):

parser = argparse.ArgumentParser()
#parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--max_model_runs', type=int, required=True)
parser.add_argument('--dds_trials', type=int, default=1)
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--use_cores_frac', type=float, default=1)
parser.add_argument('--algorithm', type=str, default='SCE')
parser.add_argument('--trial_no', type=int, required=True)

args = parser.parse_args()

# read config file
#with Path(args.config_file).open('r') as fp:
#    yaml = YAML(typ="safe")
#    yaml.allow_duplicate_keys = True
#    cfg = yaml.load(fp)  

#print(cfg)

# extract training dates
#with open(cfg['train_dates_file'], 'rb') as f:
#    train_dates = pkl.load(f)

#print(train_dates)

# list all basins in this experiment    
basins = '87654321' #list(train_dates['start_dates'].keys())
#basins = list(train_dates['start_dates'].keys())
#assert len(basins) == 531

# create output directory
#out_dir_run = Path(args.out_dir) / f"{str(args.config_file).split('/')[-1][:-4]}"
out_dir_run = '/home/u9/yhwang0730/PB-LSTM-Papers/no-snow-SACSMA-SNOW17/' + args.out_dir + '/' + args.out_dir + '_' + str(args.trial_no)
shutil.rmtree(out_dir_run, ignore_errors=True)
os.mkdir(out_dir_run)

# parallel loop over basins
num_cores = multiprocessing.cpu_count()
use_n_cores = int(num_cores*args.use_cores_frac)
#print(f'Using {use_n_cores} cores of {num_cores} total.')
#Parallel(n_jobs=use_n_cores)(delayed(run_basin)(basin, 
#                                                train_dates,
#                                                args.algorithm,
#                                                args.max_model_runs,
#                                                args.dds_trials,
#                                                out_dir_run) 
#                             for basin in basins)

#train_dates1 = cfg['train_start_date'] #"01/10/1999"
#train_dates2 = cfg['train_end_date'] #"30/09/2008"

#train_dates = {'start_dates': '01/01/1945', 'end_dates':'12/31/1988'}
train_dates = {'start_dates': '10/02/1945', 'end_dates':'09/30/1988'}

#run_basin(basins[0], train_dates, args.algorithm, args.max_model_runs, args.dds_trials, out_dir_run)
run_basin(basins, train_dates, args.algorithm, args.max_model_runs, args.dds_trials, out_dir_run)
#run_basin(basins, train_dates1, train_dates2, args.algorithm, args.max_model_runs, args.dds_trials, out_dir_run)


