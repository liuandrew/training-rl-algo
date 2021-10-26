import pandas as pd
import pickle
from datetime import datetime
import os
import shutil

def convert_config_to_command(file):
    '''
    when passed a file name in experiment_configs, load the config and turn it into
    a command line line to run the experiment
    '''
    config = pickle.load(open('experiment_configs/' + file, 'rb'))
    run_string = 'python main.py '
    for key in config:
        if config[key] is True:
            run_string = run_string + '--' + key.replace('_', '-') + ' '
        else:
            run_string = run_string + '--' + key.replace('_', '-')  + ' ' + str(config[key]) + ' '

    #additionally add file name flag
    run_string = run_string + '--config-file-name ' + file + ' '
    return run_string




def save_exp_log(exp_log):
    '''
    save an updated experiment log
    '''
    pickle.dump(exp_log, open('experiment_log', 'wb'))
    



def load_exp_log():
    '''
    load experiment log to globals
    '''
    exp_log = pickle.load(open('experiment_log', 'rb'))
    return exp_log


    

def add_exp_row(file):
    '''
    Add a config to the experiment log
    '''
    exp_log = load_exp_log()
    config = pickle.load(open('experiment_configs/' + file, 'rb'))
    index = len(exp_log)
    exp_log = exp_log.append(config, ignore_index=True)
    exp_log.loc[index, 'begin'] = datetime.now()
    exp_log.loc[index, 'file'] = file
    save_exp_log(exp_log)




def write_latest_exp_complete(file):
    '''
    Write the time at which the experiment is completed for a certain filename
    Note that if there are multiple entries in the experiment log with the 
    same filename, we will just pick the one with highest index to update
    '''
    exp_log = load_exp_log()

    idx = exp_log[exp_log['file'] == file].index.max()
    exp_log.loc[idx, 'end'] = datetime.now()
    exp_log.loc[idx, 'success'] = True
    save_exp_log(exp_log)




def run_experiment(file):
    '''
    Pass a config file to run an experiment
    Save the experiment to experiment log
    If the experiment is successfully complete ('end' column is filled)
        then archive the config file
    Otherwise delete the row that was added
    '''
    add_exp_row(file)
    run_string = convert_config_to_command(file)
    os.system(run_string)

    exp_log = load_exp_log()
    idx = exp_log[exp_log['file'] == file].index.max()
    if exp_log.loc[idx, 'success']:
        #experiment completed successfully
        ext = str(int(datetime.now().timestamp()))
        shutil.move('experiment_configs/' + file, 'experiment_configs/archive/' + file + ext)
    else:
        exp_log.loc[idx, 'success'] = False

    save_exp_log(exp_log)