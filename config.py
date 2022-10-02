import configparser

CONFIG_PATH = "./config.ini"

config = configparser.ConfigParser()

config['DEFAULT'] = {
    'artifact_dir': 'artifacts'
}
config['WANDB'] = {
    'project_name': 'primKgEval-testWandb'
}
config['TRAINING'] = {
    'n_epochs': '10'
}
config['HPO'] = {
    'n_trials': '5',
    'sampler': 'RandomSampler',
    'stopper': 'early',  # Terminate unpromising trials
    'result_dir': 'hpo_result'
}

with open(CONFIG_PATH, 'w') as configfile:
    config.write(configfile)
