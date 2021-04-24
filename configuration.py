import os
import configparser

def config():
    '''
    Function that returns project configuration.
    If there is no config.ini file in project root folder it creates it.
    If this file exists it loads an existing configuration.
    Can be used like this:
    config = config()
    raw_lob_data_folder = config['folders']['raw_lob_data']
    Returns: ConfigParser object
    '''

    config = configparser.ConfigParser()
    wd = os.getcwd()

    if os.path.isfile('project.conf'):
        config.read('project.conf')

    else:
        config['folders'] = {
            'experiments': f'{wd}/Experiments',
            'resampled_data': f'{wd}/Experiments/resampled',
            'raw_lob_data': f'{wd}/Experiments/input/raw/lob',
            'raw_trade_data': f'{wd}/Experiments/input/raw/trades'
            }

        with open('project.conf', 'w') as configfile:    # save
            config.write(configfile)

    return config