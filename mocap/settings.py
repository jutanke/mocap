from os.path import join, dirname, abspath, isfile
from os import remove
import json
from time import sleep

SETTINGS_FILE = join(dirname(__file__), 'data/settings.json')

SETTINGS = None
if isfile(SETTINGS_FILE):
    with open(SETTINGS_FILE, 'r') as f:
        SETTINGS = json.load(f)


def get_data_path():
    global SETTINGS
    if SETTINGS is None:
        return join(abspath(dirname(__file__)), 'data')
    else:
        return SETTINGS['data_path']


def set_data_path(path):
    global SETTINGS, SETTINGS_FILE
    if SETTINGS is None:
        SETTINGS = {}
    SETTINGS['data_path'] = path
    if isfile(SETTINGS_FILE):
        remove(SETTINGS_FILE)
        sleep(0.5)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(SETTINGS, f)


def reset_settings(user_warning_prompt=True):
    """
    Resets the settings to the default values
    """
    global SETTINGS, SETTINGS_FILE
    if user_warning_prompt:
        print('\n\033[1m\033[93m[mocap][Settings] WARNING! deleting settings!\033[0m')
        choice = input('This cannot be undone! Continue? [y/N]\n').lower()
        print()

        if choice != 'y':
            print('aborting... \033[92mSettings are NOT reset!\033[0m')
            print()
            return

    print('\033[93m[mocap][Settings] resetting Settings...\033[0m')
    print()

    SETTINGS = None
    if isfile(SETTINGS_FILE):
        remove(SETTINGS_FILE)
        sleep(0.5)
