import json
import os.path
from datetime import datetime
import shutil
from typing import List, Tuple
import uuid
import hashlib
from glob import glob
from simple_asserts import assert_types
import logging
import inspect

########################################################################
# Constants
########################################################################

# Paths
PATH_USER_DATA = "users.json"
PATH_FOLDER_BACKUP_DATA = "backups"
PATH_FOLDER_LABEL_SUBMISSION = 'submissions'
PATH_FOLDER_TUTORIAL_DATA_JSON = os.path.join("static", "info_tutorial.json")
PATH_FOLDER_LABEL_BAD_AUDIO = 'submissions_bad_audio'
PATH_FOLDER_USER_FEEDBACK = "user_feedback"

# Image and audio
image_paths = [f"/static/images/{os.path.basename(p)}" for p in glob("./static/images/*.jpg")]
audio_paths = [f"/static/songs/{os.path.basename(p)}" for p in glob("./static/songs/*.wav")]
all_country_list = ['', 'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'The Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo, Democratic Republic of the', 'Congo, Republic of the', 'Costa Rica', 'Côte d’Ivoire', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Timor (Timor-Leste)', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'The Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea, North', 'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia, Federated States of', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar (Burma)', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Sudan, South', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']

# User defined
number_of_tutorial_samples:int = 3
INVITE_CODE = "supersecret"
BYPASS_REGISTRATION_KEY = "DGexperiment"


########################################################################
# Helpers
########################################################################

def get_labeling_data(user_id) -> Tuple[str, str, List[str], int, int]:
    # Load user data
    user_submission_paths = glob(os.path.join(PATH_FOLDER_LABEL_SUBMISSION, f"{user_id}_tutorial*.json"))
    user_current_progress_index = len(user_submission_paths)

    # Find out if the user still need to complete tutorial tasks
    assert (number_of_tutorial_samples is not None) and (number_of_tutorial_samples > 0)
    if user_current_progress_index < number_of_tutorial_samples:
        with open(PATH_FOLDER_TUTORIAL_DATA_JSON, "r") as f:
            tutorial_data = json.load(f)
        audio_path, image_paths = tutorial_data[str(user_current_progress_index)]
        new_submission_path = os.path.join(PATH_FOLDER_LABEL_SUBMISSION, f"{user_id}_tutorial_{user_current_progress_index:03}.json")
        total_number_of_samples = number_of_tutorial_samples
    else:
        return None, None, None, None, None

    assert os.path.exists(audio_path)
    assert len(image_paths) == 12
    assert all(os.path.exists(p) for p in image_paths)
    assert not os.path.exists(new_submission_path), new_submission_path
    return new_submission_path, audio_path, image_paths, user_current_progress_index, total_number_of_samples


def setup_database():
    # Folders
    folder_paths_to_check = PATH_FOLDER_BACKUP_DATA, PATH_FOLDER_LABEL_SUBMISSION, PATH_FOLDER_LABEL_BAD_AUDIO, PATH_FOLDER_USER_FEEDBACK
    for folder_path_to_check in folder_paths_to_check:
        os.makedirs(folder_path_to_check, exist_ok=True)
        assert os.path.exists(folder_path_to_check), f"Failed to find `{folder_path_to_check}`. This should not be possible!"

    # User json
    if not os.path.exists(PATH_USER_DATA):
        with open(PATH_USER_DATA, "x") as f:
            json.dump({}, f)
    assert os.path.exists(PATH_USER_DATA)

    # Tutorial
    with open(PATH_FOLDER_TUTORIAL_DATA_JSON, "r") as f:
        data = json.load(f)
        assert number_of_tutorial_samples == len(data)
        assert list(data.keys()) == list(sorted(set(data.keys()), key=lambda x:int(x))) == [str(i) for i in range(number_of_tutorial_samples)]
    assert os.path.exists(PATH_FOLDER_TUTORIAL_DATA_JSON)
    logging.info(f"[{inspect.currentframe().f_code.co_name}, None, None, None] --> Database successfully initialized")


def get_path_valid_timestamp():
    timestamp = datetime.now()
    timestamp = str(timestamp).replace(" ", "-").replace(".", "-").replace(":", "-")
    return timestamp


def backup_user_data():
    assert os.path.exists(PATH_USER_DATA), f"Failed to find the `user.json` at path `{PATH_USER_DATA}` file this should not be possible!"
    timestamp = get_path_valid_timestamp()
    shutil.copy(PATH_USER_DATA, os.path.join(PATH_FOLDER_BACKUP_DATA, f"{timestamp}_{PATH_USER_DATA}"))
    return timestamp


def save_user_info(username, data):
    backup_user_data()
    with open(PATH_USER_DATA, 'r+') as f:
        users = json.load(f)
        assert username in users, "Could not find username in database. This should not be possible."
        assert "user_info" in users[username], "user_info should have been defined at user creation."
        assert users[username]["user_info"] == {}, "User information have already been submitted. This should not be possible."
        if username in users:
            users[username]['user_info'] = data
            f.seek(0)
            json.dump(users, f, indent=4)
            f.truncate()

def save_user_feedback(username, feedback):
    timestamp = get_path_valid_timestamp()
    save_path = f"{PATH_FOLDER_USER_FEEDBACK}/{username}.json"
    feedback_submission_data = {
        'timestamp': timestamp,
        'feedback': feedback
    }
    try:
        with open(save_path, 'w') as file:
            json.dump(feedback_submission_data, file)
    except Exception:
        # TODO
        return False
    return True




def save_user(username, password, has_done_intro=False):
    timestamp = backup_user_data()
    user_data = {
        "user_id": ("user"+str(uuid.uuid4())).lower(),
        "timestamp": timestamp,
        "password": hash_password(password),
        "has_done_intro": has_done_intro,
        "user_info": {}
    }
    with open(PATH_USER_DATA, 'r+') as f:
        users = json.load(f)
        users[username] = user_data
        f.seek(0)
        json.dump(users, f, indent=4)
        f.truncate()


def user_exists(username) -> bool:
    if os.stat(PATH_USER_DATA).st_size == 0:
        return False
    with open(PATH_USER_DATA, 'r') as f:
        users = json.load(f)
        return username in users


def user_has_submitted_data(username) -> bool:
    assert os.stat(PATH_USER_DATA).st_size != 0, "I feel like the user should have been defined if this function is called."
    assert user_exists(username)
    with open(PATH_USER_DATA, 'r') as f:
        users = json.load(f)
        if users[username]["user_info"] != {}:
            return True
    return False


def submit_labeling(save_path:str, is_selected:List[int], image_paths:List[str], audio_path:str, liked_song:str, user_interactions:List[dict]) -> bool:
    assert isinstance(save_path, str)
    assert not os.path.exists(save_path)
    assert isinstance(is_selected, list)
    assert len(is_selected) == 12
    assert all(isinstance(i, int) and (i in [0, 1]) for i in is_selected)
    assert isinstance(image_paths, list)
    assert len(image_paths) == 12
    assert isinstance(audio_path, str)
    assert isinstance(liked_song, str)
    assert liked_song in ["yes", "no", "unsure"]
    assert isinstance(user_interactions, list)

    # Prepare the data to save
    timestamp = get_path_valid_timestamp()
    submission_data = {
        'timestamp': timestamp,
        'is_selected': is_selected,
        'image_paths': image_paths,
        'audio_path': audio_path,
        'liked_song': liked_song,
        'user_interactions': user_interactions
    }
    try:
        with open(save_path, 'w') as file:
            json.dump(submission_data, file)
    except Exception:
        # TODO LOG
        return False
    return True


def submit_bad_audio(audio_path:str, submission_path:str) -> bool:
    assert isinstance(audio_path, str), audio_path
    assert isinstance(submission_path, str), submission_path
    assert not os.path.exists(submission_path), submission_path
    timestamp = get_path_valid_timestamp()
    filename = os.path.join(PATH_FOLDER_LABEL_BAD_AUDIO, f'bad_audio_{timestamp}.json')
    try:
        with open(filename, 'w') as file:
            json.dump(audio_path, file)
        with open(submission_path, 'w') as file:
            json.dump("{}", file)
    except Exception:
        # TODO LOG
        return False
    return True


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def update_user_has_done_intro(username):
    backup_user_data()
    with open(PATH_USER_DATA, 'r+') as f:
        users = json.load(f)
        if username in users:
            users[username]['has_done_intro'] = True
            f.seek(0)
            json.dump(users, f, indent=4)
            f.truncate()


def get_tutorial_dataset():
    with open(PATH_FOLDER_TUTORIAL_DATA_JSON, "r") as f:
        data = json.load(f)
        data = {int(k):v for (k,v) in data.items()}
    return data


def validate_json(data: dict, expected_keys: list, expected_types: list):
    """
    Validates if all expected keys are present in the given data and checks if the values match the expected types.

    :param data: The JSON data from request.json (dictionary).
    :param expected_keys: A list of expected keys (strings) to be checked in the data.
    :param expected_types: A list of expected data types corresponding to each key in expected_keys.
    :return: Tuple (True, None, None) if all keys and types are valid.
             Otherwise, returns (False, 'key', 'error message') indicating which key is missing, has no value, or an unexpected type.
    """

    assert_types([data, expected_types, expected_types], [dict, list, list], ["data", "expected_types", "expected_types"])

    for (key, expected_type) in zip(data.keys(), expected_types):
        if key not in expected_keys:
            return False, key, f"`{key}` is missing for"
        if data[key] is None:
            return False, key, "value not defined for `{key=}`"
        if not isinstance(data[key], expected_type):
            return False, key, f"unexpected type `{type(data[key])}` for `{key=}`"
    return True, None, None


def get_users() -> dict:
    if os.stat(PATH_USER_DATA).st_size == 0:
        return {}
    with open(PATH_USER_DATA, 'r') as f:
        users = json.load(f)
    return users