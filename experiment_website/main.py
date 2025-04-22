# TODO: Fix bad audio

import time
import logging
import inspect
import os
import uuid
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from functools import wraps
from helpers import (
    save_user,
    user_exists,
    hash_password,
    update_user_has_done_intro,
    setup_database,
    save_user_info,
    user_has_submitted_data,
    submit_labeling,
    submit_bad_audio,
    INVITE_CODE,
    BYPASS_REGISTRATION_KEY,
    get_path_valid_timestamp,
    all_country_list,
    get_labeling_data,
    validate_json,
    get_users,
    save_user_feedback
)

#######################################################################
# Setup
#######################################################################

# Start the Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", f"{get_path_valid_timestamp()}.txt")
assert os.path.exists(os.path.dirname(log_path)), f"Expected to find `logs` folder in the same folder as `{__file__}`"
logging.basicConfig(filename=log_path, level=logging.DEBUG, format='!MINE! - %(asctime)s %(levelname)s:%(message)s')
logging.info(f"[{inspect.currentframe().f_code.co_name}, None, None, None] --> Server started, logging initialized.")
setup_database()


#######################################################################
# Registration
#######################################################################

@app.route('/register', methods=['GET'])
def serve_register():
    return render_template('register.html')

# Route for user registration

def handle_register(data):
    try:
        successful, failed_key, error_msg = validate_json(
            data, ['username', 'password', 'confirm_password', 'invite_code'], [str, str, str, str]
        )
        if not successful:
            logging.error(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> validate_json failed with `{error_msg}`")
            return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500
        username:str = data['username']
        password:str = data['password']
        confirm_password:str = data['confirm_password']
        invite_code:str = data['invite_code']
        logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Register attempt for username: {username}")

        # Validate username
        if not username.isalnum():
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed register attempt (invalid characters) for username: {username}")
            return jsonify({'error': 'Usernames can only contain letters and numbers'}), 400

        # Validate the access token
        if invite_code != INVITE_CODE:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed register attempt (invalid invite_code) for username: {username}")
            return jsonify({'error': 'Invalid invite code.'}), 400

        # Validate inputs
        if len(password) < 8:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed register attempt (too short password) for username: {username}")
            return jsonify({'error': 'Password must be at least 8 characters long.'}), 400
        if password != confirm_password:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed register attempt (password != password_confirm) for username: {username}")
            return jsonify({'error': 'Passwords do not match.'}), 400
        if user_exists(username):
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed register attempt (username already exists) for username: {username}")
            return jsonify({'error': 'Username already exists.'}), 400

        # Save the new user (default 'has_done_intro' is False)
        save_user(username, password, has_done_intro=False)
        logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> User registered successful for username: {username}")
        return jsonify({'status': 'successfully registered'}), 200
    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    response =  handle_register(data)
    assert len(response) == 2, f"Something went wrong. Expected a jsonify(dict_msg) and an html response code, but found: {response}"
    msg, html_code = response
    if html_code == 200:
        return handle_login({'username': data["username"], 'password': data["password"]})
    else:
        return msg, html_code


#######################################################################
# Login
#######################################################################

def login_required(f):
    @wraps(f)  # Preserve the original function's name
    def wrap(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('serve_login'))
        return f(*args, **kwargs)
    return wrap

@app.route('/login', methods=['GET'])
def serve_login():
    return render_template('login.html')

# Route for user login
def handle_login(data):
    try:
        successful, failed_key, error_msg = validate_json(
            data, ['username', 'password'], [str, str]
        )
        if not successful:
            logging.error(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> validate_json failed with `{error_msg}`")
            return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500

        username:str = data['username']
        password:str = data['password']

        if not username or not password:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed login attempt (empty password and/or username)")
            return jsonify({'error': 'Username and password are required.'}), 400

        # Check if user exists
        users = get_users()
        if not username in users:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed login attempt (username is wrong) for username: {username}")
            return jsonify({'error': 'Invalid username or password.'}), 400


        user = users[username]
        if "password" not in user:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed login attempt (Could not find password despite user being defined) for username: {username}")
        stored_password = user['password']
        if "user_id" not in user:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed login attempt (Could not find user_id despite user being defined) for username: {username}")
        user_id = user['user_id']
        if hash_password(password) != stored_password:
            time.sleep(0.5) # Hacky solution to avoid brute-force attempts
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Failed login attempt (incorrect password) for username: {username}")
            return jsonify({'error': 'Invalid username or password.'}), 400

        # Successful login
        session['username'] = username
        session["user_id"] = user_id
        session["current_submission_path"] = None

        # Redirect based on whether the user has done the intro
        if not users[username].get('has_done_intro', False):  # Check if intro is not done
            return redirect(url_for('serve_intro'))

        # If intro is done, redirect to the main page
        return redirect(url_for('serve_labeling'))

    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    return handle_login(data)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return jsonify({'status': 'Logged out successfully'})

#######################################################################
# Intro Page
#######################################################################

@app.route('/')
def default_redirect():
    key_value = request.args.get('key')  # Replace 'key' with the name of your query parameter
    if key_value == BYPASS_REGISTRATION_KEY:
        random_username = str(uuid.uuid4()).split("-")[-1]
        random_password = str(uuid.uuid4()).split("-")[-1]
        data = {
            'username': random_username,
            'password': random_password,
            'confirm_password': random_password,
            'invite_code': INVITE_CODE,
        }
        response_msg, html_code = handle_register(data)
        if html_code == 200:
            return handle_login({'username': data["username"], 'password': data["password"]})
    return redirect(url_for('serve_intro'))

@app.route('/intro', methods=['GET'])
@login_required
def serve_intro():
    try:
        username = session['username']
        if user_has_submitted_data(username):
            return render_template('intro_tutorial.html')
        else:
            return render_template('intro.html', country_list=all_country_list)
    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.route('/submit_user_info', methods=['POST'])
@login_required
def submit_user_info():
    try:
        data = request.json
        successful, failed_key, error_msg = validate_json(
            data=data,
            expected_keys=["age", "sex", "current_nationality", "birth_nationality", "music_listening", "preferred_language"],
            expected_types=[str, str, str, str, str, str]
        )
        if not successful:
            logging.error(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> validate_json failed with `{error_msg}`")
            return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500

        username:str = session.get('username')
        logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Attempt user info submission for user: `{username}`")
        save_user_info(username, data)
        logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Successfully saved user info submission")
        return jsonify({"status": "success", "message": "Data received successfully"}), 200

    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500



# Route to handle feedback from the intro and mark it as done
@app.route('/intro_done', methods=['POST'])
@login_required
def intro_done():
    try:
        username = session['username']
        update_user_has_done_intro(username)
        return redirect(url_for('serve_labeling'))
    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500


#######################################################################
# Protected routes
#######################################################################

@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        data = request.json
        successful, failed_key, error_msg = validate_json(data=data, expected_keys=["feedback"], expected_types=[str])
        if not successful:
            logging.error(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> validate_json failed with `{error_msg}`")
            return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500

        username:str = session.get('username')
        logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Attempt user feedback submission: `{username}`")
        save_user_feedback(username, data["feedback"])
        logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Successfully saved user info submission")
        return jsonify({"status": "success", "message": "Data received successfully"}), 200

    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500


@app.route('/labeling.html')
@login_required
def serve_labeling():
    return render_template('labeling.html')

@app.route('/what-langauge-do-the-user-prefer')
@login_required
def get_language_preference():
    try:
        all_users = get_users()
        logged_in_username = session["username"]
        assert logged_in_username in all_users, logged_in_username
        user_info = all_users[logged_in_username]
        preferred_language = user_info["user_info"]["preferred_language"]
        assert preferred_language in ["prefer-english", "prefer-danish"], preferred_language
        return jsonify({'preferred_language': preferred_language}), 200
    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.route('/intro_tutorial.html')
@login_required
def serve_intro_tutorial():
    return render_template('intro_tutorial.html')

@app.route('/thank_you_page.html')
@login_required
def serve_thank_you():
    return render_template('thank_you_page.html')

# Fetch image and audio data (only for logged-in users)
@app.route('/get_data', methods=['GET'])
@login_required
def get_data():
    try:
        new_submission_path, audio_path, image_paths, user_current_progress_index, total_number_of_samples = get_labeling_data(session["user_id"])
        # TODO: This is hacky as fuck, fix it
        if all(v is None for v in [new_submission_path, audio_path, image_paths, user_current_progress_index, total_number_of_samples]):
            return redirect(url_for('serve_thank_you'))
        session["current_submission_path"] = new_submission_path
        return jsonify({'images': image_paths,'audio': audio_path, "user_current_progress_index":user_current_progress_index, "total_number_of_samples":total_number_of_samples})
    except Exception as e:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

# Handle the selection data submission (only for logged-in users)
@app.route('/submit_data', methods=['POST'])
@login_required
def submit_data():
    try:
        data = request.json
        successful, failed_key, error_msg = validate_json(
            data, ['is_selected', 'image_paths', 'audio_path', 'liked_song', 'user_interactions'], [list, list, str, str, list]
        )
        if not successful:
            logging.error(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> validate_json failed with `{error_msg}`")
            return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500

        is_selected = data['is_selected']
        image_paths = data['image_paths']
        audio_path =  data['audio_path']
        liked_song =  data['liked_song']
        user_interactions = data['user_interactions']
        was_successful = submit_labeling(session["current_submission_path"], is_selected, image_paths, audio_path, liked_song, user_interactions)

        session["current_submission_path"] = None
        if was_successful:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Successful data submission")
            return jsonify({'status': 'success'}), 200
        else:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Unsuccessful data submission")
            return jsonify({'error': "Unexpected error while saving submission."}), 400

    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.route('/report_bad_audio', methods=['POST'])
@login_required
def report_bad_audio():
    try:
        data = request.json
        successful, failed_key, error_msg = validate_json(data, ["audio_path"], [str])
        if not successful:
            logging.error(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> validate_json failed with `{error_msg}`")
            return jsonify({'error': f'An internal error occurred. Please try again later.'}), 500

        audio_path =  data['audio_path']
        was_successful = submit_bad_audio(audio_path, session["current_submission_path"])
        if was_successful:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Successful bad audio submission")
            return jsonify({'status': 'success'}), 200
        else:
            logging.info(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> Unsuccessful bad audio submission")
            return jsonify({'error': "Unexpected error while saving submission."}), 400

    except Exception:
        logging.critical(f"[{inspect.currentframe().f_code.co_name}, {session.get('username')}, {request.url}, {request.method}] --> !UNCAUGHT ERROR!", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500


# This code will only be entered when running main.py directly (i.e. while debugging), when using Gunicorn (or some different production WSGI) it should not be used.
if __name__ == '__main__':
    app.run(debug=True)