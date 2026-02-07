



import os
import json
import hashlib

USER_DB = os.path.join(os.path.dirname(__file__), "users.json")
SUPERUSER = "admin"

def _read_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_users(users):
    with open(USER_DB, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    users = _read_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": _hash(password),
        "role": "admin" if username == SUPERUSER else "user"
    }
    _write_users(users)
    return True, "User registered."

def login(username, password):
    users = _read_users()
    hashed = _hash(password)
    if username not in users or users[username]["password"] != hashed:
        return False, "Invalid credentials."
    return True, "Login successful."

def get_users():
    return _read_users()

def delete_user(username):
    users = _read_users()
    if username in users:
        del users[username]
        _write_users(users)
        return True
    return False

def is_admin(username):
    return username == SUPERUSER




##import json
##import os
##
##USER_FILE = 'modules/users.json'
##
##def load_users():
##    if not os.path.exists(USER_FILE):
##        return {}
##    with open(USER_FILE, 'r') as f:
##        return json.load(f)
##
##def save_users(users):
##    with open(USER_FILE, 'w') as f:
##        json.dump(users, f, indent=2)
##
##def login(username, password):
##    users = load_users()
##    if username in users and users[username]['password'] == password:
##        return True, "Login successful"
##    return False, "Invalid credentials"
##
##def signup(username, password):
##    users = load_users()
##    if username in users:
##        return False, "Username already exists"
##    users[username] = {
##        "password": password,
##        "role": "user"
##    }
##    save_users(users)
##    return True, "Signup successful"
##



### modules/auth_manager.py
##import os
##import json
##import hashlib
##
##USER_FILE = os.path.join(os.path.dirname(__file__), "users.json")
##
##def load_users():
##    if not os.path.exists(USER_FILE):
##        return {}
##    with open(USER_FILE, "r") as f:
##        return json.load(f)
##
##def save_users(users):
##    with open(USER_FILE, "w") as f:
##        json.dump(users, f, indent=2)
##
##def hash_password(password):
##    return hashlib.sha256(password.encode()).hexdigest()
##
##def signup(username, password):
##    users = load_users()
##    if username in users:
##        return False, "User already exists"
##    users[username] = {
##        "password": hash_password(password)
##    }
##    save_users(users)
##    return True, "User created"
##
##def login(username, password):
##    users = load_users()
##    if username not in users:
##        return False, "User not found"
##    if users[username]["password"] != hash_password(password):
##        return False, "Incorrect password"
##    return True, "Login successful"
