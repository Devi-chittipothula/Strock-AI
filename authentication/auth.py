
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "strokeai.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_users_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)


def register_user(name, email, password):
    create_users_table()

    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name.strip(), email.strip().lower(), hash_password(password))
            )
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    except sqlite3.Error:
        return False, "Unable to create account. Please try again."


def login_user(email, password):
    create_users_table()

    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, name, email FROM users WHERE email = ? AND password = ?",
            (email.strip().lower(), hash_password(password))
        ).fetchone()

    if user:
        return True, {
            "id": user[0],
            "name": user[1],
            "email": user[2]
        }

    return False, None
