from . import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(120), nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    pw_hash  = db.Column(db.String(256), nullable=False)

    def set_password(self, pw): self.pw_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.pw_hash, pw)

class Project(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    specs   = db.Column(db.JSON)
    code    = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
