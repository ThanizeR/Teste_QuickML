from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object("instance.config.Config")

    db.init_app(app)
    login_manager.init_app(app)
    CSRFProtect(app)

    from ..auth import auth_bp
    from ..wizard import wizard_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(wizard_bp)

    with app.app_context():
        db.create_all()

    return app
