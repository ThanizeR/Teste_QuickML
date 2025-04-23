from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length
from flask_wtf import FlaskForm

from .models import User, db
from . import login_manager

auth_bp = Blueprint("auth", __name__)

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

class RegisterForm(FlaskForm):
    name  = StringField("Nome Completo", validators=[DataRequired()])
    email = StringField("Email", validators=[Email(), DataRequired()])
    pw    = PasswordField("Senha", validators=[Length(min=6)])
    submit= SubmitField("Registrar")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[Email(), DataRequired()])
    pw    = PasswordField("Senha", validators=[DataRequired()])
    submit= SubmitField("Entrar")

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash("Email já cadastrado"); return redirect(url_for("auth.register"))
        u = User(name=form.name.data, email=form.email.data)
        u.set_password(form.pw.data)
        db.session.add(u); db.session.commit()
        login_user(u)
        return redirect(url_for("wizard.specs"))
    return render_template("register.html", form=form)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        u = User.query.filter_by(email=form.email.data).first()
        if u and u.check_password(form.pw.data):
            login_user(u); return redirect(url_for("wizard.specs"))
        flash("Credenciais inválidas")
    return render_template("login.html", form=form)

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user(); return redirect(url_for("auth.login"))
