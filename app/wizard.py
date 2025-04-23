from flask import Blueprint, render_template, session, redirect, url_for, request, abort, send_file
from flask_login import login_required, current_user
from wtforms import SelectField, SubmitField
from flask_wtf import FlaskForm
import tempfile

from .models import Project, db
from .services import generate_code

wizard_bp = Blueprint("wizard", __name__, url_prefix="/wizard")

MODEL_TYPES = [("PyTorch","PyTorch"),("Scikit-learn","Scikit-learn"),
               ("TensorFlow","TensorFlow"),("Keras","Keras")]
INPUT_TYPES = [("imagem","Imagem"),("numerico","Numérico"),("texto","Texto")]
FRAMEWORKS  = [("Streamlit","Streamlit"),("Gradio","Gradio"),("Flask","Flask")]

class SpecForm(FlaskForm):
    model_type = SelectField("Modelo", choices=MODEL_TYPES)
    input_type = SelectField("Tipo de Dados", choices=INPUT_TYPES)
    framework  = SelectField("Framework", choices=FRAMEWORKS)
    submit     = SubmitField("Gerar Prévia")

@wizard_bp.route("/specs", methods=["GET", "POST"])
@login_required
def specs():
    form = SpecForm(data=session.get("specs"))
    if form.validate_on_submit():
        session["specs"] = form.data
        return redirect(url_for("wizard.preview"))
    return render_template("wizard_specs.html", form=form)

@wizard_bp.route("/preview", methods=["GET", "POST"])
@login_required
def preview():
    specs = session.get("specs")
    if not specs: return redirect(url_for("wizard.specs"))
    code = generate_code(specs["model_type"], specs["input_type"], specs["framework"])
    if request.method == "POST":
        prj = Project(specs=specs, code=code, user_id=current_user.id)
        db.session.add(prj); db.session.commit()
        session.pop("specs")
        return redirect(url_for("wizard.modify", pid=prj.id))
    return render_template("wizard_preview.html", code=code)

@wizard_bp.route("/modify/<int:pid>", methods=["GET", "POST"])
@login_required
def modify(pid):
    prj = Project.query.get_or_404(pid)
    if prj.user_id != current_user.id: abort(403)
    form = SpecForm(data=prj.specs)
    if form.validate_on_submit():
        prj.specs = form.data
        prj.code  = generate_code(*(form.data.values()))
        db.session.commit()
        return redirect(url_for("wizard.modify", pid=pid))
    return render_template("wizard_modify.html", form=form, code=prj.code, project=prj)

@wizard_bp.route("/download/<int:pid>")
@login_required
def download(pid):
    prj = Project.query.get_or_404(pid)
    if prj.user_id != current_user.id: abort(403)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    tmp.write(prj.code.encode()); tmp.close()
    fn = f"{prj.specs['model_type']}_{prj.specs['input_type']}_{prj.specs['framework']}.py"
    return send_file(tmp.name, as_attachment=True, download_name=fn)
