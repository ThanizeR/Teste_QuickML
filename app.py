from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from flask import request, jsonify
from flask import send_from_directory

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# --- Models ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

class Download(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Wizard code generation functions ---


def generate_streamlit_h5_numeric_input(model_name="Keras .h5 Model with Numeric Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model

st.title("Model Deployment - {model_name}")

uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded!")

input_number = st.number_input("Enter a numeric value")
if st.button("Predict"):
    st.write("Prediction for numeric input (exemplo)")
'''

def generate_streamlit_h5_text_input(model_name="Keras .h5 Model with Text Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model

st.title("Model Deployment - {model_name}")

uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded!")

input_text = st.text_area("Enter your text here")
if st.button("Predict"):
    st.write("Prediction for text input (exemplo)")
'''

def generate_streamlit_h5_image_input(model_name="Keras .h5 Model with Image Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded!")

uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
'''

def generate_streamlit_resnet_numeric_input(model_name="ResNet Model with Numeric Input"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_number = st.number_input("Enter a numeric value")
if st.button("Predict"):
    st.write("Prediction for numeric input (exemplo)")
'''

def generate_streamlit_resnet_text_input(model_name="ResNet Model with Text Input"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_text = st.text_area("Enter your text here")
if st.button("Predict"):
    st.write("Prediction for text input (exemplo)")
'''

def generate_streamlit_resnet_image_input(model_name="ResNet Model with Image Input"):
    return f'''import streamlit as st
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
'''

def generate_streamlit_pytorch_numeric_input(model_name="PyTorch Model with Numeric Input"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_number = st.number_input("Enter a numeric value")
if st.button("Predict"):
    st.write("Prediction for numeric input (exemplo)")
'''

def generate_streamlit_pytorch_text_input(model_name="PyTorch Model with Text Input"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_text = st.text_area("Enter your text here")
if st.button("Predict"):
    st.write("Prediction for text input (exemplo)")
'''

def generate_streamlit_pytorch_image_input(model_name="PyTorch Model with Image Input"):
    return f'''import streamlit as st
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
'''

# --- Funções geradoras Gradio ---

def generate_gradio_h5_numeric_input(model_name="Keras .h5 Model with Numeric Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model

def predict(number):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_h5_text_input(model_name="Keras .h5 Model with Text Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model

def predict(text):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_h5_image_input(model_name="Keras .h5 Model with Image Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model

def predict(image):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet_numeric_input(model_name="ResNet Model with Numeric Input"):
    return f'''import gradio as gr

def predict(number):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet_text_input(model_name="ResNet Model with Text Input"):
    return f'''import gradio as gr

def predict(text):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet_image_input(model_name="ResNet Model with Image Input"):
    return f'''import gradio as gr

def predict(image):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch_numeric_input(model_name="PyTorch Model with Numeric Input"):
    return f'''import gradio as gr

def predict(number):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch_text_input(model_name="PyTorch Model with Text Input"):
    return f'''import gradio as gr

def predict(text):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch_image_input(model_name="PyTorch Model with Image Input"):
    return f'''import gradio as gr

def predict(image):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

# --- Função para escolher a função geradora correta ---

def generate_code(framework, model_type, data_type, model_name):
    key = f"{model_type}_{data_type}"

    if framework == 'streamlit':
        mapping = {
            'h5_numeric_input': generate_streamlit_h5_numeric_input,
            'h5_text_input': generate_streamlit_h5_text_input,
            'h5_image_input': generate_streamlit_h5_image_input,
            'resnet_numeric_input': generate_streamlit_resnet_numeric_input,
            'resnet_text_input': generate_streamlit_resnet_text_input,
            'resnet_image_input': generate_streamlit_resnet_image_input,
            'pytorch_numeric_input': generate_streamlit_pytorch_numeric_input,
            'pytorch_text_input': generate_streamlit_pytorch_text_input,
            'pytorch_image_input': generate_streamlit_pytorch_image_input,
        }
    else:  # gradio
        mapping = {
            'h5_numeric_input': generate_gradio_h5_numeric_input,
            'h5_text_input': generate_gradio_h5_text_input,
            'h5_image_input': generate_gradio_h5_image_input,
            'resnet_numeric_input': generate_gradio_resnet_numeric_input,
            'resnet_text_input': generate_gradio_resnet_text_input,
            'resnet_image_input': generate_gradio_resnet_image_input,
            'pytorch_numeric_input': generate_gradio_pytorch_numeric_input,
            'pytorch_text_input': generate_gradio_pytorch_text_input,
            'pytorch_image_input': generate_gradio_pytorch_image_input,
        }

    generator = mapping.get(key)
    if not generator:
        return f"# Combinação não suportada: {key}"
    return generator(model_name)

# --- Routes ---

@app.route('/')
@login_required
def index():
    return render_template('wizard_preview.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        email = request.form.get('email').strip()
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username, email=email,
                        password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        flash("Registered successfully! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            flash("Invalid credentials", "danger")
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

from flask import send_from_directory

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory('downloads', filename, as_attachment=True)


@app.route('/generate_code', methods=['POST'])
@login_required
def generate_code_route():
    framework = request.form.get('framework')
    model_type = request.form.get('model_type')
    data_type = request.form.get('data_type')
    model_name = request.form.get('model_name')

    if not all([framework, model_type, data_type, model_name]):
        flash("Por favor, preencha todos os campos.", "danger")
        return redirect(request.referrer or url_for('index'))

    code = generate_code(framework, model_type, data_type, model_name)
    filename = f"{framework}_{model_type}_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"

    downloads_dir = os.path.join(app.root_path, 'downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    file_path = os.path.join(downloads_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Salvar referência ao download no banco de dados
    download_entry = Download(filename=filename, user_id=current_user.id)
    db.session.add(download_entry)
    db.session.commit()

    flash("Código gerado e salvo!", "success")
    return redirect(url_for('download_history'))

@app.route('/preview_code', methods=['POST'])
def preview_code():
    data = request.json or {}
    framework = data.get('framework')
    model_type = data.get('model_type')
    data_type = data.get('data_type')
    model_name = data.get('model_name', 'Meu Modelo')

    if not all([framework, model_type, data_type]):
        return jsonify({'error': 'Parâmetros insuficientes'}), 400

    code = generate_code(framework, model_type, data_type, model_name)
    return jsonify({'code': code})


@app.route('/download_history')
@login_required
def download_history():
    downloads = Download.query.filter_by(user_id=current_user.id).order_by(Download.timestamp.desc()).all()
    return render_template('download_history.html', downloads=downloads)

# --- Run ---

# Armazena as mensagens em memória (apenas para demonstração)
chat_messages = []

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    global chat_messages
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            # Adiciona a mensagem ao chat com username e timestamp
            chat_messages.append({
                'user': current_user.username,
                'message': message,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        return redirect(url_for('chat'))

    return render_template('chat.html', messages=chat_messages)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)