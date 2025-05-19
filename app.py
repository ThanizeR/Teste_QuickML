from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

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

def generate_streamlit_h5(model_name="Keras .h5 Model"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

st.title("Model Deployment - {model_name}")
uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded!")
    # Aqui, você pode adicionar lógica para receber input e usar o modelo

input_data = st.number_input("Input numeric data")
if st.button("Predict"):
    st.write("Prediction result (exemplo)")
'''

def generate_streamlit_resnet(model_name="ResNet Model"):
    return f'''import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    st.write("Imagem processada pronta para predição")
'''

def generate_streamlit_pytorch(model_name="PyTorch Model"):
    return f'''import streamlit as st
import torch

st.title("Model Deployment - {model_name}")

input_text = st.text_area("Enter text for prediction")

if st.button("Predict"):
    st.write("Prediction for input text (exemplo)")
'''

def generate_streamlit_image_input(model_name="Image Input Model"):
    return f'''import streamlit as st
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
'''

def generate_streamlit_numeric_input(model_name="Numeric Input Model"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_number = st.number_input("Enter a numeric value")

if st.button("Predict"):
    st.write("Prediction for numeric input (exemplo)")
'''

def generate_streamlit_text_input(model_name="Text Input Model"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_text = st.text_area("Enter your text here")

if st.button("Predict"):
    st.write("Prediction for text input (exemplo)")
'''

def generate_gradio_h5(model_name="Keras .h5 Model"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model

def predict(input_data):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet(model_name="ResNet Model"):
    return f'''import gradio as gr
from PIL import Image

def predict(image):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch(model_name="PyTorch Model"):
    return f'''import gradio as gr

def predict(text):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_image_input(model_name="Image Input Model"):
    return f'''import gradio as gr

def predict(image):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_numeric_input(model_name="Numeric Input Model"):
    return f'''import gradio as gr

def predict(number):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_text_input(model_name="Text Input Model"):
    return f'''import gradio as gr

def predict(text):
    return "Prediction result (exemplo)"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_code(framework, model_type, model_name):
    # Map framework + model_type to proper generator
    if framework == 'streamlit':
        mapping = {
            'h5': generate_streamlit_h5,
            'resnet': generate_streamlit_resnet,
            'pytorch': generate_streamlit_pytorch,
            'image_input': generate_streamlit_image_input,
            'numeric_input': generate_streamlit_numeric_input,
            'text_input': generate_streamlit_text_input
        }
    else:
        mapping = {
            'h5': generate_gradio_h5,
            'resnet': generate_gradio_resnet,
            'pytorch': generate_gradio_pytorch,
            'image_input': generate_gradio_image_input,
            'numeric_input': generate_gradio_numeric_input,
            'text_input': generate_gradio_text_input
        }

    generator = mapping.get(model_type)
    if not generator:
        return f"# Unsupported model type: {model_type}"
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
                        password=generate_password_hash(password, method='sha256'))
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

@app.route('/generate_code', methods=['POST'])
@login_required
def generate_code_route():
    framework = request.form.get('framework')
    model_type = request.form.get('model_type')
    model_name = request.form.get('model_name')

    code = generate_code(framework, model_type, model_name)
    filename = f"{current_user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"

    os.makedirs('downloads', exist_ok=True)
    file_path = os.path.join('downloads', filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    new_download = Download(filename=filename, user_id=current_user.id)
    db.session.add(new_download)
    db.session.commit()

    flash("Code generated and saved!", "success")
    return redirect(url_for('download_history'))

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
