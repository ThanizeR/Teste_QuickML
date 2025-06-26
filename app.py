from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import pytz
from io import BytesIO
import zipfile
from pytz import timezone, utc

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
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone("America/Sao_Paulo")))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# --- Wizard code generation functions ---


def generate_streamlit_h5_numeric_input(model_name="Keras .h5 Model with Numeric Input"):
    return f'''import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("Model Deployment - {model_name}")

# Upload do arquivo do modelo .h5
uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
model = None
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded successfully!")

# Input numérico para o modelo
input_number = st.number_input("Enter a numeric value")

if st.button("Predict"):
    if model:
        # Prepara o input para o modelo
        input_data = np.array([[input_number]])
        # Faz a predição (exemplo, ajuste conforme seu modelo)
        prediction = model.predict(input_data)
        st.write(f"Prediction: {{prediction[0][0]:.4f}}")
    else:
        st.error("Please upload a valid model first.")
'''

def generate_streamlit_h5_text_input(model_name="Keras .h5 Model with Text Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model

st.title("Model Deployment - {model_name}")

uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
model = None
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded successfully!")

input_text = st.text_area("Enter your text here")

if st.button("Predict"):
    if model:
        # Aqui você pode preprocessar o texto para o modelo
        # prediction = model.predict(preprocess(input_text))
        prediction = "Prediction example (adjust for your model)"
        st.write(f"Prediction: {{prediction}}")
    else:
        st.error("Please upload a valid model first.")
'''

def generate_streamlit_h5_image_input(model_name="Keras .h5 Model with Image Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Model Deployment - {model_name}")

uploaded_file = st.file_uploader("Upload your .h5 model file", type=["h5"])
model = None
if uploaded_file:
    model = load_model(uploaded_file)
    st.success("Model loaded successfully!")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image = None
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Predict"):
    if model and image:
        # Exemplo de preprocessamento e predição (ajuste conforme seu modelo)
        # image = image.resize((224, 224))
        # input_data = np.expand_dims(np.array(image), axis=0)
        # prediction = model.predict(input_data)
        prediction = "Prediction example (adjust for your model)"
        st.write(f"Prediction: {{prediction}}")
    elif not model:
        st.error("Please upload a valid model first.")
    else:
        st.error("Please upload an image to predict.")
'''

def generate_streamlit_resnet_numeric_input(model_name="ResNet Model with Numeric Input"):
    return f'''import streamlit as st
import numpy as np

st.title("Model Deployment - {model_name}")

input_number = st.number_input("Enter a numeric value")

if st.button("Predict"):
    # Aqui conecte ao seu modelo ResNet para input numérico
    prediction = 0.75  # Exemplo fake
    st.write(f"Prediction: {{prediction:.4f}}")
'''

def generate_streamlit_resnet_text_input(model_name="ResNet Model with Text Input"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_text = st.text_area("Enter your text here")

if st.button("Predict"):
    # Aqui conecte ao seu modelo ResNet para input texto
    prediction = "Prediction example (adjust for your model)"
    st.write(f"Prediction: {{prediction}}")
'''

def generate_streamlit_resnet_image_input(model_name="ResNet Model with Image Input"):
    return f'''import streamlit as st
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image = None
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Predict"):
    # Aqui conecte ao seu modelo ResNet para input imagem
    prediction = "Prediction example (adjust for your model)"
    st.write(f"Prediction: {{prediction}}")
'''

def generate_streamlit_pytorch_numeric_input(model_name="PyTorch Model with Numeric Input"):
    return f'''import streamlit as st
import torch
import numpy as np

st.title("Model Deployment - {model_name}")

input_number = st.number_input("Enter a numeric value")

if st.button("Predict"):
    # Exemplo de predição PyTorch (ajuste para seu modelo)
    prediction = 0.88  # Valor fake
    st.write(f"Prediction: {{prediction:.4f}}")
'''

def generate_streamlit_pytorch_text_input(model_name="PyTorch Model with Text Input"):
    return f'''import streamlit as st

st.title("Model Deployment - {model_name}")

input_text = st.text_area("Enter your text here")

if st.button("Predict"):
    # Exemplo de predição PyTorch texto (ajuste para seu modelo)
    prediction = "Prediction example (adjust for your model)"
    st.write(f"Prediction: {{prediction}}")
'''

def generate_streamlit_pytorch_image_input(model_name="PyTorch Model with Image Input"):
    return f'''import streamlit as st
from PIL import Image

st.title("Model Deployment - {model_name}")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image = None
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("Predict"):
    # Exemplo de predição PyTorch imagem (ajuste para seu modelo)
    prediction = "Prediction example (adjust for your model)"
    st.write(f"Prediction: {{prediction}}")
'''


# --- Funções geradoras Gradio ---

def generate_gradio_h5_numeric_input(model_name="Keras .h5 Model with Numeric Input"):
    return f'''import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Carregue seu modelo .h5 aqui (exemplo comentado)
# model = load_model("seu_modelo.h5")

def predict(number):
    # Converta a entrada para o formato esperado pelo modelo
    input_data = np.array([[number]])
    
    # Faça a predição com o modelo carregado
    # prediction = model.predict(input_data)
    
    # Exemplo de predição fake para evitar erro:
    prediction = np.array([[0.42]])
    
    # Retorne o resultado formatado
    return f"Prediction: {{prediction[0][0]:.4f}}"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''
def generate_gradio_h5_text_input(model_name="Keras .h5 Model with Text Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model

# Carregue seu modelo .h5 (descomente e ajuste o caminho)
# model = load_model("seu_modelo.h5")

def predict(text):
    # Exemplo: processar texto e fazer predição
    # prediction = model.predict(processar_texto(text))
    prediction = "resultado de predição (exemplo)"  # Exemplo fake
    return f"Prediction: {{prediction}}"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_h5_image_input(model_name="Keras .h5 Model with Image Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Carregue seu modelo .h5 (descomente e ajuste o caminho)
# model = load_model("seu_modelo.h5")

def predict(image):
    # Exemplo: preprocessar imagem para o modelo
    # image = image.resize((224, 224))
    # input_data = np.expand_dims(np.array(image), axis=0)
    # prediction = model.predict(input_data)
    prediction = "resultado de predição (exemplo)"  # Exemplo fake
    return f"Prediction: {{prediction}}"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet_numeric_input(model_name="ResNet Model with Numeric Input"):
    return f'''import gradio as gr
import numpy as np

# Carregue seu modelo ResNet (exemplo)
# model = ...

def predict(number):
    input_data = np.array([[number]])
    # prediction = model.predict(input_data)
    prediction = np.array([[0.77]])  # Exemplo fake
    return f"Prediction: {{prediction[0][0]:.4f}}"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet_text_input(model_name="ResNet Model with Text Input"):
    return f'''import gradio as gr

# Carregue seu modelo ResNet (exemplo)
# model = ...

def predict(text):
    # prediction = model.predict(process_text(text))
    prediction = "resultado de predição (exemplo)"  # Exemplo fake
    return f"Prediction: {{prediction}}"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_resnet_image_input(model_name="ResNet Model with Image Input"):
    return f'''import gradio as gr
from PIL import Image
import numpy as np

# Carregue seu modelo ResNet (exemplo)
# model = ...

def predict(image):
    # image = image.resize((224, 224))
    # input_data = np.expand_dims(np.array(image), axis=0)
    # prediction = model.predict(input_data)
    prediction = "resultado de predição (exemplo)"  # Exemplo fake
    return f"Prediction: {{prediction}}"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch_numeric_input(model_name="PyTorch Model with Numeric Input"):
    return f'''import gradio as gr
import torch
import numpy as np

# Carregue seu modelo PyTorch (exemplo)
# model = torch.load("seu_modelo.pth")
# model.eval()

def predict(number):
    input_tensor = torch.tensor([[number]], dtype=torch.float32)
    # with torch.no_grad():
    #     output = model(input_tensor)
    # prediction = output.numpy()
    prediction = np.array([[0.88]])  # Exemplo fake
    return f"Prediction: {{prediction[0][0]:.4f}}"

iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch_text_input(model_name="PyTorch Model with Text Input"):
    return f'''import gradio as gr
import torch

# Carregue seu modelo PyTorch (exemplo)
# model = torch.load("seu_modelo.pth")
# model.eval()

def predict(text):
    # prediction = model(process_text(text))
    prediction = "resultado de predição (exemplo)"  # Exemplo fake
    return f"Prediction: {{prediction}}"

iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''

def generate_gradio_pytorch_image_input(model_name="PyTorch Model with Image Input"):
    return f'''import gradio as gr
import torch
from PIL import Image
import numpy as np

# Carregue seu modelo PyTorch (exemplo)
# model = torch.load("seu_modelo.pth")
# model.eval()

def predict(image):
    # image = image.resize((224, 224))
    # input_tensor = transform(image).unsqueeze(0)
    # with torch.no_grad():
    #     output = model(input_tensor)
    # prediction = output.numpy()
    prediction = "resultado de predição (exemplo)"  # Exemplo fake
    return f"Prediction: {{prediction}}"

iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="Model Deployment - {model_name}")
iface.launch()
'''
def generate_requirements(framework):
    if framework == "streamlit":
        return "streamlit\ntensorflow\npillow"
    else:
        return "gradio\ntensorflow\npillow"

def generate_usage_header(framework, model_name):
    if framework == "streamlit":
        run_cmd = "streamlit run nome_do_arquivo.py"
        install_cmd = "pip install streamlit tensorflow pillow"
        deploy_section = f"""
☁️ 4. Deploy na Nuvem:

🌐 Streamlit Cloud (https://streamlit.io/cloud):
   - Acesse o site e faça login com sua conta GitHub;
   - Clique em 'New App' e selecione o repositório com este código;
   - No campo "Main file path", indique: `nome_do_arquivo.py`;
   - Certifique-se de ter um arquivo `requirements.txt` no repositório com as dependências;
   - Clique em "Deploy" e pronto!

📌 Dica:
   - Streamlit Cloud é ideal para apps interativos com UI direta no navegador;
   - Permite atualizações automáticas a partir do GitHub.
"""
        deploy_note = "📌 Ideal para dashboards interativos com interface rápida e visual."
    else:
        run_cmd = "python nome_do_arquivo.py"
        install_cmd = "pip install gradio tensorflow pillow"
        deploy_section = f"""
☁️ 4. Deploy na Nuvem:

🤗 Hugging Face Spaces (https://huggingface.co/spaces):
   - Acesse o site e faça login;
   - Clique em "Create New Space";
   - Escolha o SDK: `Gradio`;
   - Preencha nome e visibilidade do projeto;
   - Faça upload deste script (`nome_do_arquivo.py`) e do `requirements.txt`;
   - O arquivo será executado automaticamente se contiver:
     ```python
     import gradio as gr
     ...
     interface.launch()
     ```
   - Após o deploy, seu app estará acessível publicamente via URL.

📌 Dica:
   - Hugging Face Spaces é ideal para deploys rápidos e gratuitos com Gradio;
   - Excelente para demonstração de modelos de IA com inputs customizados.
"""
        deploy_note = "📌 Ideal para APIs visuais simples e publicação fácil via Hugging Face Spaces."

    return f'''"""
📦 Instruções de Uso - {model_name}

Este código foi gerado automaticamente pelo QuickML Creator utilizando o framework: {framework.upper()}.
{deploy_note}

🔧 1. Instale as dependências:
   {install_cmd}

▶️ 2. Execute localmente:
   {run_cmd}

🧠 3. Como usar:
   - Faça upload do seu modelo (ex: .h5, .pt);
   - Insira os dados conforme o tipo de entrada (imagem, texto ou número);
   - Veja os resultados da predição diretamente na interface.

{deploy_section}

📁 Arquivo principal: nome_do_arquivo.py
"""
‼️ Agora siga com o seu código para gerar a aplicação web desejada.

Este arquivo serve como ponto de partida. Implemente a lógica do seu modelo, as funções de predição e a interface de usuário conforme o framework escolhido.

Explore os recursos do Streamlit ou Gradio para criar uma experiência interativa e intuitiva para seu projeto de machine learning.

Boa codificação! 🚀
'''


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

import zipfile
import os

def save_zip(framework, model_type, data_type, model_name, zip_path="app_package.zip"):
    # 1. Conteúdo dos arquivos
    code = generate_code(framework, model_type, data_type, model_name)
    readme = generate_usage_header(framework, model_name)
    requirements = generate_requirements(framework)

    # 2. Criação dos arquivos temporários
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(code)
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    with open("README.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    # 3. Compactar em .zip
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write("app.py")
        zipf.write("requirements.txt")
        zipf.write("README.txt")

    # 4. Limpeza dos arquivos temporários (opcional)
    os.remove("app.py")
    os.remove("requirements.txt")
    os.remove("README.txt")

    return zip_path


# --- Routes ---


@app.route('/')
def index():
    return render_template('inicio.html', active_page='index')

@app.route('/gerar_codigo')
@login_required
def wizard_preview():
    return render_template('wizard_preview.html', active_page='wizard_preview')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Exemplo no registro
        if User.query.filter_by(username=username).first():
            flash("Nome de usuário já existe", "danger")
        elif User.query.filter_by(email=email).first():
            flash("E-mail já registrado", "danger")
        elif password != confirm_password:
            flash("As senhas não coincidem", "warning")
        else:
            # salvar usuário
            flash("Conta criada com sucesso! Faça login.", "success")

        # Criar usuário
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Conta criada com sucesso! Faça login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if not user:
            flash("Usuário não encontrado", "danger")
        elif not check_password_hash(user.password, password):
            flash("Senha incorreta", "danger")
        else:
            login_user(user)
            flash("Login realizado com sucesso", "success")
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

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    downloads_dir = os.path.join(app.root_path, 'downloads')
    return send_from_directory(downloads_dir, filename, as_attachment=True)

@app.route('/generate_code', methods=['POST'])
@login_required
def generate_code_route():
    framework = request.form.get('framework')
    model_type = request.form.get('model_type')
    data_type = request.form.get('data_type')
    model_name = request.form.get('model_name')

    if not all([framework, model_type, data_type, model_name]):
        flash("Por favor, preencha todos os campos.", "danger")
        return redirect(request.referrer or url_for('wizard_preview'))

    # Gerar arquivos
    code = generate_code(framework, model_type, data_type, model_name)
    readme = generate_usage_header(framework, model_name)
    requirements = generate_requirements(framework)

    # Nome do ZIP
    zip_filename = f"{framework}_{model_type}_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    # Criar em memória
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr("app.py", code)
        zf.writestr("README.txt", readme)
        zf.writestr("requirements.txt", requirements)
    memory_file.seek(0)

    # Salvar fisicamente para histórico
    downloads_dir = os.path.join(app.root_path, 'downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    file_path = os.path.join(downloads_dir, zip_filename)
    with open(file_path, 'wb') as f:
        f.write(memory_file.getvalue())

    # Hora em fuso de São Paulo
    br_tz = pytz.timezone('America/Sao_Paulo')
    now_br = datetime.now(br_tz)

    # Registrar no banco com horário correto
    download_entry = Download(filename=zip_filename, user_id=current_user.id, timestamp=now_br)
    db.session.add(download_entry)
    db.session.commit()

    # Retornar como download para o navegador (automático)
    return send_file(
        memory_file,
        mimetype='application/zip',
        download_name=zip_filename,
        as_attachment=True
    )

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
    tz_sp = timezone("America/Sao_Paulo")

    for d in downloads:
        if d.timestamp is not None:
            if d.timestamp.tzinfo is None:
                # Marca como UTC e converte para São Paulo
                d.timestamp = utc.localize(d.timestamp).astimezone(tz_sp)
            else:
                # Se já tiver timezone, converte para São Paulo
                d.timestamp = d.timestamp.astimezone(tz_sp)

    return render_template('download_history.html', downloads=downloads)

# --- Run ---
chat_messages = []

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    global chat_messages
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            chat_messages.append({
                'user': current_user.username,
                'message': message,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        return redirect(url_for('chat'))

    return render_template('chat.html', messages=chat_messages)


@app.route('/clear_download_history', methods=['POST'])
def clear_download_history():
    # Limpa todos os downloads do banco de dados
    Download.query.delete()
    db.session.commit()
    flash('Histórico de downloads limpo com sucesso.', 'success')
    return redirect(url_for('download_history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
