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
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config.update(
    MAIL_SERVER=os.getenv('MAIL_SERVER'),
    MAIL_PORT=int(os.getenv('MAIL_PORT')),
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv('MAIL_USERNAME'),
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
)

print(f"MAIL_SERVER: {os.getenv('MAIL_SERVER')}")
print(f"MAIL_PORT: {os.getenv('MAIL_PORT')}")
print(f"MAIL_USERNAME: {os.getenv('MAIL_USERNAME')}")

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

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
#####KERAS
# ========== KERAS (.h5) - ENTRADA NUMÉRICA ==========
def generate_streamlit_h5_numeric_input(model_name="Keras (.h5) - Numeric Input"):
    return f'''import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("Model - Keras (.h5) - Numeric Input")

# ETAPA 1 - CARREGAR MODELO LOCALMENTE
@st.cache_resource
def load_model_from_disk():
    model = load_model("models/{model_name}.h5)

model = load_model_from_disk()

# ETAPA 2 - INPUT NUMÉRICO
input_number = st.number_input("Digite um valor numérico:")

# ETAPA 3 - PREDIÇÃO
if st.button("Prever"):
    input_data = np.array([[input_number]])
    prediction = model.predict(input_data)
    # st.write(f"Predição: {{prediction[0][0]:.4f}}")
    st.write("Predição realizada! (Descomente a linha acima para exibir o valor)")
'''

# ========== KERAS (.h5) - ENTRADA DE TEXTO ==========
def generate_streamlit_h5_text_input(model_name="Keras (.h5) - Text Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model

st.title("Model - Keras (.h5) - Text Input")

# ETAPA 1 - CARREGAR MODELO LOCALMENTE
@st.cache_resource
def load_model_from_disk():
    model = load_model("models/{model_name}.h5")  # Ajuste o caminho
    return model

model = load_model_from_disk()

# ETAPA 2 - INPUT DE TEXTO
input_text = st.text_area("Digite um texto:")

# ETAPA 3 - PREDIÇÃO
if st.button("Analisar"):
    # Exemplo fictício. Substitua pelo seu pré-processamento real:
    # input_processed = preprocess_text(input_text)
    # prediction = model.predict(input_processed)
    prediction = "Positivo (exemplo)"
    st.write(f"Sentimento previsto: {{prediction}}")
'''

# ========== KERAS (.h5) - ENTRADA DE IMAGEM ==========
def generate_streamlit_h5_image_input(model_name="Keras (.h5) - Image Input"):
    return f'''import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Model - Keras (.h5) - Image Input")

# ETAPA 1 - CARREGAR MODELO LOCALMENTE
@st.cache_resource
def load_model_from_disk():
    model = load_model("models/{model_name}.h5")  # Ajuste o caminho
    return model

model = load_model_from_disk()

# ETAPA 2 - UPLOAD DA IMAGEM
uploaded_image = st.file_uploader("Upload da imagem", type=["jpg", "jpeg", "png"])
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

# ETAPA 3 - PRÉ-PROCESSAMENTO
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ETAPA 4 - PREDIÇÃO
if st.button("Classificar") and image:
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    score = prediction[0][0]
    label = "Cachorro" if score >= 0.5 else "Gato"
    st.write(f"Previsão: {{label}} (Score: {{score:.4f}})")
'''

# ========== PYTORCH (.pth) - ENTRADA NUMÉRICA ==========
def generate_streamlit_pytorch_numeric_input(model_name="PyTorch (.pth) - Numeric Input"):
    return f'''import streamlit as st
import torch
import torch.nn as nn

st.title("Model - PyTorch (.pth) - Numeric Input")

# ETAPA 1 - DEFINIR ARQUITETURA
class SimpleNumericModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# ETAPA 2 - CARREGAR MODELO
@st.cache_resource
def load_model():
    model = SimpleNumericModel()
    model.load_state_dict(torch.load("models/{model_name}.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ETAPA 3 - INPUT
input_number = st.number_input("Digite um valor numérico:")

# ETAPA 4 - PREDIÇÃO
if st.button("Prever"):
    with torch.no_grad():
        input_tensor = torch.tensor([[input_number]], dtype=torch.float32)
        output = model(input_tensor)
        prediction = output.item()
        st.write(f"Predição: {{prediction:.4f}}")
'''

# ========== PYTORCH (.pth) - ENTRADA DE TEXTO ==========
def generate_streamlit_pytorch_text_input(model_name="PyTorch (.pth) - Text Input"):
    return f'''import streamlit as st
import torch
import torch.nn as nn

st.title("Modelo - PyTorch (.pth) - Análise de Sentimento")

# ETAPA 1 - DEFINIR ARQUITETURA DO MODELO
class SimpleTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)  # Exemplo: 100 features de entrada

    def forward(self, x):
        return self.fc(x)

# ETAPA 2 - CARREGAR O MODELO
@st.cache_resource
def load_model():
    model = SimpleTextModel()
    model.load_state_dict(torch.load("models/{model_name}.pth", map_location="cpu"))  # Ajuste o caminho
    model.eval()
    return model

model = load_model()

# ETAPA 3 - FUNÇÃO DE PRÉ-PROCESSAMENTO
def preprocess_text(text):
    """
    Substitua por seu pré-processamento real.
    Aqui estamos simulando um vetor de 100 features como entrada.
    """
    vector = torch.randn(1, 100)  # Exemplo fictício
    return vector

# ETAPA 4 - INPUT DO USUÁRIO
input_text = st.text_area("Digite um texto:")

# ETAPA 5 - PREDIÇÃO
if st.button("Analisar Sentimento") and input_text:
    with torch.no_grad():
        input_tensor = preprocess_text(input_text)
        output = model(input_tensor)
        score = torch.sigmoid(output).item()
        sentimento = "Positivo" if score >= 0.5 else "Negativo"
        st.write(f"Sentimento previsto: {{sentimento}} (Score: {{score:.4f}})")
elif st.button("Analisar Sentimento"):
    st.warning("Por favor, digite um texto antes de analisar.")

'''

# ========== PYTORCH (.pth) - ENTRADA DE IMAGEM ==========
def generate_streamlit_pytorch_image_input(model_name="PyTorch (.pth) - Image Input"):
    return f'''import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

st.title("Model - PyTorch (.pth) - Image Input")

# ETAPA 1 - DEFINIR O MODELO
@st.cache_resource
def load_model():
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load("models/{model_name}.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ETAPA 2 - PRÉ-PROCESSAMENTO
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ETAPA 3 - UPLOAD DE IMAGEM
uploaded_image = st.file_uploader("Upload da imagem", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

    if st.button("Classificar"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            score = torch.sigmoid(output).item()
            label = "Cachorro" if score >= 0.5 else "Gato"
            st.write(f"Previsão: {{label}} (Score: {{score:.4f}})")
'''

# ========== PICKLE (.pkl) - ENTRADA NUMÉRICA ==========
def generate_streamlit_resnet_numeric_input(model_name="Pickle (.pkl) - Numeric Input"):
    return f'''import streamlit as st
import numpy as np
import pickle

st.title("Model - Pickle (.pkl) - Numeric Input")

# ETAPA 1 - CARREGAR MODELO LOCALMENTE
@st.cache_resource
def load_model():
    with open("models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ETAPA 2 - INPUT NUMÉRICO
input_number = st.number_input("Digite um valor numérico:")

# ETAPA 3 - PREDIÇÃO
if st.button("Prever"):
    input_data = np.array([[input_number]])
    prediction = model.predict(input_data)
    st.write(f"Predição: {{prediction[0]:.4f}}")
'''

# ========== PICKLE (.pkl) - ENTRADA DE TEXTO ==========
def generate_streamlit_resnet_text_input(model_name="Pickle (.pkl) - Text Input"):
    return f'''import streamlit as st
import pickle

st.title("Model - Pickle (.pkl) - Text Input")

# ETAPA 1 - CARREGAR MODELO
@st.cache_resource
def load_model():
    with open("models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ETAPA 2 - INPUT DE TEXTO
input_text = st.text_area("Digite um texto:")

# ETAPA 3 - PREDIÇÃO
if st.button("Analisar"):
    # input_processed = preprocess(input_text)
    prediction = "Positivo (exemplo)"
    st.write(f"Sentimento previsto: {{prediction}}")
'''

# ========== PICKLE (.pkl) - ENTRADA DE IMAGEM ==========
def generate_streamlit_resnet_image_input(model_name="Pickle (.pkl) - Image Input"):
    return f'''import streamlit as st
from PIL import Image
import numpy as np
import pickle

st.title("Model - Pickle (.pkl) - Image Input")

# ETAPA 1 - CARREGAR MODELO LOCALMENTE
@st.cache_resource
def load_model():
    with open("models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ETAPA 2 - UPLOAD DA IMAGEM
uploaded_image = st.file_uploader("Upload da imagem", type=["jpg", "jpeg", "png"])
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

# ETAPA 3 - EXTRAIR FEATURES
def extract_features(image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    return img_array.flatten().reshape(1, -1)

# ETAPA 4 - PREDIÇÃO
if st.button("Classificar") and image:
    input_features = extract_features(image)
    prediction = model.predict(input_features)
    label_map = {{0: "Gato", 1: "Cachorro"}}
    label = label_map.get(prediction[0], "Desconhecido")
    st.write(f"Previsão: {{label}}")
'''
#########################################GRADIO############

def generate_gradio_h5_numeric_input(model_name="Keras .h5 Model with Numeric Input"):
    return f'''import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# ETAPA 1 - Carregamento do modelo (ajuste o caminho do arquivo .h5)
model = load_model("models/{model_name}.h5")

# ETAPA 2 - Função de predição
def predict(number):
    input_data = np.array([[number]])
    prediction = model.predict(input_data)
    return f"Predição: {{prediction[0][0]:.4f}}"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_h5_text_input(model_name="Keras .h5 Model with Text Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model

# ETAPA 1 - Carregamento do modelo (ajuste o caminho do arquivo .h5)
model = load_model("models/{model_name}.h5")

# ETAPA 2 - Função de predição
def predict(text):
    # Exemplo fictício, substitua pelo pré-processamento real
    # input_processed = preprocess_text(text)
    # prediction = model.predict(input_processed)
    prediction = "Positivo (exemplo)"
    return f"Sentimento previsto: {{prediction}}"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_h5_image_input(model_name="Keras .h5 Model with Image Input"):
    return f'''import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# ETAPA 1 - Carregamento do modelo
model = load_model("models/{model_name}.h5")

# ETAPA 2 - Função de pré-processamento e predição
def predict(image):
    image = image.resize((224, 224))
    input_data = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(input_data)
    score = prediction[0][0]
    label = "Cachorro" if score >= 0.5 else "Gato"
    return f"Previsão: {{label}} (Score: {{score:.4f}})"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_resnet_numeric_input(model_name="Pickle .pkl Model with Numeric Input"):
    return f'''import gradio as gr
import numpy as np
import pickle

# ETAPA 1 - Carregamento do modelo
model = pickle.load(open("models/{model_name}.pkl", "rb"))

# ETAPA 2 - Função de predição
def predict(number):
    input_data = np.array([[number]])
    prediction = model.predict(input_data)
    return f"Predição: {{prediction[0]:.4f}}"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_resnet_text_input(model_name="Pickle .pkl  Model with Text Input"):
    return f'''import gradio as gr
import pickle

# ETAPA 1 - Carregamento do modelo
model = pickle.load(open("models/{model_name}.pkl", "rb"))

# ETAPA 2 - Função de predição
def predict(text):
    # Exemplo: input_processed = preprocess_text(text)
    # prediction = model.predict([input_processed])
    prediction = "Positivo (exemplo)"
    return f"Sentimento previsto: {{prediction}}"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_resnet_image_input(model_name="Pickle .pkl Model with Image Input"):
    return f'''import gradio as gr
from PIL import Image
import numpy as np
import pickle

# ETAPA 1 - Carregamento do modelo
model = pickle.load(open("models/{model_name}.pkl", "rb"))

# ETAPA 2 - Função de predição
def extract_features(image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_vector = img_array.flatten()
    return img_vector

def predict(image):
    input_features = extract_features(image).reshape(1, -1)
    prediction = model.predict(input_features)
    label_map = {0: "Gato", 1: "Cachorro"}
    label = label_map.get(prediction[0], "Desconhecido")
    return f"Previsão: {{label}}"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_pytorch_numeric_input(model_name="PyTorch Model with Numeric Input"):
    return f'''import gradio as gr
import torch
import torch.nn as nn

# ETAPA 1 - Definição e carregamento do modelo
class SimpleNumericModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleNumericModel()
model.load_state_dict(torch.load("models/{model_name}.pth", map_location="cpu"))
model.eval()

# ETAPA 2 - Função de predição
def predict(number):
    with torch.no_grad():
        input_tensor = torch.tensor([[number]], dtype=torch.float32)
        output = model(input_tensor)
        prediction = output.item()
        return f"Predição: {{prediction:.4f}}"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_pytorch_text_input(model_name="PyTorch Model with Text Input"):
    return f'''import gradio as gr
import torch
import torch.nn as nn

# ETAPA 1 - Definição e carregamento do modelo
class SimpleTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleTextModel()
model.load_state_dict(torch.load("models/{model_name}.pth", map_location="cpu"))
model.eval()

# Função fictícia de pré-processamento
def preprocess_text(text):
    return torch.randn(1, 100)

# ETAPA 2 - Função de predição
def predict(text):
    with torch.no_grad():
        input_tensor = preprocess_text(text)
        output = model(input_tensor)
        score = torch.sigmoid(output).item()
        label = "Positivo" if score >= 0.5 else "Negativo"
        return f"Sentimento previsto: {{label}} (Score: {{score:.4f}})"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="{model_name}")
iface.launch()
'''

def generate_gradio_pytorch_image_input(model_name="PyTorch Model with Image Input"):
    return f'''import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# ETAPA 1 - Definição e carregamento do modelo
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
model.load_state_dict(torch.load("models/{model_name}.pth", map_location="cpu"))
model.eval()

# ETAPA 2 - Pré-processamento e predição
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        score = torch.sigmoid(output).item()
        label = "Cachorro" if score >= 0.5 else "Gato"
        return f"Previsão: {{label}} (Score: {{score:.4f}})"

# ETAPA 3 - Interface Gradio
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="{model_name}")
iface.launch()
'''


#############################################
def generate_requirements(framework):
    if framework == "streamlit":
        return "streamlit\ntensorflow\npillow"
    else:
        return "gradio\ntensorflow\npillow"

def generate_usage_header(framework, model_name):
    if framework == "streamlit":
        run_cmd = "streamlit run nome_do_arquivo.py"
        install_cmd = "pip install -r requirements.txt"
        deploy_section = f"""
4. Deploy na Nuvem:

Streamlit Cloud (https://streamlit.io/cloud):
   - Acesse o site e conecte sua conta GitHub;
   - Crie uma nova aplicação e selecione o repositório com seu código;
   - Informe o caminho do arquivo principal (ex: nome_do_arquivo.py);
   - Garanta que o repositório contenha o arquivo requirements.txt com as dependências;
   - O Streamlit Cloud fará o deploy e disponibilizará a aplicação publicamente.

Recomendações:
   - Mantenha seu código e requisitos atualizados no GitHub para atualizações automáticas;
   - Use variáveis de ambiente para proteger credenciais, se necessário;
   - Verifique limites de recursos gratuitos do Streamlit Cloud para seu uso.

"""
        deploy_note = "Este framework é indicado para aplicações interativas com interface gráfica direta no navegador, ideal para dashboards e protótipos rápidos."
        requirements_txt = """streamlit
tensorflow
torch
torchvision
gradio
pillow
numpy
"""
    else:
        run_cmd = "python nome_do_arquivo.py"
        install_cmd = "pip install -r requirements.txt"
        deploy_section = f"""
4. Deploy na Nuvem:

Hugging Face Spaces (https://huggingface.co/spaces):
   - Crie uma conta ou faça login no Hugging Face;
   - Crie um novo Space e escolha o SDK Gradio;
   - Faça upload do seu script Python (ex: nome_do_arquivo.py) e do requirements.txt;
   - O Space detecta e executa automaticamente o arquivo para iniciar seu app;
   - A aplicação ficará disponível via URL pública.

Recomendações:
   - Configure um arquivo requirements.txt preciso para evitar erros no deploy;
   - Utilize variáveis de ambiente para manter segredos seguros;
   - Aproveite o versionamento do Hugging Face para gerenciar atualizações.

"""
        deploy_note = "Indicado para aplicações rápidas, fáceis de publicar e compartilhar com inputs customizados, ideal para demonstrações de modelos ML/IA."
        requirements_txt = """gradio
tensorflow
torch
torchvision
pillow
numpy
"""

    return f'''"""
Instruções de Uso - {model_name}

Este código foi gerado automaticamente pelo QuickML Creator usando o framework: {framework.upper()}.

1. Instale as dependências necessárias:
   {install_cmd}

   Conteúdo sugerido para requirements.txt:
{requirements_txt}

2. Execute o aplicativo localmente:
   {run_cmd}

3. Como usar a aplicação:
   - Faça upload do seu modelo treinado (ex: .h5, .pt, .pkl) no caminho especificado;
   - Insira os dados de entrada conforme o tipo (número, texto ou imagem);
   - Visualize os resultados das predições diretamente na interface web.

{deploy_section}

Arquivo principal da aplicação: nome_do_arquivo.py

Este arquivo serve como ponto de partida para sua aplicação web de machine learning.
Implemente a lógica do modelo, funções de pré-processamento, predição e a interface de usuário conforme o framework escolhido.

Considere também:
- Tratar erros e validar entradas para garantir uma boa experiência;
- Documentar o código para facilitar manutenções futuras;
- Proteger dados sensíveis, utilizando variáveis de ambiente para chaves e credenciais;
- Testar a aplicação localmente antes do deploy para evitar falhas.

Explore os recursos do Streamlit ou Gradio para criar uma experiência interativa e intuitiva para seu projeto de machine learning.

Boa codificação! 🚀
"""
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
def home():
    # Página inicial pública, sem menu, só welcome e botão login
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('home.html')  # seu arquivo com welcome e botão entrar

@app.route('/index')
@login_required
def index():
    # Página inicial após login, com menu
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
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
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

@app.route('/recuperar', methods=['GET', 'POST'])
def recuperar():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        user = User.query.filter_by(email=email).first()

        if not user:
            flash("E-mail não encontrado", "danger")
            return render_template('recuperar.html')

        # Gera token para reset
        token = serializer.dumps(email, salt='recuperar-senha')

        # Monta link para resetar senha
        link = url_for('resetar_senha', token=token, _external=True)

        # Cria mensagem
        msg = Message(
            subject="Recuperação de Senha",
            recipients=[email],
            body=f"Olá {user.username},\n\nUse o link abaixo para redefinir sua senha:\n{link}\n\n"
                 "Se não solicitou, ignore este email."
        )

        try:
            mail.send(msg)
            flash("E-mail de recuperação enviado! Verifique sua caixa de entrada.", "success")
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Erro ao enviar e-mail: {str(e)}", "danger")

    return render_template('recuperar.html')

@app.route('/resetar_senha/<token>', methods=['GET', 'POST'])
def resetar_senha(token):
    try:
        email = serializer.loads(token, salt='recuperar-senha', max_age=3600)  # token válido por 1h
    except SignatureExpired:
        flash("Link expirou. Solicite a recuperação novamente.", "warning")
        return redirect(url_for('recuperar'))
    except BadSignature:
        flash("Link inválido.", "danger")
        return redirect(url_for('recuperar'))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("Usuário não encontrado.", "danger")
        return redirect(url_for('recuperar'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not password or password != confirm_password:
            flash("Senhas não coincidem ou estão vazias.", "warning")
            return render_template('resetar.html')

        user.password = generate_password_hash(password, method='pbkdf2:sha256')
        db.session.commit()
        flash("Senha redefinida com sucesso! Faça login.", "success")
        return redirect(url_for('login'))

    return render_template('resetar.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for('home'))

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