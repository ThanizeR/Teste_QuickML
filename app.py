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
from keras.models import load_model

def predict_numeric(input_data):
    model = load_model("models/{model_name}.h5")  # ajuste o caminho do seu modelo
    input_array = np.array([input_data], dtype=np.float32)
    pred_prob = model.predict(input_array)[0][0]
    pred_class = 1 if pred_prob >= 0.5 else 0
    return pred_class, pred_prob

def page_numeric():
    st.header("Predição com modelo Keras (.h5) - Entrada Numérica")

    Pregnancies = st.number_input('Número de Gestações', min_value=0, step=1)
    Glucose = st.number_input('Nível de Glicose', min_value=0)
    BloodPressure = st.number_input('Pressão Arterial', min_value=0)
    SkinThickness = st.number_input('Espessura da Pele', min_value=0)
    Insulin = st.number_input('Nível de Insulina', min_value=0)
    BMI = st.number_input('IMC', min_value=0.0, format="%.2f")
    DiabetesPedigreeFunction = st.number_input('Função Pedigree Diabetes', min_value=0.0, format="%.4f")
    Age = st.number_input('Idade', min_value=0, step=1)

    if st.button("Prever"):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]
        try:
            pred_class, pred_prob = predict_numeric(input_data)
            label = "Diabético" if pred_class == 1 else "Não diabético"
            st.success(f"Predição: {{label}} (Probabilidade: {{pred_prob:.4f}})")
        except Exception as e:
            st.error(f"Erro na predição: {{e}}")

menu = st.sidebar.radio(
    "Navegação",
    ["🔢 Entrada Numérica"]
)

def get_selected_page(menu):
    if menu == "🔢 Entrada Numérica":
        return "numeric"

selected_page = get_selected_page(menu)

def main(selected_page):
    if selected_page == "numeric":
        page_numeric()

if __name__ == "__main__":
    main(selected_page)
'''

# ========== KERAS (.h5) - ENTRADA DE TEXTO ==========
def generate_streamlit_h5_text_input(model_name="Keras (.h5) - Text Input"):
    return f'''import streamlit as st
import numpy as np
from keras.models import load_model

def dummy_text_preprocess(text):
    max_len = 100
    vec = np.zeros((1, max_len))
    for i, c in enumerate(text.lower()):
        if i >= max_len:
            break
        vec[0, i] = ord(c) / 255
    return vec

def predict_text(input_vec):
    model = load_model("models/{model_name}.h5")  # ajuste o caminho do seu modelo
    pred_prob = model.predict(input_vec)[0][0]
    pred_class = 1 if pred_prob >= 0.5 else 0
    return pred_class, pred_prob

def page_text():
    st.header("Predição com modelo Keras (.h5) - Entrada Texto")

    input_text = st.text_area("Digite o texto para predição")

    if st.button("Analisar"):
        if not input_text.strip():
            st.warning("Digite algum texto para análise.")
            return
        try:
            input_vec = dummy_text_preprocess(input_text)
            pred_class, pred_prob = predict_text(input_vec)
            label = "Positivo" if pred_class == 1 else "Negativo"
            st.success(f"Predição: {{label}} (Probabilidade: {{pred_prob:.4f}})")
        except Exception as e:
            st.error(f"Erro na predição: {{e}}")

menu = st.sidebar.radio(
    "Navegação",
    ["✍️ Entrada Texto"]
)

def get_selected_page(menu):
    if menu == "✍️ Entrada Texto":
        return "text"

selected_page = get_selected_page(menu)

def main(selected_page):
    if selected_page == "text":
        page_text()

if __name__ == "__main__":
    main(selected_page)
'''

# ========== KERAS (.h5) - ENTRADA DE IMAGEM ==========
def generate_streamlit_h5_image_input(model_name="Keras (.h5) - Image Input"):
    return f'''import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

def predict_malaria(img):
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((1,36,36,3))
    model = load_model("models/{model_name}.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob

def predict_pneumonia(img):
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((1,36,36,1))
    model = load_model("models/{model_name}.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob

def page_malaria():
    st.header("Previsão de Malária")
    uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de malária", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Imagem enviada", use_column_width=True)
            pred_class, pred_prob = predict_malaria(img)
            if pred_class == 1:
                st.success(f"Previsão: Infectado - Probabilidade: {{pred_prob*100:.2f}}%")
            else:
                st.success(f"Previsão: Não está infectado - Probabilidade: {{pred_prob*100:.2f}}%")
        except Exception as e:
            st.error(f"Erro ao prever Malária: {{str(e)}}")

def page_pneumonia():
    st.header("Previsão de Pneumonia")
    uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de pneumonia", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Imagem enviada", use_column_width=True)
            pred_class, pred_prob = predict_pneumonia(img)
            if pred_class == 1:
                st.success(f"Previsão: Pneumonia - Probabilidade: {{pred_prob*100:.2f}}%")
            else:
                st.success(f"Previsão: Saudável - Probabilidade: {{pred_prob*100:.2f}}%")
        except Exception as e:
            st.error(f"Erro ao prever Pneumonia: {{str(e)}}")

def main(selected_page):
    if selected_page == "Malaria":
        page_malaria()
    elif selected_page == "Pneumonia":
        page_pneumonia()

menu = st.sidebar.radio(
    "Navegação",
    ["🦟 Detecção Malária", "🫁 Detecção Pneumonia"]
)

def get_selected_page(menu):
    if menu == "🦟 Detecção Malária":
        return "Malaria"
    elif menu == "🫁 Detecção Pneumonia":
        return "Pneumonia"

selected_page = get_selected_page(menu)

if __name__ == "__main__":
    main(selected_page)

'''

# ========== PICKLE (.pkl) - ENTRADA NUMÉRICA ==========
def generate_streamlit_resnet_numeric_input(model_name="Pickle (.pkl) - Numeric Input"):
    return f'''import streamlit as st
import numpy as np
import pickle

st.title("Predição com modelo Pickle (.sav/.pkl) - Entrada Numérica")

# Função para carregar modelo pickle (.sav ou .pkl)
@st.cache_resource
def load_model():
    with open("models/{model_name}.sav", "rb") as f:  # ajuste o caminho e nome do arquivo
        return pickle.load(f)

model = load_model()

# Entrada numérica - Exemplo: 8 características para diabetes
st.write("Insira os valores numéricos para predição:")

col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.number_input('Número de Gestações', min_value=0, step=1)
with col2:
    Glucose = st.number_input('Nível de Glicose', min_value=0)
with col3:
    BloodPressure = st.number_input('Pressão Arterial', min_value=0)

with col1:
    SkinThickness = st.number_input('Espessura da Pele', min_value=0)
with col2:
    Insulin = st.number_input('Nível de Insulina', min_value=0)
with col3:
    BMI = st.number_input('IMC', min_value=0.0, format="%.2f")

with col1:
    DiabetesPedigreeFunction = st.number_input('Função Pedigree Diabetes', min_value=0.0, format="%.4f")
with col2:
    Age = st.number_input('Idade', min_value=0, step=1)

if st.button("Prever Diabetes"):
    # Preparar input para o modelo (array 2D)
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                            BMI, DiabetesPedigreeFunction, Age]])
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Resultado: A pessoa é diabética")
        else:
            st.success("Resultado: A pessoa não é diabética")
    except Exception as e:
        st.error(f"Erro na predição: {{e}}")

'''

# ========== PICKLE (.pkl) - ENTRADA DE TEXTO ==========
def generate_streamlit_resnet_text_input(model_name="Pickle (.pkl) - Text Input"):
    return f'''import streamlit as st
import pickle

st.title("Predição com modelo Pickle (.sav/.pkl) - Entrada Texto")

@st.cache_resource
def load_model():
    with open("models/{model_name}.pkl", "rb") as f:  # ajuste o nome do arquivo
        return pickle.load(f)

model = load_model()

input_text = st.text_area("Digite o texto para análise")

if st.button("Analisar"):
    try:
        # Aqui normalmente você faria algum pré-processamento do texto,
        # ex: vetorizar, tokenizar, etc. Exemplo:
        # processed_input = preprocess(input_text)
        # prediction = model.predict([processed_input])
        # Para exemplo, vou só simular a predição:
        
        prediction = model.predict([input_text])  # ou outra forma dependendo do modelo
        st.write(f"Resultado da predição: {{prediction[0]}}")
    except Exception as e:
        st.error(f"Erro na predição: {{e}}")

'''

# ========== PICKLE (.pkl) - ENTRADA DE IMAGEM ==========
def generate_streamlit_resnet_image_input(model_name="Pickle (.pkl) - Image Input"):
    return f'''import streamlit as st
from PIL import Image
import numpy as np
import pickle

st.title("Predição com modelo Pickle (.sav/.pkl) - Entrada Imagem")

@st.cache_resource
def load_model():
    with open("models/{model_name}.sav", "rb") as f:  # ajuste o caminho/nome
        return pickle.load(f)

model = load_model()

uploaded_file = st.file_uploader("Faça upload da imagem (jpg, png)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagem carregada", use_column_width=True)

        def extract_features(image):
            # Exemplo: redimensiona e transforma em vetor
            image = image.resize((64,64))
            img_array = np.array(image)/255.0
            return img_array.flatten().reshape(1, -1)

        features = extract_features(img)

        if st.button("Classificar imagem"):
            prediction = model.predict(features)
            label_map = {0: "Classe 0", 1: "Classe 1"}  # ajuste as classes reais
            label = label_map.get(prediction[0], "Desconhecido")
            st.write(f"Previsão: {{label}}")
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {{e}}")

'''
#########################################GRADIO############

def generate_gradio_h5_numeric_input(model_name="Keras .h5 Model with Numeric Input"):
    return f'''import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo Keras .h5
model = load_model("models/{model_name}.h5")

# Função de predição com valor numérico
def predict_risk(value):
    input_data = np.array([[value]])
    prediction = model.predict(input_data)
    score = prediction[0][0]
    status = "Alto Risco" if score > 0.5 else "Baixo Risco"
    return f"{{status}} (Confiança: {{score:.2%}})"

# Interface Gradio
iface = gr.Interface(
    fn=predict_risk,
    inputs=gr.Number(label="Valor Numérico de Entrada"),
    outputs=gr.Text(label="Resultado da Predição"),
    title="Previsão de Risco com Modelo .h5"
)

iface.launch()

'''

def generate_gradio_h5_text_input(model_name="Keras .h5 Model with Text Input"):
    return f'''import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Exemplo de tokenizer fictício (você deve treinar e salvar um real)
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(["exemplo"])  # necessário para evitar erro se o tokenizer for fictício

# Carregar o modelo .h5
model = load_model("models/{model_name}.h5")

# Função de predição com entrada de texto
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)
    score = prediction[0][0]
    label = "Positivo" if score >= 0.5 else "Negativo"
    return f"Sentimento: {{label}} (Confiança: {{score:.2%}})"

# Interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Digite um texto aqui..."),
    outputs=gr.Text(label="Resultado da Predição"),
    title="Classificação de Sentimento com Modelo .h5"
)

iface.launch()

'''
def generate_gradio_h5_image_input(model_name="Keras .h5 Model with Image Input"):
    return f'''import gradio as gr
import numpy as np
from PIL import Image
from keras.models import load_model

# Função de previsão de pneumonia
def predict_pneumonia(img):
    img = Image.fromarray(np.uint8(img))
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,1))
    img = img / 255.0
    model = load_model("{model_name}.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    if pred_class == 1:
        pred_label = "Pneumonia"
    else:
        pred_label = "Saudável"
    return pred_label, float(pred_prob)

# Interface Gradio
iface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="numpy", label="Upload da Imagem de Raio-X"),
    outputs=[
        gr.Text(label="Diagnóstico"),
        gr.Number(label="Probabilidade")
    ],
    title="Previsão de Pneumonia com Modelo .h5",
    description="Faça upload de uma imagem de raio-x para detectar pneumonia (modelo Keras .h5)"
)

iface.launch()

'''

def generate_gradio_resnet_numeric_input(model_name="Pickle .pkl Model with Numeric Input"):
    return f'''import gradio as gr
import pickle

# Carregamento do modelo
with open("{model_name}.sav ou .pkl", "rb") as file:
    diabetes_model = pickle.load(file)

# Função de predição para Diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    try:
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]

        prediction = diabetes_model.predict([user_input])
        if prediction[0] == 1:
            return "A pessoa é diabética"
        else:
            return "A pessoa não é diabética"
    except Exception as e:
        return f"Erro: {{str(e)}}"

# Interface Gradio
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Textbox(label="Número de Gestações"),
        gr.Textbox(label="Nível de Glicose"),
        gr.Textbox(label="Pressão Arterial"),
        gr.Textbox(label="Espessura da Pele"),
        gr.Textbox(label="Nível de Insulina"),
        gr.Textbox(label="IMC"),
        gr.Textbox(label="Função de Pedigree"),
        gr.Textbox(label="Idade")
    ],
    outputs="text",
    title="Previsão de Diabetes com Modelo Pickle (.sav)",
    description="Insira os dados clínicos para prever se o paciente tem diabetes (modelo Random Forest carregado com pickle)"
)

iface.launch()

'''

def generate_gradio_resnet_text_input(model_name="Pickle .pkl  Model with Text Input"):
    return f'''import gradio as gr
import pickle

# Carregar o modelo e o vetor de texto
with open("models/{model_name}.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Função de predição
def predict_sentiment(text):
    if not text.strip():
        return "Texto vazio!"
    try:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        return f"Sentimento previsto: {{prediction}}"
    except Exception as e:
        return f"Erro ao prever: {{str(e)}}"

# Interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Digite um texto"),
    outputs="text",
    title="Classificação de Sentimentos com Pickle (.pkl)",
    description="Classificador treinado em dados de texto. Usa vectorizer + modelo .pkl"
)

iface.launch()

'''

def generate_gradio_resnet_image_input(model_name="Pickle .pkl Model with Image Input"):
    return f'''import gradio as gr
import pickle
import numpy as np
from PIL import Image

# Carrega o modelo treinado
with open("models/{model_name}.pkl", "rb") as f:
    model = pickle.load(f)

# Pré-processamento e predição
def predict_image(img):
    try:
        img = Image.fromarray(np.uint8(img)).convert("RGB")
        img = img.resize((64, 64))  # ajuste conforme necessário
        features = np.array(img) / 255.0
        features = features.flatten().reshape(1, -1)
        prediction = model.predict(features)[0]
        label_map = {0: "Gato", 1: "Cachorro"}
        return label_map.get(prediction, "Desconhecido")
    except Exception as e:
        return f"Erro: {{str(e)}}"

# Interface Gradio
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Imagem de Entrada"),
    outputs="text",
    title="Classificação de Imagem com Pickle (.pkl)",
    description="Modelo treinado para classificar imagens simples (flattened RGB)"
)

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