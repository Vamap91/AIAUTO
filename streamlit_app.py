import streamlit as st
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Tenta importar mmdet, se não conseguir, avisa o usuário (necessário para deploy)
try:
    from mmdet.apis import init_detector, inference_detector
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False

# Configurações de Página
st.set_page_config(page_title="A.I. AutoInspector - Streamlit", layout="wide")

# --- CSS Customizado ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .damage-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constantes e Modelos ---
CONFIG_FILE = 'carDDModel/dcn_plus_cfg_small.py'
CHECKPOINT_FILE = 'carDDModel/checkpoint.pth'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_models():
    # Modelo de Detecção de Veículo (Faster R-CNN)
    det_model = fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE).eval()
    
    # Modelo de Detecção de Danos (MMDetection)
    damage_model = None
    if MMDET_AVAILABLE:
        if os.path.exists(CONFIG_FILE) and os.path.exists(CHECKPOINT_FILE):
            try:
                damage_model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
            except Exception as e:
                st.error(f"Erro ao carregar o modelo de danos: {e}")
    
    return det_model, damage_model

# --- Funções Utilitárias ---
def get_best_vehicle_box(det_output):
    max_score = 0
    max_bbox = None
    vehicle_classes = [3, 4, 7, 8]  # Car, motorcycle, bus, truck no MS COCO (PyTorch usa 1-based ou 0-based dependendo da versão, COCO original: 3=car)
    # Nota: torchvision fasterrcnn usa COCO labels. 3=car, 4=motorcycle, 7=train, 8=truck
    for i in range(len(det_output['boxes'])):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        label = det_output['labels'][i].item()
        if label in vehicle_classes and score > max_score:
            max_bbox = bbox
            max_score = score
    return max_bbox

def adjust_bbox(bbox, img_width, img_height, margin=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x_margin = int(width * margin)
    y_margin = int(height * margin)
    return [max(0, x1 - x_margin), max(0, y1 - y_margin), min(img_width, x2 + x_margin), min(img_height, y2 + y_margin)]

def process_damage_analysis(result, classification_dict, confidence_threshold=0.3):
    damage_info = {classification_dict[i]: [] for i in classification_dict.keys()}
    # MMDetection result format depends on version. Assuming [bboxes, masks]
    bboxes_list = result[0] if isinstance(result, tuple) else result
    
    for class_idx, bboxes in enumerate(bboxes_list):
        # MMDetection class_idx starts from 0. carDD classes: dent, scratch, crack, glass shatter, lamp broken, tire flat
        label = class_idx + 1
        if label not in classification_dict: continue
        damage_type = classification_dict[label]
        
        for i, bbox in enumerate(bboxes):
            score = bbox[4]
            if score >= confidence_threshold:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)
                damage_info[damage_type].append({'index': i + 1, 'area': area, 'score': score})
    return damage_info

def calculate_costs(damage_info, car_value, car_age, image_size):
    damage_factors = {'dent': 0.3, 'scratch': 0.2, 'crack': 0.3, 'glass_shatter': 0.35, 'lamp_broken': 0.4, 'tire_flat': 0.1}
    total_area = image_size[0] * image_size[1]
    age_factor = max(0.005, (min(car_age, 10) / 10.0))
    
    repair_cost = 0
    details = []
    for d_type, damages in damage_info.items():
        factor = damage_factors.get(d_type, 0.1)
        for d in damages:
            cost = (d['area'] / total_area) * d['score'] * car_value * factor * age_factor
            repair_cost += cost
            details.append({'type': d_type, 'cost': cost, 'score': d['score']})
    return repair_cost, details

# --- Interface Streamlit ---
st.title("🚗 A.I. AutoInspector")
st.subheader("Detecção Inteligente de Danos Veiculares")

if not MMDET_AVAILABLE:
    st.warning("⚠️ MMDetection não detectado. A detecção de danos será simulada ou desabilitada. Certifique-se de instalar as dependências via `setup.sh`.")

# Sidebar para configurações
with st.sidebar:
    st.header("Configurações do Veículo")
    car_model = st.text_input("Modelo do Carro", "Toyota Corolla")
    car_year = st.number_input("Ano do Carro", min_value=1900, max_value=2026, value=2020)
    car_value = st.number_input("Valor Estimado (USD)", min_value=0, value=20000)
    
    st.divider()
    st.info("Este app utiliza IA para identificar danos e estimar custos de reparo baseados no valor do veículo.")

# Upload de Imagem
uploaded_file = st.file_uploader("Escolha uma imagem do carro...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Imagem Original", use_container_width=True)
    
    if st.button("Analisar Danos"):
        with st.status("Processando imagem...", expanded=True) as status:
            det_model, damage_model = load_models()
            
            # 1. Detecção de Veículo
            st.write("Detectando veículo...")
            transform = T.Compose([T.ToTensor()])
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                det_output = det_model(img_tensor)[0]
            
            best_bbox = get_best_vehicle_box(det_output)
            
            if best_bbox is not None:
                bbox_np = best_bbox.detach().cpu().numpy().astype(int)
                adj_bbox = adjust_bbox(bbox_np, image.size[0], image.size[1])
                cropped_img = image.crop(adj_bbox)
                
                with col2:
                    st.image(cropped_img, caption="Veículo Detectado (Crop)", use_container_width=True)
                
                # 2. Detecção de Danos
                if damage_model:
                    st.write("Analisando danos na estrutura...")
                    cropped_np = np.array(cropped_img)
                    damage_result = inference_detector(damage_model, cropped_np)
                    
                    classification_dict = {1: 'dent', 2: 'scratch', 3: 'crack', 4: 'glass_shatter', 5: 'lamp_broken', 6: 'tire_flat'}
                    damage_info = process_damage_analysis(damage_result, classification_dict)
                    
                    # 3. Cálculo de Custos
                    repair_cost, details = calculate_costs(damage_info, car_value, 2026 - car_year, cropped_img.size)
                    
                    status.update(label="Análise concluída!", state="complete")
                    
                    # Exibição de Resultados
                    st.divider()
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        st.metric("Custo Estimado de Reparo", f"${repair_cost:,.2f}")
                        
                        # Tabela de Danos
                        if details:
                            st.write("### Detalhes dos Danos")
                            for d in details:
                                st.markdown(f"""
                                <div class="damage-card">
                                    <strong>Tipo:</strong> {d['type'].replace('_', ' ').capitalize()}<br>
                                    <strong>Confiança:</strong> {d['score']:.2f}<br>
                                    <strong>Custo Est.:</strong> ${d['cost']:,.2f}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("Nenhum dano significativo detectado!")
                    
                    with res_col2:
                        # Gráfico de pizza dos custos
                        if details:
                            fig, ax = plt.subplots()
                            types = [d['type'] for d in details]
                            costs = [d['cost'] for d in details]
                            ax.pie(costs, labels=types, autopct='%1.1f%%', startangle=140)
                            ax.set_title("Distribuição de Custos")
                            st.pyplot(fig)
                else:
                    st.error("Modelo de danos não disponível. Verifique os arquivos no diretório `carDDModel/`.")
            else:
                st.error("Nenhum veículo detectado na imagem. Tente outra foto.")
                status.update(label="Falha na detecção", state="error")

else:
    st.info("Faça o upload de uma imagem para iniciar a inspeção.")
