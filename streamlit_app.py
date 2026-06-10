import streamlit as st
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import base64
import json
import io
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIGURAÇÃO DE PÁGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="A.I. AutoInspector",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    background-color: #007bff;
    color: white;
    font-weight: bold;
}
.damage-card {
    padding: 1.2rem;
    border-radius: 0.5rem;
    background-color: white;
    box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.1);
    margin-bottom: 0.8rem;
    border-left: 4px solid #007bff;
}
.cost-box {
    background: linear-gradient(135deg, #007bff, #0056b3);
    color: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

DAMAGE_LABELS_PT = {
    'dent':          'Amassado',
    'scratch':       'Arranhão',
    'crack':         'Trinca',
    'glass_shatter': 'Vidro Estilhaçado',
    'lamp_broken':   'Farol Quebrado',
    'tire_flat':     'Pneu Furado'
}

DAMAGE_FACTORS = {
    'dent':          0.30,
    'scratch':       0.20,
    'crack':         0.30,
    'glass_shatter': 0.35,
    'lamp_broken':   0.40,
    'tire_flat':     0.10
}

# ─────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────
@st.cache_resource
def load_vehicle_detector():
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE).eval()
    return model

# ─────────────────────────────────────────
# FUNÇÕES UTILITÁRIAS
# ─────────────────────────────────────────
def get_best_vehicle_box(det_output, threshold=0.3):
    vehicle_classes = [3, 4, 6, 8]
    best_score = 0
    best_bbox = None
    for i in range(len(det_output['boxes'])):
        label = det_output['labels'][i].item()
        score = det_output['scores'][i].item()
        if label in vehicle_classes and score > best_score and score > threshold:
            best_bbox = det_output['boxes'][i]
            best_score = score
    return best_bbox, best_score

def adjust_bbox(bbox, img_width, img_height, margin=0.1):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    xm, ym = int(w * margin), int(h * margin)
    return [
        max(0, x1 - xm),
        max(0, y1 - ym),
        min(img_width,  x2 + xm),
        min(img_height, y2 + ym)
    ]

def draw_bbox_on_image(image, bbox):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(bbox, outline="#00FF00", width=4)
    return img_copy

def image_to_base64(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def calculate_costs(damages: list, car_value: float, car_age: int) -> tuple:
    age_factor = max(0.05, min(car_age, 10) / 10.0)
    total_cost = 0
    details = []
    for d in damages:
        dtype  = d.get('type', 'scratch')
        score  = float(d.get('confidence', 0.5))
        factor = DAMAGE_FACTORS.get(dtype, 0.2)
        severity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        severity_factor = severity_map.get(d.get('severity', 'medium'), 0.6)
        cost = car_value * factor * score * severity_factor * age_factor
        total_cost += cost
        details.append({
            'type':     dtype,
            'label':    DAMAGE_LABELS_PT.get(dtype, dtype.replace('_', ' ').title()),
            'score':    score,
            'severity': d.get('severity', 'medium'),
            'location': d.get('location', 'não especificado'),
            'cost':     cost
        })
    return total_cost, details

# ─────────────────────────────────────────
# ANÁLISE GPT-4o VISION
# ─────────────────────────────────────────
def analyze_damage_gpt4o(image: Image.Image, api_key: str, car_model: str, car_year: int) -> dict:
    client = OpenAI(api_key=api_key)
    img_b64 = image_to_base64(image)

    prompt = f"""Você é um especialista em avaliação de danos veiculares para seguradoras.
Analise esta imagem do veículo ({car_model}, {car_year}) e identifique todos os danos visíveis.

Responda APENAS com um JSON válido, sem texto adicional, no seguinte formato:
{{
  "vehicle_detected": true,
  "damages": [
    {{
      "type": "dent",
      "confidence": 0.92,
      "severity": "high",
      "location": "porta dianteira esquerda",
      "description": "amassado profundo com deformação da lataria"
    }}
  ],
  "overall_condition": "poor",
  "summary": "Resumo geral dos danos encontrados"
}}

Tipos de dano válidos: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
Severidade: low, medium, high
Condição geral: excellent, good, fair, poor, critical

Se não houver danos visíveis, retorne damages como lista vazia.
Se não houver veículo na imagem, retorne vehicle_detected como false."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    # Remove possíveis blocos markdown ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    result = json.loads(raw.strip())
    result['_usage'] = {
        'input_tokens':  response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'cost_usd':      (response.usage.prompt_tokens * 2.50 / 1_000_000) +
                         (response.usage.completion_tokens * 10.00 / 1_000_000)
    }
    return result

# ─────────────────────────────────────────
# INTERFACE PRINCIPAL
# ─────────────────────────────────────────
st.title("🚗 A.I. AutoInspector")
st.subheader("Detecção Inteligente de Danos Veiculares — Powered by GPT-4o Vision")

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("🔑 API OpenAI")
    api_key = st.text_input(
        "Chave da API (OPENAI_API_KEY)",
        type="password",
        placeholder="sk-...",
        help="Sua chave da API OpenAI. Não é armazenada."
    )

    st.divider()

    st.subheader("🚘 Dados do Veículo")
    car_model_name = st.text_input("Modelo", "Toyota Corolla")
    car_year       = st.number_input("Ano", min_value=1990, max_value=2026, value=2020)
    car_value      = st.number_input("Valor Estimado (USD)", min_value=1000, value=20000, step=500)

    st.divider()

    st.subheader("🔬 Detecção de Veículo")
    use_vehicle_detection = st.toggle("Usar Faster R-CNN para recortar veículo", value=True)
    det_threshold = st.slider("Confiança mínima", 0.1, 0.9, 0.3, 0.05)

    st.divider()
    st.caption(f"Dispositivo PyTorch: `{DEVICE}`")
    st.caption("Modelo de análise: `gpt-4o` (Vision)")
    st.caption("Custo estimado: ~$0.005–$0.01 por análise")

# ── UPLOAD ───────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Escolha uma imagem do carro...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Imagem Original", use_container_width=True)

    if st.button("🔍 Analisar Danos com GPT-4o"):

        # Validar API key
        if not api_key or not api_key.startswith("sk-"):
            st.error("❌ Insira uma chave de API OpenAI válida no painel lateral.")
            st.stop()

        with st.status("Processando imagem...", expanded=True) as status:

            # ── ETAPA 1: Detecção do veículo ──
            img_to_analyze = image
            cropped_img    = None
            confidence     = 0.0

            if use_vehicle_detection:
                st.write("🚗 Detectando veículo com Faster R-CNN...")
                det_model  = load_vehicle_detector()
                transform  = T.Compose([T.ToTensor()])
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    det_output = det_model(img_tensor)[0]

                best_bbox, confidence = get_best_vehicle_box(det_output, threshold=det_threshold)

                if best_bbox is not None:
                    bbox_np      = best_bbox.detach().cpu().numpy().astype(int)
                    adj_bbox     = adjust_bbox(bbox_np, image.size[0], image.size[1])
                    cropped_img  = image.crop(adj_bbox)
                    annotated    = draw_bbox_on_image(image, adj_bbox)
                    img_to_analyze = cropped_img

                    with col2:
                        st.image(
                            annotated,
                            caption=f"Veículo Detectado (conf: {confidence:.0%})",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Veículo não detectado pelo Faster R-CNN. Enviando imagem completa para o GPT-4o.")
                    with col2:
                        st.image(image, caption="Imagem completa (sem crop)", use_container_width=True)
            else:
                with col2:
                    st.image(image, caption="Imagem completa (detecção desativada)", use_container_width=True)

            # ── ETAPA 2: Análise GPT-4o ──
            st.write("🤖 Analisando danos com GPT-4o Vision...")
            try:
                car_age    = 2026 - car_year
                gpt_result = analyze_damage_gpt4o(img_to_analyze, api_key, car_model_name, car_year)

                if not gpt_result.get('vehicle_detected', True):
                    st.error("❌ O GPT-4o não identificou um veículo na imagem.")
                    status.update(label="Falha na análise", state="error")
                    st.stop()

                damages             = gpt_result.get('damages', [])
                repair_cost, details = calculate_costs(damages, car_value, car_age)
                usage               = gpt_result.get('_usage', {})

                status.update(label="✅ Análise concluída!", state="complete")

                # ── ETAPA 3: Resultados ──
                st.divider()
                st.subheader(f"📋 Relatório — {car_model_name} ({car_year})")

                # Resumo do GPT
                if gpt_result.get('summary'):
                    st.info(f"💬 **Avaliação GPT-4o:** {gpt_result['summary']}")

                # Métricas principais
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 Custo Estimado", f"${repair_cost:,.2f}")
                m2.metric("🔢 Danos Encontrados", len(details))
                m3.metric("📊 Condição Geral", gpt_result.get('overall_condition', 'N/A').upper())
                m4.metric("💵 Custo da Chamada API", f"${usage.get('cost_usd', 0):.5f}")

                st.divider()
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    if details:
                        st.write("#### 🔍 Detalhes dos Danos")
                        severity_colors = {
                            'low':    '#28a745',
                            'medium': '#ffc107',
                            'high':   '#dc3545'
                        }
                        severity_labels = {
                            'low': 'Leve', 'medium': 'Médio', 'high': 'Grave'
                        }
                        for d in details:
                            color = severity_colors.get(d['severity'], '#007bff')
                            sev   = severity_labels.get(d['severity'], d['severity'])
                            st.markdown(f"""
                            <div class="damage-card" style="border-left-color: {color};">
                                <strong>{d['label']}</strong>
                                &nbsp;<span style="color:{color};font-weight:bold;">[{sev}]</span><br>
                                📍 <em>{d['location']}</em><br>
                                🎯 Confiança: {d['score']:.0%} &nbsp;|&nbsp;
                                💰 Custo Est.: <strong>${d['cost']:,.2f}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Nenhum dano identificado pelo GPT-4o!")

                with res_col2:
                    if details:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        labels = [d['label'] for d in details]
                        costs  = [d['cost']  for d in details]
                        colors = ['#dc3545', '#ffc107', '#007bff', '#28a745', '#6f42c1', '#fd7e14']
                        ax.pie(
                            costs,
                            labels=labels,
                            autopct='%1.1f%%',
                            startangle=140,
                            colors=colors[:len(details)]
                        )
                        ax.set_title("Distribuição dos Custos de Reparo")
                        st.pyplot(fig)

                    # Tokens usados
                    with st.expander("📊 Uso de tokens desta chamada"):
                        st.write(f"- Tokens de entrada: `{usage.get('input_tokens', 0)}`")
                        st.write(f"- Tokens de saída: `{usage.get('output_tokens', 0)}`")
                        st.write(f"- Custo total: `${usage.get('cost_usd', 0):.6f}` USD")

            except json.JSONDecodeError as e:
                st.error(f"❌ Erro ao interpretar resposta do GPT-4o: {e}")
                status.update(label="Erro na análise", state="error")
            except Exception as e:
                st.error(f"❌ Erro na chamada da API: {e}")
                status.update(label="Erro na API", state="error")

else:
    st.info("📸 Faça o upload de uma foto do veículo para iniciar a inspeção.")
