import streamlit as st
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
.info-box {
    background-color: #e8f4fd;
    border-left: 4px solid #007bff;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CHAVE DA API — Streamlit Secrets
# ─────────────────────────────────────────
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "❌ Chave da API OpenAI não encontrada. "
        "Configure `OPENAI_API_KEY` em **Settings → Secrets** no Streamlit Cloud."
    )
    st.stop()

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

# Faixas de custo de reparo em R$ no Brasil por tipo e severidade
# Baseado em valores médios praticados por oficinas no Brasil (2024-2025)
REPAIR_COST_BRL = {
    'dent': {
        'low':    (300,   800),
        'medium': (800,  2500),
        'high':   (2500, 6000)
    },
    'scratch': {
        'low':    (150,   400),
        'medium': (400,  1200),
        'high':   (1200, 3000)
    },
    'crack': {
        'low':    (200,   600),
        'medium': (600,  2000),
        'high':   (2000, 5000)
    },
    'glass_shatter': {
        'low':    (400,   800),
        'medium': (800,  2000),
        'high':   (2000, 4500)
    },
    'lamp_broken': {
        'low':    (300,   700),
        'medium': (700,  2000),
        'high':   (2000, 5000)
    },
    'tire_flat': {
        'low':    (200,   400),
        'medium': (400,   900),
        'high':   (900,  2000)
    }
}

# ─────────────────────────────────────────
# MODELO DE DETECÇÃO DE VEÍCULO
# ─────────────────────────────────────────
@st.cache_resource
def load_vehicle_detector():
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(DEVICE).eval()
    return model

# ─────────────────────────────────────────
# FUNÇÕES UTILITÁRIAS
# ─────────────────────────────────────────
def get_best_vehicle_box(det_output, threshold=0.5):
    """Retorna o bounding box do veículo com maior confiança."""
    vehicle_classes = [3, 4, 6, 8]  # car, motorcycle, bus, truck no COCO
    best_score = 0
    best_bbox  = None
    for i in range(len(det_output['boxes'])):
        label = det_output['labels'][i].item()
        score = det_output['scores'][i].item()
        if label in vehicle_classes and score > best_score and score > threshold:
            best_bbox  = det_output['boxes'][i]
            best_score = score
    return best_bbox, best_score

def adjust_bbox(bbox, img_width, img_height, margin=0.05):
    """Adiciona margem pequena ao bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    xm, ym = int(w * margin), int(h * margin)
    return [
        max(0, x1 - xm),
        max(0, y1 - ym),
        min(img_width,  x2 + xm),
        min(img_height, y2 + ym)
    ]

def bbox_is_too_large(bbox, img_width, img_height, max_ratio=0.85):
    """Verifica se o bbox ocupa quase a imagem inteira (detecção ruim)."""
    x1, y1, x2, y2 = bbox
    bbox_area  = (x2 - x1) * (y2 - y1)
    image_area = img_width * img_height
    return (bbox_area / image_area) > max_ratio

def draw_bbox_on_image(image, bbox):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(bbox, outline="#00FF00", width=4)
    return img_copy

def image_to_base64(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def calculate_repair_cost(damages: list) -> tuple:
    """
    Calcula custo de reparo em R$ usando faixas realistas do mercado brasileiro.
    Usa a média da faixa ponderada pela confiança do modelo.
    """
    total_cost = 0
    details    = []
    for d in damages:
        dtype    = d.get('type', 'scratch')
        score    = float(d.get('confidence', 0.5))
        severity = d.get('severity', 'medium')

        faixa   = REPAIR_COST_BRL.get(dtype, {}).get(severity, (500, 1500))
        # Custo = média da faixa, ponderada pela confiança
        custo_medio = (faixa[0] + faixa[1]) / 2
        cost    = custo_medio * score

        total_cost += cost
        details.append({
            'type':        dtype,
            'label':       DAMAGE_LABELS_PT.get(dtype, dtype.replace('_', ' ').title()),
            'score':       score,
            'severity':    severity,
            'location':    d.get('location', 'não especificado'),
            'description': d.get('description', ''),
            'cost':        cost,
            'faixa_min':   faixa[0],
            'faixa_max':   faixa[1],
        })
    return total_cost, details

# ─────────────────────────────────────────
# ANÁLISE GPT-4o VISION — PROMPT APRIMORADO
# ─────────────────────────────────────────
def analyze_damage_gpt4o(image: Image.Image) -> dict:
    client  = OpenAI(api_key=OPENAI_API_KEY)
    img_b64 = image_to_base64(image)

    prompt = """Você é um perito automotivo especializado em avaliação de danos para seguradoras no Brasil.

Analise esta imagem e faça o seguinte:

1. IDENTIFIQUE o veículo: marca, modelo, versão aproximada e ano estimado.
2. ESTIME o valor de mercado do veículo no Brasil em Reais (R$), com base na tabela FIPE atual.
3. IDENTIFIQUE TODOS os danos visíveis com máximo de detalhes.
4. IGNORE papéis, pessoas, mãos ou objetos em frente ao veículo — foque apenas nos danos na lataria, vidros, faróis e pneus.

Responda APENAS com JSON válido, sem texto adicional, neste formato exato:
{
  "vehicle_detected": true,
  "vehicle": {
    "brand": "Chevrolet",
    "model": "Spin",
    "version": "LTZ",
    "year_estimated": 2019,
    "fipe_value_brl": 75000,
    "fipe_reference": "Tabela FIPE estimada"
  },
  "damages": [
    {
      "type": "glass_shatter",
      "confidence": 0.97,
      "severity": "high",
      "location": "vidro traseiro",
      "description": "vidro traseiro completamente destruído, ausência total do vidro com fragmentos visíveis"
    }
  ],
  "overall_condition": "poor",
  "summary": "Chevrolet Spin com vidro traseiro completamente destruído. Dano grave que compromete a segurança do veículo."
}

Tipos de dano válidos: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
Severidade: low (superficial), medium (moderado), high (grave/profundo)
Condição geral: excellent, good, fair, poor, critical

Se não houver veículo visível, retorne vehicle_detected: false.
Se não houver danos visíveis, retorne damages como lista vazia."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{img_b64}",
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
        max_tokens=1200,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    # Remove blocos markdown ```json ... ``` se existirem
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif "{" in part:
                raw = part.strip()
                break

    result = json.loads(raw.strip())
    result['_usage'] = {
        'input_tokens':  response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'cost_usd':      (response.usage.prompt_tokens  * 2.50  / 1_000_000) +
                         (response.usage.completion_tokens * 10.00 / 1_000_000)
    }
    return result

# ─────────────────────────────────────────
# INTERFACE PRINCIPAL
# ─────────────────────────────────────────
st.title("🚗 A.I. AutoInspector")
st.subheader("Detecção Inteligente de Danos Veiculares — Powered by GPT-4o Vision")
st.success("🔐 API Key carregada com segurança via Streamlit Secrets")

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("🔬 Detecção de Veículo")
    use_vehicle_detection = st.toggle(
        "Recortar veículo com Faster R-CNN",
        value=True,
        help="Tenta isolar o veículo antes de enviar para o GPT-4o. "
             "Desative se a detecção estiver cortando o carro errado."
    )
    det_threshold = st.slider(
        "Confiança mínima da detecção",
        0.1, 0.9, 0.5, 0.05,
        help="Valores mais altos = menos falsos positivos no crop"
    )

    st.divider()
    st.caption(f"Dispositivo PyTorch: `{DEVICE}`")
    st.caption("Modelo de análise: `gpt-4o` (Vision)")
    st.caption("Custo estimado: 3 Centavos até 6 Centavos por análise")
    st.caption("Valores de reparo baseados no mercado brasileiro")

# ── UPLOAD ───────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Escolha uma imagem do carro...",
    type=["jpg", "jpeg", "png"],
    help="Foto do veículo danificado. Quanto mais clara e próxima, melhor a análise."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Imagem Original", use_container_width=True)

    if st.button("🔍 Analisar Danos com GPT-4o"):
        with st.status("Processando imagem...", expanded=True) as status:

            # ── ETAPA 1: Detecção e crop do veículo ──
            img_to_analyze = image
            crop_aplicado  = False

            if use_vehicle_detection:
                st.write("🚗 Detectando veículo com Faster R-CNN...")
                det_model  = load_vehicle_detector()
                transform  = T.Compose([T.ToTensor()])
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    det_output = det_model(img_tensor)[0]

                best_bbox, confidence = get_best_vehicle_box(
                    det_output, threshold=det_threshold
                )

                if best_bbox is not None:
                    bbox_np  = best_bbox.detach().cpu().numpy().astype(int)
                    adj_bbox = adjust_bbox(bbox_np, image.size[0], image.size[1])

                    # Verifica se o bbox é muito grande (detecção ruim)
                    if bbox_is_too_large(adj_bbox, image.size[0], image.size[1]):
                        st.warning(
                            "⚠️ O crop detectado cobre quase a imagem inteira — "
                            "usando imagem completa para preservar contexto dos danos."
                        )
                        with col2:
                            st.image(image, caption="Imagem completa (crop ignorado)", use_container_width=True)
                    else:
                        cropped_img    = image.crop(adj_bbox)
                        annotated      = draw_bbox_on_image(image, adj_bbox)
                        img_to_analyze = cropped_img
                        crop_aplicado  = True
                        with col2:
                            st.image(
                                annotated,
                                caption=f"Veículo Detectado (conf: {confidence:.0%})",
                                use_container_width=True
                            )
                else:
                    st.warning(
                        "⚠️ Nenhum veículo detectado pelo Faster R-CNN. "
                        "Enviando imagem completa para o GPT-4o."
                    )
                    with col2:
                        st.image(image, caption="Imagem completa", use_container_width=True)
            else:
                with col2:
                    st.image(image, caption="Imagem completa (crop desativado)", use_container_width=True)

            # ── ETAPA 2: Análise GPT-4o ──
            st.write("🤖 Identificando veículo e analisando danos com GPT-4o...")
            try:
                gpt_result           = analyze_damage_gpt4o(img_to_analyze)

                if not gpt_result.get('vehicle_detected', True):
                    st.error("❌ O GPT-4o não identificou um veículo na imagem.")
                    status.update(label="Falha na análise", state="error")
                    st.stop()

                vehicle              = gpt_result.get('vehicle', {})
                damages              = gpt_result.get('damages', [])
                repair_cost, details = calculate_repair_cost(damages)
                usage                = gpt_result.get('_usage', {})
                fipe_value           = vehicle.get('fipe_value_brl', 0)

                status.update(label="✅ Análise concluída!", state="complete")

                # ── ETAPA 3: Resultados ──
                st.divider()

                # Cabeçalho com dados do veículo identificado
                veiculo_str = (
                    f"{vehicle.get('brand','?')} {vehicle.get('model','?')} "
                    f"{vehicle.get('version','')} ({vehicle.get('year_estimated','?')})"
                ).strip()

                st.subheader(f"📋 Relatório — {veiculo_str}")

                # Box com dados do veículo
                st.markdown(f"""
                <div class="info-box">
                    🚘 <strong>Veículo Identificado:</strong> {veiculo_str}<br>
                    💰 <strong>Valor de Mercado (FIPE est.):</strong>
                        R$ {fipe_value:,.0f}<br>
                    📌 <strong>{vehicle.get('fipe_reference','')}</strong>
                </div>
                """, unsafe_allow_html=True)

                # Resumo do GPT
                if gpt_result.get('summary'):
                    st.info(f"💬 {gpt_result['summary']}")

                # Métricas principais
                custo_api_brl = usage.get('cost_usd', 0) * 5.0  # USD → BRL aprox
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 Custo Total de Reparo",  f"R$ {repair_cost:,.2f}")
                m2.metric("🔢 Danos Encontrados",      len(details))
                m3.metric("📊 Condição Geral",         gpt_result.get('overall_condition','N/A').upper())
                m4.metric("💵 Custo da Chamada API",   f"R$ {custo_api_brl:.4f}")

                st.divider()
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    if details:
                        st.write("#### 🔍 Detalhes dos Danos")
                        severity_colors = {
                            'low':    '#28a745',
                            'medium': '#ffc107',
                            'high':   '#dc3545'
                        }
                        severity_labels = {
                            'low':    'Leve',
                            'medium': 'Moderado',
                            'high':   'Grave'
                        }
                        for d in details:
                            color = severity_colors.get(d['severity'], '#007bff')
                            sev   = severity_labels.get(d['severity'], d['severity'])
                            st.markdown(f"""
                            <div class="damage-card" style="border-left-color:{color};">
                                <strong>{d['label']}</strong>
                                &nbsp;<span style="color:{color};font-weight:bold;">[{sev}]</span><br>
                                📍 <em>{d['location']}</em><br>
                                📝 {d['description']}<br>
                                🎯 Confiança: {d['score']:.0%}
                                &nbsp;|&nbsp;
                                💰 Faixa: R$ {d['faixa_min']:,.0f} – R$ {d['faixa_max']:,.0f}
                                &nbsp;|&nbsp;
                                <strong>Est.: R$ {d['cost']:,.2f}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Nenhum dano identificado!")

                with res_col2:
                    if details:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        labels = [d['label'] for d in details]
                        costs  = [d['cost']  for d in details]
                        colors = [
                            '#dc3545','#ffc107','#007bff',
                            '#28a745','#6f42c1','#fd7e14'
                        ]
                        ax.pie(
                            costs,
                            labels=labels,
                            autopct='%1.1f%%',
                            startangle=140,
                            colors=colors[:len(details)]
                        )
                        ax.set_title("Distribuição dos Custos (R$)")
                        st.pyplot(fig)

                    # Tokens e custo da chamada
                    with st.expander("📊 Detalhes do consumo da API"):
                        st.write(f"- Tokens de entrada: `{usage.get('input_tokens', 0)}`")
                        st.write(f"- Tokens de saída: `{usage.get('output_tokens', 0)}`")
                        st.write(f"- Custo: `${usage.get('cost_usd', 0):.6f}` USD "
                                 f"≈ `R$ {custo_api_brl:.4f}`")

            except json.JSONDecodeError as e:
                st.error(f"❌ Erro ao interpretar resposta do GPT-4o. Tente novamente. ({e})")
                status.update(label="Erro na análise", state="error")
            except Exception as e:
                st.error(f"❌ Erro na chamada da API: {e}")
                status.update(label="Erro na API", state="error")

else:
    st.info("📸 Faça o upload de uma foto do veículo para iniciar a inspeção.")
