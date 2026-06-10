# A.I. AutoInspector - Streamlit Version

Este repositório contém uma adaptação do projeto **A.I. AutoInspector** para a plataforma **Streamlit**, permitindo o upload de imagens de carros e a detecção automática de danos com estimativa de custos.

## Como fazer o deploy no Streamlit Cloud

1.  **Crie um repositório no GitHub** e envie todos os arquivos desta pasta para lá.
2.  **Importante: Arquivo de Pesos (Model Weights)**:
    O arquivo `carDDModel/checkpoint.pth` é grande (~500MB). Você tem duas opções:
    *   **Git LFS**: Configure o Git LFS no seu repositório para gerenciar o arquivo `.pth`.
    *   **Download Externo**: Caso não queira usar LFS, você pode hospedar o arquivo em um serviço como Google Drive/S3 e modificar o `streamlit_app.py` para baixá-lo automaticamente se não existir localmente.

3.  **Configuração no Streamlit Cloud**:
    *   Conecte seu repositório GitHub ao [Streamlit Cloud](https://streamlit.io/cloud).
    *   O Streamlit detectará automaticamente o `requirements.txt` e o `packages.txt`.
    *   **Nota sobre MMDetection**: A biblioteca `mmdet` requer compilação. Se o deploy direto falhar, recomenda-se usar uma imagem Docker ou instalar via `pip install openmim && mim install mmdet` no início do script.

## Estrutura do Projeto

*   `streamlit_app.py`: Código principal da aplicação.
*   `requirements.txt`: Dependências Python.
*   `packages.txt`: Dependências do sistema (OpenCV, etc).
*   `carDDModel/`: Contém as configurações e pesos do modelo de detecção de danos.

## Funcionalidades

*   **Upload de Imagem**: Suporta JPG, JPEG e PNG.
*   **Detecção de Veículo**: Identifica e recorta o carro na imagem.
*   **Análise de Danos**: Identifica 6 tipos de danos (amassados, arranhões, rachaduras, vidro quebrado, lanterna quebrada, pneu furado).
*   **Estimativa de Custo**: Calcula o custo de reparo baseado no valor e idade do veículo.
