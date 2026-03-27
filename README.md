# IC Prompt - Transformação de Imagem com Webcam e Stable Diffusion

Script que captura uma imagem da webcam e utiliza **Stable Diffusion img2img** para transformá-la com base em um prompt de texto. Tudo roda localmente na sua máquina.

## Pré-requisitos

- Ubuntu (testado no Ubuntu 24.04)
- Python 3.10+
- GPU NVIDIA com suporte a CUDA
- Webcam conectada ao computador

## Instalação

### 1. Criar o ambiente virtual Python

No Ubuntu, não é possível instalar pacotes Python diretamente no sistema. É necessário criar um ambiente virtual:

```bash
cd ~/Desktop/IC
python3 -m venv venv
```

### 2. Ativar o ambiente virtual

```bash
source venv/bin/activate
```

Você vai notar que o terminal agora mostra `(venv)` no início da linha, indicando que o ambiente está ativo.

### 3. Instalar as dependências Python

```bash
pip install opencv-python pillow diffusers transformers accelerate torch
```

Na primeira execução do script, o modelo Stable Diffusion (~5 GB) será baixado automaticamente.

## Como usar

### 1. Ative o ambiente virtual

```bash
cd ~/Desktop/IC
source venv/bin/activate
```

### 2. Execute o script

```bash
python ic_prompt/main.py "seu prompt aqui"
```

### Exemplos de uso

```bash
# Transformar em pintura a óleo
python ic_prompt/main.py "transform into an oil painting"

# Estilo aquarela
python ic_prompt/main.py "watercolor painting style"

# Transformar em desenho animado
python ic_prompt/main.py "cartoon style, vibrant colors"

# Estilo cyberpunk
python ic_prompt/main.py "cyberpunk style, neon lights, futuristic"
```

### Parâmetros internos

O script usa dois parâmetros que controlam a geração:

- **strength** (padrão: 0.75) — quanto a imagem original é alterada. Valores mais altos geram imagens mais diferentes da original (0 = sem mudança, 1 = ignora a original).
- **guidance_scale** (padrão: 7.5) — quanto o modelo segue o prompt. Valores mais altos seguem o prompt mais fielmente.

## Saída

O script gera dois arquivos:

- **input_frame.png** — a imagem original capturada pela webcam
- **output_YYYYMMDD_HHMMSS.png** — a imagem gerada pelo modelo

## Solução de problemas

| Problema | Solução |
|---|---|
| `Erro: nao foi possivel acessar a webcam` | Verifique se a webcam está conectada e se nenhum outro programa está usando ela |
| `externally-managed-environment` | Ative o ambiente virtual com `source venv/bin/activate` |
| `CUDA out of memory` | Feche outros programas que usam a GPU. O modelo precisa de pelo menos 4 GB de VRAM |
| Geração muito lenta | Verifique se o torch está usando CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |
