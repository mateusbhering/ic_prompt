import os
import sys
import cv2
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from ollama import chat
from ollama import ChatResponse

def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel acessar a webcam.")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erro: nao foi possivel capturar frame.")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def analyze_image(frame: Image.Image, prompt: str) -> str:
    """Analisa imagem usando Ollama + Llava localmente"""
    
    # Converte imagem para base64
    buffered = BytesIO()
    frame.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    try:
        print("Processando com modelo local (ollama)...")
        
        # Envia para ollama
        response: ChatResponse = chat(
            model='llava:7b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_base64]
                }
            ],
        )
        
        result = response.message.content
        return result
        
    except Exception as e:
        print(f"Erro ao conectar com Ollama: {e}")
        print("Certifique-se que Ollama está rodando: ollama serve")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print('Uso: python main.py "<seu prompt aqui>"')
        print('Exemplo: python main.py "Descreva essa pessoa em detalhes"')
        sys.exit(1)

    prompt = sys.argv[1]

    print("Capturando frame da webcam...")
    frame = capture_frame()
    frame.save("input_frame.png")
    print("Frame salvo em input_frame.png")

    print(f"Enviando para Ollama com prompt: {prompt}")
    result = analyze_image(frame, prompt)

    if result:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{timestamp}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\nResposta salva em {output_path}")
        print(f"\n{'='*50}")
        print("RESPOSTA:")
        print(f"{'='*50}")
        print(result)
        print(f"{'='*50}")


if __name__ == "__main__":
    main()