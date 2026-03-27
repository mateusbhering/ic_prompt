import sys
import time
import cv2
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionImg2ImgPipeline
import torch


def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel acessar a webcam.")
        sys.exit(1)

    # Descarta alguns frames para a câmera ajustar exposição
    for _ in range(30):
        cap.read()
    time.sleep(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erro: nao foi possivel capturar frame.")
        sys.exit(1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def generate_image(image: Image.Image, prompt: str, output_path: str):
    """Transforma uma imagem usando Stable Diffusion img2img"""

    print("Carregando modelo Stable Diffusion...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    # Redimensiona para resolução compatível com o modelo
    image = image.resize((512, 512))

    print(f"Gerando imagem com prompt: {prompt}")
    negative_prompt = "blurry, distorted, deformed, low quality, artifacts, ugly"
    result = pipe(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        strength=0.55,
        guidance_scale=7.5,
        num_inference_steps=50,
    )
    output_image = result.images[0]

    output_image.save(output_path)
    print(f"Imagem salva em: {output_path}")


def main():
    if len(sys.argv) < 2:
        print('Uso: python main.py "<seu prompt aqui>"')
        print('Exemplo: python main.py "transform into a watercolor painting"')
        sys.exit(1)

    prompt = sys.argv[1]

    print("Capturando frame da webcam...")
    frame = capture_frame()
    frame.save("input_frame.png")
    print("Frame salvo em input_frame.png")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.png"

    generate_image(frame, prompt, output_path)


if __name__ == "__main__":
    main()
