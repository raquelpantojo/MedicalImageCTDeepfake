import gradio as gr
from predict import predict

def gradio_predict(image: gr.Image):
    # Convert Gradio image to PIL Image
    img_pil = image.convert("RGB")
    
    # Get predictions
    img_with_boxes, status = predict(img_pil)
    
    # Convert PIL image to bytes for Gradio
    return img_with_boxes, status

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox(label="Detection Status")],
    title="Detecção de deepfake de imagens de CT ",
    description=" Envie sua imagem de CT para detectar deepfake. Uma caixa delimitadora será desenhada apenas  se um deepfake for detectado, e uma mensagem de status indicará se um deepfake foi encontrado."
)

if __name__ == "__main__":
    iface.launch(debug=True)
