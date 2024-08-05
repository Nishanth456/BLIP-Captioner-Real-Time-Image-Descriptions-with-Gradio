import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image_path = "C:/Users/nisha/OneDrive/Pictures/Wallpaper/lion.jpg"
image = Image.open(image_path)

# Prepare the image for unconditional captioning
inputs_unconditional = processor(images=image, return_tensors="pt")

# Generate an unconditional caption
outputs_unconditional = model.generate(**inputs_unconditional, max_new_tokens=50)
caption_unconditional = processor.decode(outputs_unconditional[0], skip_special_tokens=True)
print("Unconditional Caption:", caption_unconditional)

# Prepare the image and prompt for conditional captioning
prompt = "a photography of"
inputs_conditional = processor(images=image, text=prompt, return_tensors="pt")

# Generate a conditional caption
outputs_conditional = model.generate(**inputs_conditional, max_new_tokens=50)
caption_conditional = processor.decode(outputs_conditional[0], skip_special_tokens=True)
print("Conditional Caption:", caption_conditional)