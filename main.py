from PIL import Image
import torch
from torchvision import transforms
from model_definition import CatDogModel

model = CatDogModel()
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=torch.device('cpu')))
model.eval()

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return "cat" if predicted.item() == 0 else "dog"

# Example usage
print(predict("test/cat_or_dog.jpg"))