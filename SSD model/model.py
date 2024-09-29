import torch
import torchvision
import cv2
from torchvision import transforms
import torchvision.models as models
import os


def transfrom_img_SSD(image):
  transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((300, 300)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.482, 0.459, 0.408], std=[0.492, 0.492, 0.492])
  ])
  return transform(image).unsqueeze(0)


model_SSD = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)
model_path = './SSD.pt'

model_SSD.load_state_dict(torch.load(model_path, map_location='cpu'))
model_SSD.eval()

test_folder_path = './IVUF-100.v1i.tensorflow/test'
df = './IVUF-100.v1i.tensorflow/test/_annotations.csv'



def show_predictions(image, predictions, threshold, img_id):
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    image = cv2.resize(image, (300, 300))

    for box, label, score in zip(boxes, labels, scores):

        if score >= threshold:
          x1, y1, x2, y2 = box.astype(int)
          label_text = "Lumen" if label == 1 else "No Lumen"
          cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(image, f"{label_text} conf: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = f"./output/{img_id}.jpg"
    cv2.imwrite(output_path, image)
    
    
img_id = 1

for filename in os.listdir(test_folder_path):
  if filename.endswith('.jpg'):
    image_path = os.path.join(test_folder_path, filename)
    image_rgb = cv2.imread(image_path)
    image = transfrom_img_SSD(image_rgb)

    with torch.no_grad():
      predictions = model_SSD(image)   
    
    show_predictions(image_rgb, predictions,1,img_id)
    img_id += 1