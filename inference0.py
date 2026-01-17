import os
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


model = models.mobilenet_v2(pretrained=False)  
model.classifier[1] = torch.nn.Linear(model.last_channel, 4)
model = model.to(device)


model.load_state_dict(torch.load("model0.pth", map_location=device))
model.eval()


class_names = ["overripe", "ripe", "rotten", "unripe"]


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_test(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, dim=1).item()

    return class_names[pred]


test_folder = "test"
test_images = sorted(os.listdir(test_folder))

results = []

for img_name in test_images:
    img_path = os.path.join(test_folder, img_name)
    pred = predict_image(img_path)
    results.append([img_name, pred])
    print(img_name, "=>", pred)


df = pd.DataFrame(results, columns=["ID", "Ripeness"])
df.to_csv("submission.csv", index=False)

print("\n🎉 submission.csv 生成成功！")
