import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn tới mô hình
model_path = r'checkpoints\15_gen.pth.tar'

# Khởi tạo mô hình (giả sử bạn đã có lớp mô hình)
from VGGPytorch import VGGNet  # Thay đổi đường dẫn nếu cần

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGNet().to(device)
checkpoint = torch.load(model_path, weights_only=True, map_location=device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

# Tải ảnh
image_path = r'data\train\photo\2005-06-26 14_04_52.jpg'# Thay đổi đường dẫn
image = Image.open(image_path).convert('RGB')
# Lưu kích thước của ảnh gốc
original_size = image.size  # (width, height)
# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0).to(device)
print(image_tensor.shape)
# Dự đoán với mô hình
with torch.no_grad():
    cartoon_image_tensor = model(image_tensor)
    print(cartoon_image_tensor.shape)

# Chuyển đổi tensor thành numpy array
cartoon_image = cartoon_image_tensor.squeeze(0).cpu().detach().numpy()

# Đảm bảo kích thước ảnh đầu ra khớp với kích thước ảnh gốc
cartoon_image = np.transpose(cartoon_image, (1, 2, 0))  # Chuyển đổi về dạng (H, W, C)

# Thay đổi kích thước ảnh đầu ra để khớp với kích thước ảnh gốc
cartoon_image = np.clip(cartoon_image * 255, 0, 255).astype(np.uint8)  # Đảm bảo giá trị pixel trong khoảng [0, 255]
cartoon_image = Image.fromarray(cartoon_image)  # Chuyển đổi về dạng ảnh
cartoon_image = cartoon_image.resize(original_size, Image.BILINEAR)  # Thay đổi kích thước

# Hiển thị ảnh gốc và ảnh hoạt hình
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cartoon_image)
plt.title("Cartoonized Image")
plt.axis('off')

plt.show()