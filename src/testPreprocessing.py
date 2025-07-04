import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.Morphological_Module import fillHoles
from utils.Normalize_Module import normalize, adjustRange
from scipy.signal import convolve2d, windows

def applied_gaussian(mask, kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize the kernel

    filter = convolve2d(mask, kernel, mode="same")
    return filter

input_path = r"D:\KSIP_Research\Latent\Database\NIST27\LatentRename\069L7U.bmp"

# โหลดภาพ grayscale
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

h, w = img.shape

# 2. Threshold เพื่อหา pixel ดำ
_, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

temp = binary.copy()

# สร้าง mask ที่เป็นสี่เหลี่ยมกลางภาพ
h, w = binary.shape
mask_center = np.zeros_like(binary, dtype=np.uint8)

# กำหนดขอบเขต "กลางภาพ" ที่จะเอาออก
x1 = int(w * 0.15)
x2 = w - x1
y1 = int(h * 0.25)

# mask_center[y1:, x1:x2] = 255  # pixel ในช่วงนี้จะถูก "ลบออก"
# วาดวงรีแนวตั้ง
center = (w // 2, h // 2)   # จุดศูนย์กลาง
axes = (200, 300)                    # แนวนอนสั้น, แนวตั้งยาว → แนวตั้ง
angle = 0                            # ไม่หมุน
start_angle = 0
end_angle = 360
color = 255
thickness = -1                       # เติมเต็มวงรี

cv2.ellipse(mask_center, center, axes, angle, start_angle, end_angle, color, thickness)

binary[mask_center == 255] = 0

kernel_size = 11
sigma = 1.5



# 5. ลบด้วย inpainting
# result = cv2.inpaint(img, mask, 30, cv2.INPAINT_TELEA)
mean_value = np.mean(img[binary == 0])
result = img.copy()
result[binary == 255] = mean_value

# result = applied_gaussian(result, kernel_size=kernel_size, sigma=sigma)

# 6. แสดงผล
plt.figure(figsize=(15,5))
plt.subplot(1,4,1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1,4,2); plt.imshow(temp, cmap='gray'); plt.title('Thresholded')
plt.subplot(1,4,3); plt.imshow(mask_center, cmap='gray'); plt.title('mask_center')
plt.subplot(1,4,4); plt.imshow(result, cmap='gray'); plt.title('After Inpainting')
plt.tight_layout(); plt.show()