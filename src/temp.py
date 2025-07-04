from utils.Fourier_Module import Fourier2D
import cv2 as cv

input_img = r"D:\KSIP_Research\Latent\Latent Fingerprint Enhancement & Restoration\output\watershed_idea_segment\058L4U.bmp.bmp"


input_read = cv.imread(input_img)
gray = cv.cvtColor(input_read, cv.COLOR_BGR2GRAY)

raw_fft = Fourier2D(gray)
raw_fft.fft()
raw_magnitude = raw_fft.showMagnitude()