import cv2
import matplotlib.pyplot as plt

image = cv2.imread("test2.jpg")

#opencv reads colors as BGR so convert before showing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 4)) #figure dimensions

plt.subplot(1, 4, 1) #first slot
plt.imshow(image_rgb)
plt.title("Original") #header
plt.axis("off") #removes plot axis

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(1, 4, 2) #second slot
plt.imshow(gray, cmap="gray") #draws grayscale image, cmap="gray" means it will all be shades of gray
plt.title("Grayscale")
plt.axis("off")

#crop a peice
cropped = image[165:315, 245:395]

plt.subplot(1, 4, 3) #third slot
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)) #converts BGR to RGB, crops
plt.title("Cropped")
plt.axis("off")

#make a smaller version
small = cv2.resize(image, (224, 224))

plt.subplot(1, 4, 4) #fourth slot
plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)) #converts BGR to RBG, resized to 224x224
plt.title("Resized")
plt.axis("off")

plt.show() #opens window and displays everything

