import cv2
from PIL import Image
img_cv2=cv2.imread("comsen.jpg")
if img_cv2 is None:
    print("失败")
    exit()

cv2.imshow("comsen",img_cv2)

b_channel,g_channel,r_channel=cv2.split(img_cv2)
cv2.imshow("B通道",b_channel)
cv2.imshow("G通道",g_channel)
cv2.imshow("R通道",r_channel)

img_pil=Image.open("comsen.jpg")
r_pil,g_pil,b_pil=img_pil.split()
img_pil.show(title="Pillow-原图")
r_pil.show(title="Pillow-R通道")
g_pil.show(title="Pillow-G通道")
b_pil.show(title="Pillow-B通道")

# OpenCV保存
cv2.imwrite("task1_comsen.jpg", img_cv2)
cv2.imwrite("task1_opencv_B_channel.jpg", b_channel)
cv2.imwrite("task1_opencv_G_channel.jpg", g_channel)
cv2.imwrite("task1_opencv_R_channel.jpg", r_channel)

# Pillow保存
img_pil.save("task1_Pillow原图.jpg")
r_pil.save("task1_Pillow_R通道.jpg")
g_pil.save("task1_Pillow_G通道.jpg")
b_pil.save("task1_Pillow_B通道.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("第一题完成")

