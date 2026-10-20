import cv2
import numpy as np

img = cv2.imread("example1.jpg")
if img is None:
    print("失败")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny边缘检测
edges = cv2.Canny(blur, 50, 150)

# 形成闭合轮廓
kernel = np.ones((3, 3), np.uint8)
edges_closed = cv2.dilate(edges, kernel, iterations=2)

#轮廓
contours,hierarchy=cv2.findContours(edges_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 显示预处理结果
cv2.imshow("原图", img)
cv2.imshow("二值化图", edges_closed)
# 去掉噪点
if contours:
    contours = [contours[0]]

img_contour=img.copy()
cv2.drawContours(img_contour,contours,-1,(0,255,0),2)
cv2.imshow("轮廓绘制",img_contour)

#周长面积
total_area = cv2.contourArea(contours[0])
total_perimeter = cv2.arcLength(contours[0], True)

print(f"有效轮廓数：{len(contours)}")
print(f"总面积：{total_area:.2f} 像素²")
print(f"总周长：{total_perimeter:.2f} 像素")

#保存
cv2.imwrite("task3_final_contour.jpg", img_contour)


cv2.waitKey(0)
cv2.destroyAllWindows()
