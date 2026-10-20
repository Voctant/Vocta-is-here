import cv2
import numpy as np

#原图
img_cv2=cv2.imread("comsen.jpg")
if img_cv2 is None:
    print("失败")
    exit()

cv2.imshow("comsen", img_cv2)

#翻转
img_flip_1=cv2.flip(img_cv2,flipCode=1)
cv2.imshow("左右翻转",img_flip_1)
img_flip_2=cv2.flip(img_cv2,flipCode=0)
cv2.imshow("上下翻转",img_flip_2)

#放缩
img_small=cv2.resize(img_cv2,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
cv2.imshow("缩小一半",img_small)
img_big=cv2.resize(img_cv2,dsize=None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
cv2.imshow("放大一倍",img_big)

#旋转
img_90=cv2.rotate(img_cv2,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("顺时针90度",img_90)
img_180=cv2.rotate(img_cv2,cv2.ROTATE_180)
cv2.imshow("顺时针180度",img_180)

#平移
h,w=img_cv2.shape[:2]
center=(w//2,h//2)
tx=100
ty=50
M_move=np.float32([[1,0,tx],[0,1,ty]])
img_move=cv2.warpAffine(img_cv2,M_move,dsize=(w,h))
cv2.imshow("向右下平移",img_move)

#保存图片
cv2.imwrite("task2_origin.jpg", img_cv2)
cv2.imwrite("task2_flip_horizontal.jpg", img_flip_1)
cv2.imwrite("task2_flip_vertical.jpg", img_flip_2)
cv2.imwrite("task2_resize_half.jpg", img_small)
cv2.imwrite("task2_resize_double.jpg", img_big)
cv2.imwrite("task2_rotate_90.jpg", img_90)
cv2.imwrite("task2_rotate_180.jpg", img_180)
cv2.imwrite("task2_translate.jpg", img_move)

cv2.waitKey(0)
cv2.destroyAllWindows()