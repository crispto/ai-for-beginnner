import mediapipe as mp
import cv2


def main():
    pic = cv2.imread('cup-test.jpg')
    # 创建一个 mp 目标检测示例
    mp_drawing = mp.solutions.drawing_utils

    # 将检测结果绘制到图像上  



if __name__ == '__main__':
    main()