import cv2
import numpy as np
import base64
# from skimage.transform import radon, rotate
# import matplotlib.pyplot as plt
import KNN.knn as knn

# 是否为车牌
def get_plate_type(plate):
    # 检测蓝色区域
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # 检测绿色区域
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 膨胀 mask
    kernel = np.ones((2, 2), np.uint8)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)

    # 获取 mask 的轮廓
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选轮廓 裁剪 plate
    for contour in contours_blue:
        area = cv2.contourArea(contour)
        if area > 1000:
            return 'blue'
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 1000:
            return 'green'

    return 'none'

    
# 处理车牌检测
def process_plate(plate, type):

    mask = None

    if type == 'blue':
    # 检测蓝色区域
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    elif type == 'green':
    # 检测绿色区域
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

    # 膨胀 mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
   
    # 获取 mask 的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选轮廓 裁剪 plate
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            plate = plate[y:y+h, x:x+w]
            break

    # 转换为灰度图像
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 膨胀
    kernel = np.ones((2, 2), np.uint8)
    dilate = cv2.dilate(binary, kernel, iterations=1)
    # 腐蚀
    erode = cv2.erode(dilate, kernel, iterations=1)


    # 处理上边缘
    index = 0
    for row in erode:
        # 计算图像每一行的黑白跳变次数
        count = 0
        for i in range(len(row) - 1):
            if row[i] != row[i + 1]:
                count += 1
        if count > 14:
            break
        index += 1
    # 填充上边缘
    for i in range(index):
        erode[i] = 0

    # 处理下边缘
    index = len(erode) - 1
    for row in reversed(erode):
        # 计算图像每一行的黑白跳变次数
        count = 0
        for i in range(len(row) - 1):
            if row[i] != row[i + 1]:
                count += 1
        if count > 12:
            break
        index -= 1
    # 填充下边缘
    for i in range(index, len(erode)):
        erode[i] = 0

    try:
        # 水平方向投影（沿行求和）
        horizontal_projection = np.sum(erode, axis=1)
        # 找到非零区域（去除上下水平边框）
        threshold = np.max(horizontal_projection) * 0.1  # 设定投影的阈值
        non_zero_rows = np.where(horizontal_projection > threshold)[0]
        # 裁剪图像，去除上下边框
        top, bottom = non_zero_rows[0], non_zero_rows[-1]
        cropped = erode[top:bottom + 1, :]
    except:
        return None

    # 垂直方向投影（沿列求和）
    vertical_projection = np.sum(cropped, axis=0)
    # 找到非零区域（去除左右垂直边框）
    threshold = np.max(vertical_projection) * 0.1  # 设定投影的阈值
    non_zero_cols = np.where(vertical_projection > threshold)[0]
    # 裁剪图像，去除左右边框
    left, right = non_zero_cols[0], non_zero_cols[-1]
    cropped = cropped[:, left:right + 1]

    # 左右增加边框
    cropped = np.pad(cropped, ((0, 0), (10, 10)), 'constant', constant_values=0)

    #重新求投影
    vertical_projection = np.sum(cropped, axis=0)

    # 显示垂直方向投影
    # print(vertical_projection)
    # plt.plot(vertical_projection)
    # plt.show()


    # 车牌字符分割
    cut_points = []
    char_images = []
    for i in range(len(vertical_projection) - 1):
        if vertical_projection[i] == 0 and vertical_projection[i + 1] != 0:
            cut_points.append(i)
        if vertical_projection[i] != 0 and vertical_projection[i + 1] == 0:
            cut_points.append(i)

    for i in range(0, len(cut_points), 2):
        start = cut_points[i]
        end = cut_points[i + 1]
        char_width = end - start
        if char_width < 10:
            continue
        char_image = cropped[:, start:end]
        #char_image = np.pad(char_image, ((5, 5), (5, 5)), 'constant', constant_values=0)
        char_images.append(char_image)

    # 判断字符情况
    is_valid = True
    if len(char_images) != 7:
        is_valid = False
        char_images = []
        
    if is_valid:
        # 显示裁剪后的车牌
        # for i in range(len(char_images)):
        #     cv2.imshow('char' + str(i), char_images[i])

        plate_str = ''
        # 预测字符
        for i in range(len(char_images)):
            predicted_label = knn.predict_character(char_images[i], i)
            plate_str += str(predicted_label)   

        # 显示车牌 test
        # print('plate:', plate_str)
        # cv2.imshow('res', cropped)
        # cv2.waitKey(0)

        return (cropped, char_images, plate_str)
    else:
        return None

def get_plate(video_path):
    # 加载Haar级联分类器进行车牌检测（OpenCV内置分类器）
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # 加载视频文件
    #video_path = 'backend/carvideo.avi'
    cap = cv2.VideoCapture(video_path)

    # 确保视频打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    # 结果列表  
    result = []

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # 视频播放完毕

        # 转换为灰度图像，因为 Haar Cascade 需要灰度图像来进行检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Haar级联分类器检测车牌
        plates = plate_cascade.detectMultiScale(gray, 1.1, 10)

        new_frame = frame.copy()

        # 筛选车牌
        for (x, y, w, h) in plates:
            # 裁剪车牌
            plate_img = frame[y:y+h, x:x+w]
            # 判断是否为车牌
            plate_type = get_plate_type(plate_img)
            if plate_type != 'none':
                # 画出车牌
                cv2.rectangle(new_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # 处理车牌
                res = process_plate(plate_img, plate_type)
                if res is not None:
                    new_plate_img, char_images, plate_str = res
                    result.append({
                        'frame': new_frame,
                        'plate': plate_str,
                        'plate_img': new_plate_img,
                        'char_imgs': char_images
                    })

        # 显示视频（可选）
        #cv2.imshow('Video', new_frame)
        #cv2.waitKey(1)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 选择 plate_str 重复度最高的结果
    plate_dict = {}
    for item in result:
        if item['plate'] not in plate_dict:
            plate_dict[item['plate']] = 1
        else:
            plate_dict[item['plate']] += 1

    if len(plate_dict) == 0:
        print('No plate detected.')
        return None

    max_plate = max(plate_dict, key=plate_dict.get)

    output = None
    # 保存结果
    for item in result:
        if item['plate'] == max_plate:
            # cv2.imwrite('backend/result.jpg', item['frame'])
            # cv2.imwrite('backend/plate.jpg', item['plate_img'])
            print('plate:', item['plate'])
            output = item
            break

    prefix = 'data:image/jpg;base64,'
    output['frame'] = prefix + base64.b64encode(cv2.imencode('.jpg', output['frame'])[1].tobytes()).decode('utf-8')
    output['plate_img'] = prefix + base64.b64encode(cv2.imencode('.jpg', output['plate_img'])[1].tobytes()).decode('utf-8')
    for i in range(len(output['char_imgs'])):
        output['char_imgs'][i] = prefix + base64.b64encode(cv2.imencode('.jpg', output['char_imgs'][i])[1].tobytes()).decode('utf-8')

    return output