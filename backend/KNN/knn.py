import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sys

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
    
# 加载数据
def load_data(data_dir):
    images = []
    labels = []
    
    # 遍历 model 文件夹中的每个子文件夹（每个子文件夹代表一个字符标签）
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                
                # 读取图像，并将其转换为固定大小的 20x20 图像（假设字符图像大小一致）
                img = None
                if sys.platform == 'win32':
                    img = cv_imread(img_path)
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # img_resized = cv2.resize(img, (20, 20))  # 重设大小为 20x20
                
                # 二值化处理（如果数据已经是二值化的，可以省略这一步）
                # _, img_bin = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
                
                # 将图像展平为一维向量
                img_flattened = img.flatten()
                
                images.append(img_flattened)
                labels.append(label)
    
    # 转换为 numpy 数组
    X = np.array(images)
    y = np.array(labels)
    
    return X, y


def train_knn(type):
    # 加载训练数据
    data_dir = os.path.join(os.path.dirname(__file__), 'data', type)  # 训练数据存放的路径
    X, y = load_data(data_dir)

    # 3. 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 初始化并训练 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 5. 测试模型
    y_pred = knn.predict(X_test)

    # 6. 计算准确率
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy({type}): {accuracy * 100:.2f}%')

    # 7. 可视化部分预测结果
    # for i in range(5):
    #     plt.subplot(1, 5, i+1)
    #     plt.imshow(X_test[i].reshape(20, 20), cmap='gray')  # 重新将一维向量恢复为 20x20 的图像
    #     plt.title(f'Pred: {y_pred[i]}')
    #     plt.axis('off')
    # plt.show()

    return knn

# 训练 KNN 模型
knn_province = train_knn('province')
knn_num = train_knn('num')

# 对单个字符进行预测
def predict_character(img, index):
    # 加载 KNN 模型
    knn = knn_province if index == 0 else knn_num

    # 读取并处理图像
    # img = cv2.imread(character_img_path, cv2.IMREAD_GRAYSCALE)
    #img_resized = cv2.resize(img, (20, 20))

    # 获取图像的尺寸
    h, w = img.shape
    # 计算缩放比例
    scale = min(20 / h, 20 / w)
    # 计算缩放后的尺寸
    new_h = int(h * scale)
    new_w = int(w * scale)
    # 等比缩放图像
    img_resized = cv2.resize(img, (new_w, new_h))
    # 创建一个 20x20 的黑色图像
    output_img = np.zeros((20, 20), dtype=np.uint8)
    # 计算将缩放后的图像放置到 20x20 图像中的位置
    top_left_x = (20 - new_w) // 2
    top_left_y = (20 - new_h) // 2
    # 将缩放后的图像放置到新的图像上
    output_img[top_left_y:top_left_y+new_h, top_left_x:top_left_x+new_w] = img_resized


    # 二值化处理
    #_, img_bin = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
    
    # 将图像展平
    img_flattened = output_img.flatten().reshape(1, -1)
    
    # 使用 KNN 模型进行预测
    predicted_label = knn.predict(img_flattened)
    return predicted_label[0]

# 测试预测函数
# test_img_path = '/Users/chrisliu/bjfu/2024 DV/课程设计/code/backend/KNN/data/0/4-3.jpg'  # 测试图像路径
# predicted_character = predict_character(test_img_path)
# print(f'Predicted character: {predicted_character}')
