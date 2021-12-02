import os

emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprised',  # 惊讶
    '6': 'normal',  # 中性
}


def labelFiles(path):
    """
    为这个路径下的文件打上标记
    :param path: 
    :return: 
    """
    # 扫文件夹.根据文件名分类
    for dir in os.listdir(path):
        dirPath = os.path.join(path, dir)
        for file in os.listdir(dirPath):
            filePath = os.path.join(dirPath, file)
            # 打上标记
            os.rename(filePath, filePath + ' ' + emotions[dir])   
