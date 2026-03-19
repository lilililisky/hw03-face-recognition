import face_recognition
import numpy as np
from PIL import Image

def detect_faces(image: Image.Image) -> list:
    """
    检测图片中的人脸位置
    返回：[(top, right, bottom, left), ...] 人脸框坐标
    """
    img_rgb = np.array(image.convert("RGB"))
    face_locations = face_recognition.face_locations(img_rgb)
    return face_locations

def get_face_encodings(image: Image.Image) -> list:
    """
    提取所有人脸的 128 维特征编码
    返回：[encoding1, encoding2, ...]
    """
    img_rgb = np.array(image.convert("RGB"))
    face_encodings = face_recognition.face_encodings(img_rgb)
    return face_encodings

def recognize_faces(image: Image.Image, known_encodings: list, known_names: list) -> list:
    """
    识别人脸（对比已知人脸库）
    返回：[(name, (top, right, bottom, left)), ...]
    """
    img_rgb = np.array(image.convert("RGB"))
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
    
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        # 取最匹配的结果
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = known_names[best_match_idx]
        results.append((name, (top, right, bottom, left)))
    return results