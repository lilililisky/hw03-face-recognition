import streamlit as st
from PIL import Image, ImageDraw
from src.face_processor import detect_faces, recognize_faces

# 页面配置
st.set_page_config(page_title="人脸识别 Demo", page_icon="👤")

# 标题
st.title("👤 人脸识别 Demo")
st.markdown("基于 `face_recognition` + `Streamlit` 实现的人脸检测与识别")

# 上传/选择图片
st.subheader("1. 选择图片")
uploaded_file = st.file_uploader("上传图片（支持 jpg/png）", type=["jpg", "jpeg", "png"])
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", use_column_width=True)
else:
    st.info("请上传一张包含人脸的图片开始测试")

# 处理图片
if image is not None:
    st.subheader("2. 处理结果")
    mode = st.radio("选择处理模式", ["仅检测人脸", "识别人脸（需配置人脸库）"])
    
    if mode == "仅检测人脸":
        # 检测人脸并画框
        face_locations = detect_faces(image)
        st.write(f"✅ 检测到 {len(face_locations)} 张人脸")
        
        # 绘制红色人脸框
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        for (top, right, bottom, left) in face_locations:
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        st.image(img_draw, caption="检测结果（红色框标注人脸）", use_column_width=True)
    
    else:
        st.warning("⚠️ 识别模式需要提前配置「已知人脸库」")
        st.info("请在代码中添加你的已知人脸图片和姓名，示例：\n"
                "```python\n"
                "known_img = Image.open(\"your_photo.jpg\")\n"
                "known_encodings = [get_face_encodings(known_img)[0]]\n"
                "known_names = [\"你的名字\"]\n"
                "results = recognize_faces(image, known_encodings, known_names)\n"
                "```")
        # （可选）你可以提前准备自己的人脸库，取消注释测试
        # known_img = Image.open("known_face.jpg")
        # known_encodings = [get_face_encodings(known_img)[0]]
        # known_names = ["你的名字"]
        # results = recognize_faces(image, known_encodings, known_names)
        # # 绘制框+标签
        # img_draw = image.copy()
        # draw = ImageDraw.Draw(img_draw)
        # for name, (top, right, bottom, left) in results:
        #     draw.rectangle([left, top, right, bottom], outline="blue", width=3)
        #     draw.text((left, top-15), name, fill="blue")
        # st.image(img_draw, caption="识别结果", use_column_width=True)

# 运行说明
st.markdown("---")
st.subheader("运行命令")
st.code("""
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
streamlit run app.py

# 3. 访问 http://localhost:8501
""")