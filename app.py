from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# 모델 로드
model = load_model('model_3.h5')
nonSkinModel = load_model('skin.h5')

from tensorflow.keras.preprocessing.image import load_img, img_to_array




# 이미지 전처리 함수
def prepare_image(file, target_size=(224, 224)):
    
    # 이미지를 배열로 변환
    #img_array = img_to_array(img)
    # FileStorage 객체를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(file.read()))  # FileStorage를 BytesIO로 변환
    # 이미지를 RGB로 불러오기 (알파 채널 제거)
    image = image.convert("RGB")  # 'rgba' 대신 'rgb'로 설정
    image = image.resize(target_size)  # 이미지 크기 조정
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # 모델 학습 시 정규화 했을 경우
    return image

# 메인 페이지 (이미지 업로드)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == '':
            return "No selected file"
        
        if file:
            # prepare_image 함수 호출 시 FileStorage 객체 전달
            image = prepare_image(file)
            prediction = nonSkinModel.predict(image)[0][0]
            print("피부?")
            print(prediction)
            if prediction >= 0.5:
                prediction = model.predict(image)[0][0]
                print(prediction)
                result = "암" if prediction >= 0.5 else "아님"
                return render_template("result.html", result=result)
            else:
                result = "피부 사진을 올려주세요"
                return render_template("result.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


image = prepare_image(file)
            