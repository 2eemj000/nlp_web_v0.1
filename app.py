import joblib 
import os
import re
from flask import Flask, render_template, request # flask 웹서버
from konlpy.tag import Okt

app = Flask(__name__)
app.debug = True

okt = Okt()

tfidf_vector = None
model_lr = None

# okt, def tw_tokenizer(text) 같이 가져와야함

def tw_tokenizer(text):
    tokenizer_ko = okt.morphs(text)
    return tokenizer_ko

def load_lr(): # 순서 중요 !
    global tfidf_vector, model_lr
    tfidf_vector = joblib.load(os.path.join(app.root_path, "model/tfidf_vect.pkl"))
    model_lr = joblib.load(os.path.join(app.root_path, "model/lr.pkl"))
    # 운영체제 기준 app.py 파일이 위치한 디렉토리를 의미
    # app.root_path와 "model/lr.pkl"을 결합
    
# 당장은 필요없어도 모든 과정 설계해야함 (숫자 삭제)
def lt_transform(review):
    review = re.sub(r"\d+", " ", review)
    tfidf_matrix = tfidf_vector.transform([review])
    return tfidf_matrix

@app.route("/predict", methods=["GET", "POST"]) # 컨텍스트가 전체시스템의 입출력 관리를 해줌 (어노테이션 등을 통해)
def npl_predict():
    if request.method == "GET": # get일때
        return render_template("predict.html")
    else : # post일때 (게시글 등록하면)
        # 내부적으로만 사용하는 변수명은 _을 달아줌
        _review = request.form["review"] # predict.html에서 name 지정해두기
        _review_matrix = lt_transform(_review)
        _review_result = model_lr.predict(_review_matrix)[0]
        _predict_result="긍정" if _review_result else "부정"
        result={
            "review":_review,
            "review_result":_predict_result
        }
        return render_template("predict_result.html", result=result)
    

@app.route("/")
def index():
    # 해당 테스트 코드는 잘 작동하지만 별도의 함수로 빠져나가야함
    test_str = "이 영화 재미있어요! 하하"
    test_matrix = tfidf_vector.transform([test_str])
    result = model_lr.predict(test_matrix)
    print(result)
    return render_template("index.html")

# @app.route("/hello")
# def hello():
#     return "하하하!"

if __name__ == "__main__":
    load_lr() # 순서 중요 !
    app.run(port=5001)
    ##