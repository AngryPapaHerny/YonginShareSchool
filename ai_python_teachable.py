import tensorflow as tf
import numpy as np
import cv2

# 모델이 저장된 경로
# model_filename ='C://Users//admin//Documents//model//keras_model.h5'
model_filename ='C://Users//admin//Downloads//keras_model.h5'

# # 케라스 모델 가져오기
model = tf.keras.models.load_model(model_filename, compile=False)

# # 모델 요약 출력
# model.summary()
# # 첫 번째 레이어의 정보 출력
# print(model.layers[0])

# # 특정 레이어의 가중치 출력
# weights = model.layers[0].get_weights()
# print(weights)

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 이미지 전처리
def preprocessing(frame):
    #frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    #print(frame_reshaped)
    return frame_reshaped

# 예측용 함수
def predict(frame):
    # TensorFlow Keras 모델의 predict 메서드를 호출
    prediction = model.predict(frame)
    return prediction

while True: 
    ret, frame = capture.read()

    if cv2.waitKey(100) > 0: 
        break

    preprocessed = preprocessing(frame)
    prediction = predict(preprocessed)
    idx = np.argmax(prediction)
    
    # if (prediction[0,0] < prediction[0,1]):
    #     print('hand off')
    #     cv2.putText(frame, 'face in', (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

    # else:
    #     cv2.putText(frame, 'face out', (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    #     print('hand on')
    if(prediction[0,idx])>0.9:
        if idx==0:
            cv2.putText(frame, 'ME', (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),2)
        elif idx==1:
            cv2.putText(frame, 'Blind', (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),2)
        else:
            cv2.putText(frame, '---', (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    
    
    print(type(prediction),prediction)

    cv2.imshow("VideoFrame", frame)
    
capture.release()
cv2.destroyAllWindows()