#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
날씨에 따른 교통사고 예측 시스템 만들기 프로젝트
로컬 실행용 Python 스크립트
"""

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import os

# TensorFlow v2 비활성화
tf.disable_v2_behavior()

def main():
    print("=== 날씨에 따른 교통사고 예측 시스템 ===")
    
    # 1. 데이터 가져오기
    print("\n1. 데이터 가져오기...")
    
    # 1.1 날씨 데이터 가져오기
    print("1.1 날씨 데이터 가져오기...")
    try:
        weather_data = pd.read_csv('OBS_ASOS_TIM_20240524114001.csv', encoding='euc-kr')
        print("날씨 데이터 로드 완료")
        print(weather_data.head())
    except FileNotFoundError:
        print("경고: OBS_ASOS_TIM_20240524114001.csv 파일을 찾을 수 없습니다.")
        return
    
    # 1.2 교통사고 데이터 가져오기
    print("\n1.2 교통사고 데이터 가져오기...")
    try:
        accident_data = pd.read_csv('도로교통공단_사망교통사고정보_2020.csv', encoding='euc-kr')
        print("교통사고 데이터 로드 완료")
        print(accident_data.head())
    except FileNotFoundError:
        print("경고: 도로교통공단_사망교통사고정보_2020.csv 파일을 찾을 수 없습니다.")
        return
    
    # 2. 필요한 차원 데이터 셋으로 저장하기
    print("\n2. 데이터 전처리...")
    
    # 2.1 날씨 데이터 전처리
    print("2.1 날씨 데이터 전처리...")
    weather_2020 = weather_data.drop(['지점','지점명'], axis=1)
    weather_2020.rename(columns={'일시':'년월일시'}, inplace=True)
    weather_2020.to_csv('weather_2020.csv', index=False)
    print("weather_2020.csv 저장 완료")
    
    # 2.2 교통사고 데이터 전처리
    print("2.2 교통사고 데이터 전처리...")
    trafficAccident_2020_seoul = accident_data[['발생년월일시','사망자수']]
    trafficAccident_2020_seoul.to_csv('trafficAccident_2020_seoul.csv', index=False)
    print("trafficAccident_2020_seoul.csv 저장 완료")
    
    # 3. 데이터 전처리
    print("\n3. 데이터 전처리...")
    
    # 3.1 교통사고 데이터 셋 전처리
    trafficAccident_2020_seoul = pd.read_csv('trafficAccident_2020_seoul.csv')
    
    # 날짜 형식 변환
    trafficAccident_2020_seoul['발생년월일시'] = trafficAccident_2020_seoul['발생년월일시'].apply(
        lambda x: pd.to_datetime(str(x), format='%Y-%m-%d %H')
    )
    trafficAccident_2020_seoul['발생년월일시'] = pd.to_datetime(
        trafficAccident_2020_seoul['발생년월일시']
    ).dt.floor('h')
    
    # 컬럼명 변경 및 인덱스 설정
    trafficAccident_2020_seoul.rename(columns={'발생년월일시':'년월일시'}, inplace=True)
    trafficAccident_2020_seoul.set_index('년월일시', inplace=True)
    
    # 시간별 리샘플링
    trafficAccident_2020_seoul = trafficAccident_2020_seoul.resample('h').agg({'사망자수': 'sum'}).fillna(0)
    
    # 날씨 데이터 로드
    weather_2020 = pd.read_csv('weather_2020.csv')
    
    # 데이터 병합
    preparation_data = pd.merge(
        weather_2020,
        trafficAccident_2020_seoul,
        how='outer',
        left_index=True,
        right_index=True
    )
    
    # 결측값 처리
    preparation_data = preparation_data.fillna(0)
    preparation_data['사망자수'] = preparation_data['사망자수'].astype(int)
    
    # 최종 데이터 저장
    preparation_data.to_csv('preparation_data.csv', index=False)
    print("preparation_data.csv 저장 완료")
    print(f"데이터 정보: {preparation_data.info()}")
    
    # 4. 머신러닝 모델 학습
    print("\n4. 머신러닝 모델 학습...")
    
    # 데이터 준비
    numeric_columns = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '적설(cm)', '사망자수']
    xy = np.array(preparation_data[numeric_columns], dtype=np.float32)
    
    x_data = xy[:, :-1]  # 입력 특성 (날씨 데이터)
    y_data = xy[:, -1:]  # 출력 (사망자수)
    
    print(f"입력 데이터 형태: {x_data.shape}")
    print(f"출력 데이터 형태: {y_data.shape}")
    
    # TensorFlow 그래프 정의
    X = tf.placeholder(tf.float32, shape=[None, 5])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([5,1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")
    
    hypothesis = tf.matmul(X, W) + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
    train = optimizer.minimize(cost)
    
    # 모델 디렉토리 생성
    os.makedirs('./model', exist_ok=True)
    
    # 세션 실행 및 학습
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        print("모델 학습 시작...")
        for step in range(100000):
            cost_opti, hypo_opti, _opti = sess.run(
                [cost, hypothesis, train],
                feed_dict={X: x_data, y: y_data}
            )
            
            if step % 500 == 0:
                print(f"\nstep(학습 횟수): {step}")
                print(f"cost(예측값과 실제값의 편차): {cost_opti}")
                print(f"예측되는 사망자 수: {hypo_opti[0]}")
        
        # 모델 저장
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./model/model.ckpt")
        print(f"\n학습된 모델을 저장하였습니다: {save_path}")
    
    # 5. 모델 테스트
    print("\n5. 모델 테스트...")
    test_model()

def test_model():
    """학습된 모델을 사용하여 예측 테스트"""
    print("모델 테스트 시작...")
    
    # 테스트 데이터 (온도, 강수량, 풍속, 습도, 적설)
    temp = 15.0
    rain = 30.0
    windflow = 20.0
    humidity = 50.0
    snowfall = 0.0
    
    print(f"테스트 입력:")
    print(f"  온도: {temp}°C")
    print(f"  강수량: {rain}mm")
    print(f"  풍속: {windflow}m/s")
    print(f"  습도: {humidity}%")
    print(f"  적설: {snowfall}cm")
    
    # TensorFlow 그래프 재정의
    tf.reset_default_graph()
    
    X = tf.placeholder(tf.float32, shape=[None, 5])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    
    W = tf.Variable(tf.random_normal([5,1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")
    
    hypothesis = tf.matmul(X, W) + b
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    # 모델 로드 및 예측
    with tf.Session() as sess:
        sess.run(init)
        save_path = "./model/model.ckpt"
        
        try:
            saver.restore(sess, save_path)
            
            data = ((temp, rain, windflow, humidity, snowfall),)
            arr = np.array(data, dtype=np.float32)
            x_data = arr[:]
            
            prediction = sess.run(hypothesis, feed_dict={X: x_data})
            
            print(f"\n예측 결과:")
            print(f"  예상 사망자 수: {prediction[0][0]:.2f}명")
            
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            print("먼저 모델을 학습시켜주세요.")

if __name__ == "__main__":
    main()