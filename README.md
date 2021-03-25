LG AI 해커톤, 블럭 장난감 제조 공정 최적화 AI경진대회
=======================================

블럭 장난감의 공정 설계 알고리즘을 설계하는 대회입니다.   
대회 링크:
https://www.dacon.io/competitions/official/235612/overview/  

Requirements
=======================================
아래와 같이 필요 패키지를 설치할 수 있습니다.
```setup
pip install -r requirements.txt
```

Dataset
==================
이 저장소에 데이터셋은 제외되어 있습니다.  
데이터셋 출처: 
https://www.dacon.io/competitions/official/235612/data/

Structure
==================
```setup
.
└── main.ipynb
└── explanation.pptx
└── module
    └── __init__.py
    └── genome.py
    └── simulator.py
    └── max_count.csv 
    └── order.csv
    └── sample_submission.csv 
    └── stock.csv
```
각각의 파일에 대한 설명은 아래와 같습니다.
* main.ipynb: EDA와 메인 함수가 포함된 파일
* genome.py: 학습 모델이 설계된 파일
* simulator.py: 모델의 score를 계산하는 파일
* explanation.pptx: EDA, 전체적인 모델 구조와 결과를 설명하는 자료


Training and Evaluation
==================
### Training
학습 방법은 다음과 같습니다.
```setup
./main.ipynb
```

### Evaluation
이 대회는 최적화 문제로, 따로 evaluation이 필요하지 않았습니다.  

Results
==================

|public score|public rank|private score|private rank|  
|:------------|:------------|:------------|:------------|
|91.34|5/99|91.34|-|

<!--- 
Others
==================  
개인적으로 많이 아쉬운 대회이다.  
꽤 공을 들였는데 결과 재현이 되지 않아 최종 제출을 포기했다.  
매 실험 과정에서 commit과 기록을 하는 것이 얼마나 중요한지 알게된 대회였다.  
--->
