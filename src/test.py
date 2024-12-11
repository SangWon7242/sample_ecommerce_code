# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import graphviz

# 1. 데이터 전처리
# 데이터 로드
file_path = 'data/sample_ecommerce.csv'
data = pd.read_csv(file_path)

# 결측치 확인
print("Missing values:\n", data.isnull().sum())

# Yearly Amount Spent를 기준으로 Spending Category 생성
data['Spending Category'] = pd.qcut(data['Yearly Amount Spent'], 
                                    q=3, 
                                    labels=['Low', 'Medium', 'High'])

# 입력 변수와 목표 변수 분리
features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
X = data[features]
y = data['Spending Category']

# 기술 통계량 계산
stats = data[features].describe()
print("Feature Statistics:\n", stats)

# 변수 간 상관관계 히트맵 작성
correlation_matrix = data[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# 2. 의사결정나무 모델링
# 데이터를 학습용(70%)과 테스트용(30%)으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 의사결정나무 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 모델 평가 출력
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy Score:", accuracy)

# 3. 결과 시각화 및 분석
# 의사결정나무 시각화
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=features,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # 의사결정나무를 PDF 파일로 저장
graph.view()

# 변수 중요도 시각화
feature_importances = model.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Decision Tree')
plt.show()
