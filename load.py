import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # เพิ่มการนำเข้า joblib

# โหลดข้อมูล
data = pd.read_csv('D:\Project\jammyhomework\healthcare-dataset-stroke-data.csv')

# แปลงตัวแปรเชิงพาณิชย์ให้เป็นตัวเลขด้วย One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# แยกคุณลักษณะ (features) และตัวแปรเป้าหมาย (target)
X = data.drop(columns=["stroke"])
y = data["stroke"]

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดคุณลักษณะ (Feature Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)

# ฝึกโมเดล
gb_model.fit(X_train, y_train)

# ทำนายผลลัพธ์
y_pred_gb = gb_model.predict(X_test)

# แสดงผล
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", accuracy_gb)

# บันทึกโมเดลและ scaler ลงไฟล์
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("โมเดลและ scaler ถูกบันทึกเรียบร้อยแล้ว!")