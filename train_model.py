import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load dataset
data = pd.read_excel("student_marks_dataset.xlsx")

# 2. Select input features
X = data[[
    "study_hours_per_day",
    "attendance_percent",
    "assignment_score",
    "previous_exam_score",
    "sleep_hours"
]]

# 3. Target value
y = data["final_marks"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully and saved as model.pkl")
