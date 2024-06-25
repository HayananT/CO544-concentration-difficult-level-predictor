import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

# Load dataset
updated_df = pd.read_csv('updated_data_set.csv')

# Drop unnecessary columns
updated_df = updated_df.drop(['Timestamp', 'Affiliations', 'Target'], axis=1)

# Drop rows with missing values
updated_df.dropna(inplace=True)

# Convert categorical variables to numeric using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Sex', 'Relationship Status', 'Occupation', 'Social Media User?']
for col in categorical_columns:
    updated_df[col] = label_encoder.fit_transform(updated_df[col])

# Map Difficulty_level to numerical categories (0, 1, 2)
difficulty_mapping = {'low': 0, 'medium': 1, 'high': 2}
updated_df['Difficulty_level'] = updated_df['Difficulty_level'].map(difficulty_mapping)

# Ensure all features are numeric for Random Forest training
X = updated_df.drop(['Difficulty_level'], axis=1)
y = updated_df['Difficulty_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RandomForestClassifier with specified parameters
RANDOM_STATE = 42
random_forest_model = RandomForestClassifier(
    n_estimators=22,
    criterion='entropy',
    max_depth=8,
    max_features=11,
    min_samples_leaf=4,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    oob_score=True,
    warm_start=True,
    bootstrap=True,
    ccp_alpha=0.01,
    class_weight=None,
    max_leaf_nodes=31,
    max_samples=0.9,
    min_impurity_decrease=0,
    n_jobs=1,
    verbose=0,
    min_weight_fraction_leaf=0,
    monotonic_cst=None
)

# Fit the model
random_forest_model.fit(X_train, y_train)

# Predict on the test set
y_pred = random_forest_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest Classifier: {accuracy:.2f}")

# Save the model
joblib.dump(random_forest_model, 'random_forest_model.joblib')

print("Random Forest Model trained and saved successfully!")
