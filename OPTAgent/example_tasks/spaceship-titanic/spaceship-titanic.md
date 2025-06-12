## Description
You are predicting which passengers were transported to an alternate dimension during the Spaceship Titanic collision, using records recovered from the spaceship’s damaged computer system.
## Metric

Submissions are evaluated based on their [classification accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy), the percentage of predicted labels that are correct.

## Submission Format

The submission format for the competition is a csv file with the following format:

```
PassengerId,Transported
0013_01,False
0018_01,False
0019_01,False
0021_01,False
etc.
```

## Dataset Description
### The input files contain:

- **train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
    - `PassengerId` - A unique Id for each passenger. Each Id takes the form `gggg_pp` where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. People in a group are often family members, but not always.
    - `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
    - `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
    - `Cabin` - The cabin number where the passenger is staying. Takes the form `deck/num/side`, where `side` can be either `P` for *Port* or `S` for *Starboard*.
    - `Destination` - The planet the passenger will be debarking to.
    - `Age` - The age of the passenger.
    - `VIP` - Whether the passenger has paid for special VIP service during the voyage.
    - `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the *Spaceship Titanic*'s many luxury amenities.
    - `Name` - The first and last names of the passenger.
    - `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
- **test.csv** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of `Transported` for the passengers in this set.
- **sample_submission.csv** - A submission file in the correct format.
    - `PassengerId` - Id for each passenger in the test set.
    - `Transported` - The target. For each passenger, predict either `True` or `False`.

## Code Template
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate target from predictors
y = train_data["Transported"]
X = train_data.drop(["Transported"], axis=1)

# Select categorical columns with relatively low cardinality
categorical_cols = [
    cname
    for cname in X.columns
    if X[cname].nunique() < 10 and X[cname].dtype == "object"
]

# Select numerical columns
numerical_cols = [
    cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="median")

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

# Evaluate the model
score = accuracy_score(y_valid, preds)
print("Accuracy:", score)

# Preprocessing of test data, fit model
preprocessed_test_data = clf.named_steps["preprocessor"].transform(test_data)

# Get test predictions
test_preds = clf.named_steps["model"].predict(preprocessed_test_data)

# Save test predictions to file
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Transported": test_preds})
output.to_csv("./submission.csv", index=False)
```