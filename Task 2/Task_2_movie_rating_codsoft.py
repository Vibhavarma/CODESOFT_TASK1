import pandas as pd

# Load the movie dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")

# Show first 5 rows
print(df.head())

# Show dataset information
print(df.shape)
print(df.columns)
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove rows where Rating is missing
df = df.dropna(subset=["Rating"])

# Convert Votes into numbers
df["Votes"] = df["Votes"].str.replace(",", "").astype(float)

# Convert Year into numbers
df["Year"] = df["Year"].str.replace(r"[()]", "", regex=True).astype(float)

# Check the cleaned data
print("\nAfter cleaning:")
print(df.head())
print(df.isnull().sum())
import matplotlib.pyplot as plt

# Distribution of movie ratings
plt.figure(figsize=(8, 5))
plt.hist(df["Rating"], bins=10)
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.title("Distribution of Movie Ratings")
plt.show()
# Top 10 movie genres
genre_counts = df["Genre"].value_counts().head(10)

plt.figure(figsize=(10, 5))
genre_counts.plot(kind="bar")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.title("Top 10 Movie Genres")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Prepare data for Machine Learning

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target
X = df[["Year", "Votes"]]
y = df["Rating"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
# Create and train the Linear Regression model

model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# Predict rating for a sample movie

sample_movie = pd.DataFrame({
    "Year": [2020],
    "Votes": [1000]
})

predicted_rating = model.predict(sample_movie)

print("\nPredicted Movie Rating:", round(predicted_rating[0], 2))
# Actual vs Predicted Ratings

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.tight_layout()
plt.show()
# Improved Movie Rating Prediction Model

from sklearn.ensemble import RandomForestRegressor

# Use simple and useful features
df["Main_Genre"] = df["Genre"].fillna("Unknown").str.split(",").str[0].str.strip()

# Select features
X = df[["Year", "Votes", "Main_Genre"]].copy()
y = df["Rating"]

# Fill missing values
X["Year"] = X["Year"].fillna(X["Year"].median())
X["Votes"] = X["Votes"].fillna(X["Votes"].median())

# Convert Genre into numerical columns
X = pd.get_dummies(X, columns=["Main_Genre"], dtype=int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nImproved Model Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# Actual vs Predicted Ratings

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.tight_layout()
plt.show()