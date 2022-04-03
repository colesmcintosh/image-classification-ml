from json import load
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

# Create a dataframe with the dataset
df = pd.DataFrame(digits.data)

df['target'] = digits.target

import matplotlib.pyplot as plt
plt.gray()
for i in range(6):
    plt.matshow(digits.images[i])

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), digits.target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
model = RandomForestClassifier(n_estimators=40)

# Train the classifier
model.fit(X_train, y_train)

# Print the score of the model
print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")

y_pred = model.predict(X_test)

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
sns.heatmap(matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
