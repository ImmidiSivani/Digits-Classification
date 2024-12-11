#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
digits=load_digits()


# In[4]:


X = digits.data  
y = digits.target


# In[ ]:





# In[8]:


plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.show()


# In[ ]:





# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# In[7]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap='viridis')
plt.show()


# In[9]:


from sklearn.svm import SVC

# Step 4: Train a model (SVM)
model1= SVC(random_state=42)
model1.fit(X_train, y_train)

# Step 5: Make predictions and evaluate
predictions = model1.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Confusion Matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap='viridis')
plt.show()

