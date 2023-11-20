from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from scipy import stats

# Arabic data about Federal Reserve and Jerome Powell (0: Negative, 1: Positive)
texts = [
    "جيروم باول يرفع سعر الفائدة",
    "الفيدرالي يخفض سعر الفائدة",
    "زيادة في سعر الفائدة من الفيدرالي",
    "جيروم باول يعلن عن خفض سعر الفائدة",
    "ارتفاع سعر الفائدة يضر بالاقتصاد",
    "الفيدرالي يدعم الاقتصاد بخفض الفائدة",
    "باول يقرر رفع الفائدة",
    "الفيدرالي يحفز النمو بخفض الفائدة",
    "زيادة الفائدة تقلل الاستثمار",
    "خفض الفائدة يزيد من الإنفاق",
    "جيروم باول يؤكد على ضرورة رفع الفائدة",
    "الفيدرالي يفكر في تقليل الفائدة",
    "رفع الفائدة يعزز قوة الدولار",
    "الفيدرالي يخفض الفائدة لتحفيز الاقتصاد",
    "رفع الفائدة يقلل من القروض",
    "الفيدرالي يريد دعم السوق بخفض الفائدة",
    "باول يقوم بزيادة الفائدة للحد من التضخم",
    "الفيدرالي يعلن عن خطط لخفض الفائدة",
    "الفيدرالي يرفع الفائدة مرة أخرى",
    "الفيدرالي يخفض الفائدة لدعم الوظائف",
    "رفع الفائدة يزيد من تكلفة الدين",
    "الفيدرالي يقلل الفائدة لزيادة الاستثمار",
    "باول يعتزم رفع الفائدة",
    "الفيدرالي يخفض الفائدة لتعزيز النمو",
    "باول يرفع الفائدة ويسبب قلق في الأسواق",
    "الفيدرالي يقلل الفائدة ويهدئ الأسواق",
    "زيادة في الفائدة تضر بالأعمال",
    "الفيدرالي يسهل الوصول للقروض بخفض الفائدة",
    "زيادة في الفائدة تقوي العملة",
    "الفيدرالي يخفض الفائدة للتحكم في التضخم",
    "باول يعتبر رفع الفائدة ضروري",
    "الفيدرالي يعتبر خفض الفائدة مفيد",
    "رفع الفائدة يقلل النشاط الاقتصادي",
    "الفيدرالي يخفض الفائدة لتحفيز الاقتصاد",
    "زيادة الفائدة تضعف القوة الشرائية",
    "الفيدرالي يزيد من الإنفاق عن طريق خفض الفائدة",
    "باول يقرر رفع الفائدة للحفاظ على استقرار الاقتصاد",
    "الفيدرالي يعتزم خفض الفائدة لتحفيز الأعمال",
    "رفع الفائدة يعني تراجع في الإنفاق",
    "الفيدرالي يقلل الفائدة لدعم المشروعات الصغيرة",
    "باول يرفع الفائدة بسبب البيانات الاقتصادية",
    "الفيدرالي يقرر خفض الفائدة للحد من الركود",
    "باول يؤكد على رفع الفائدة قريبا",
    "الفيدرالي يخطط لخفض الفائدة في المستقبل",
    "باول يزيد الفائدة ويثير الجدل",
    "الفيدرالي يخفض الفائدة ويسعى للتوازن",
    "رفع الفائدة يعني زيادة في الفقر",
    "الفيدرالي يخفض الفائدة لتعزيز التوظيف",
    "باول يتحدث عن رفع الفائدة في المؤتمر",
    "الفيدرالي ينوي خفض الفائدة لتقوية الاقتصاد",
]
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Splitting data into training and test sets
texts_train, texts_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42)

# Convert texts to feature vectors
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Train a simple Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predicting on test data
y_pred = clf.predict(X_test)

# Evaluate the classifier using a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Print classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Evaluate the mode of the labels
mode_labels = stats.mode(labels)[0]
print(f"\nMode of Labels: {mode_labels}")
