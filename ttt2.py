def classify_sentence(sentence):
    """
    Classify a given sentence as positive (1) or negative (0) 
    based on the trained Naive Bayes classifier.
    """
    # Convert the sentence to a feature vector
    sentence_vector = vectorizer.transform([sentence])
    
    # Use the classifier to predict the sentiment
    prediction = clf.predict(sentence_vector)
    
    # Return the predicted sentiment
    return prediction[0]

# Example usage:
sentence = "جيروم باول يرفع سعر الفائدة بسبب التضخم الإقتصادي"
print(f"Predicted sentiment for the sentence: {classify_sentence(sentence)}")
