# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:24:47 2017

@author: skarb
"""

# This tells matplotlib not to try opening a new window for each plot.
%matplotlib inline

# General libraries.
import re
import numpy as np
import matplotlib.pyplot as plt

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn library for importing the newsgroup data.
from sklearn.datasets import fetch_20newsgroups

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)

num_test = len(newsgroups_test.target)
test_data, test_labels = newsgroups_test.data[int(num_test/2):], newsgroups_test.target[int(num_test/2):]
dev_data, dev_labels = newsgroups_test.data[:int(num_test/2)], newsgroups_test.target[:int(num_test/2)]
train_data, train_labels = newsgroups_train.data, newsgroups_train.target

print ('training label shape:', train_labels.shape)
print ('test label shape:', test_labels.shape)
print ('dev label shape:', dev_labels.shape)
print ('labels names:', newsgroups_train.target_names)

###############################################################################

def P1(num_examples=5):
    
    for i in range(num_examples):
        print("\n")
        print("**********************")
        print("Message Example #",i)
        print("**********************")

        print("\n", "Message Category = ", categories[train_labels[i]], 
              "\n\n", "----BEGIN MESSAGE TEXT---- ", "\n\n", 
              train_data[i], "\n\n", "----END MESSAGE TEXT----")
        
P1()
###############################################################################

def P2():

# Step 1: Size of vocab and num of non-zero features
    
    print("************************************************************")
    print("Step 1: Look at size of vocab and num of non-zero features")
    print("************************************************************")
    
    vectorizer1 = CountVectorizer()
    X1 = vectorizer1.fit_transform(train_data)

    print("The training data has a corpus of {} unique words.".format(X1.shape[1]))
    
    
    cumsum = 0 #counter for avg num non-zero features
    
    for i in range(X1.shape[0]):        
        cumsum += X1[i].nnz
    avg_nnz = cumsum / X1.shape[0]

    print("There are an average of {0:.2f} non-zero features per example.".format(avg_nnz))

    tot_pt_nnz = (X1.nnz/(X1.shape[0]*X1.shape[1]))*100
    
    print("{0:.2f}% of the entries in the sparse matrix are non-zero.".format(tot_pt_nnz))
                   
    
# Step 2: Finding specific feature strings

    print("\n")
    print("*********************************************")
    print("Step 2: Finding specific feature strings")
    print("*********************************************")
    
    print("The alphabetically first word feature is \"{}\"".format(vectorizer1.get_feature_names()[0]))
    print("The alphabetically last word feature is \"{}\""
          .format(vectorizer1.get_feature_names()[len(vectorizer1.get_feature_names())-1]))        
            

# Step 3: custom vocabulary for fitting
    
    print("\n")
    print("****************************************")
    print("Step 3: custom vocabulary for fitting")
    print("****************************************")
          
    myVocab = ["atheism", "graphics", "space", "religion"] #custom vocabulary
    
    vectorizer2 = CountVectorizer(min_df=1)
    Y = vectorizer2.fit(myVocab) 
    Y = Y.transform(train_data) #transform the sparse matrix with train data

    cumsum = 0 #counter for avg num non-zero features

    for i in range(Y.shape[0]):        
        cumsum += Y[i].nnz
    avg_nnz = cumsum / Y.shape[0]

    print("There are an average of {0:.2f} non-zero features per example.".format(avg_nnz))
    
    
# Step 4: birgram/trigram character vocabulary
     
    print("\n")
    print("*********************************************")
    print("Step 4: birgram/trigram character vocabulary")
    print("*********************************************")
      
    bigram_vectorizer = CountVectorizer(analyzer = "char", ngram_range=(1,2))
    X2 = bigram_vectorizer.fit_transform(train_data)
    
    print("The training data has a bigram corpus of {} unique characters.".format(X2.shape[1]))
    
    trigram_vectorizer = CountVectorizer(analyzer = "char", ngram_range=(1,3))
    X3 = trigram_vectorizer.fit_transform(train_data)
    
    print("The training data has a trigram corpus of {} unique characters.".format(X3.shape[1]))
    
# Step 5 : prune words occurring less than 10 times   
    
    print("\n")         
    print("**************************************************")
    print("Step 5 : prune words occurring less than 10 times")
    print("**************************************************")     
          
    vectorizer3 = CountVectorizer(min_df=10)
    X4 = vectorizer3.fit_transform(train_data)
    
    print("With a 10 word min. occurrance cut off, the training data corpus has {} unique words.".format(X4.shape[1]))
    print("That is {0:.0f}% fewer words than if the corpus had no cut-off limit."
          .format((X1.shape[1]/(X1.shape[1]+X4.shape[1]))*100))
    
# Step 6: get fraction of words in dev data missing from train data

    print("\n")          
    print("*****************************************************************")
    print("Step 6: get fraction of words in dev data missing from train data")
    print("*****************************************************************") 
          
    vectorizer4 = CountVectorizer()
    X5 = vectorizer4.fit_transform(dev_data)
    
    missing_words = list(set(vectorizer4.get_feature_names()) - set(vectorizer1.get_feature_names()))
    
    print("There are {} words in the dev data that are not in the train data.".format(len(missing_words)))
    
P2()

###############################################################################

def P3():
    
    # First vectorize our train_data and dev_data 
    
    vectorizer1 = CountVectorizer()
    X1 = vectorizer1.fit_transform(train_data)  
        
    vectorizer2 = CountVectorizer()
    X2 = vectorizer2.fit(train_data)
    X2 = X2.transform(dev_data)

    # Try fitting a k-nn model

    k_values = {"n_neighbors":list(range(1,10))} # for GridSearchCV to find best k value   
    model1 = KNeighborsClassifier()
    kneighbor_model = GridSearchCV(model1, k_values, scoring="f1_macro")
    kneighbor_model.fit(X1, train_labels)
    f1score = kneighbor_model.score(X2, dev_labels) 
   
    print("\n")
    print("F1 score for {}-Nearest Neighbors Model = {:.2f}".format(kneighbor_model.best_params_["n_neighbors"],f1score))
    print("\n")

    # Try fitting a multinomial naive bayes model
    
    alphas = {'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
    model2 = MultinomialNB()
    multinomial_model = GridSearchCV(model2, alphas, scoring="f1_macro")
    multinomial_model.fit(X1, train_labels)
    f1score = multinomial_model.score(X2, dev_labels) 
    multinomial_model.best_params_["alpha"]
    
    print("\n")
    print("F1 score for a Multinomial Naive Bayes Model (w/ an alpha of {}) = {:.2f}".format(multinomial_model.best_params_["alpha"],f1score))
    print("\n")

    
    # Try fitting a logistic regression model
    C_vals = {'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
    model3 = LogisticRegression()
    regression_model = GridSearchCV(model3, C_vals, scoring="f1_macro")
    regression_model.fit(X1, train_labels)
    regression_model.best_params_
    f1score = regression_model.score(X2, dev_labels)
    regression_model.get_params    
    print("\n")
    print("F1 score for a Logistic Regression Model ( w/ a reg. strength C of {}) = {:.2f}"
          .format(regression_model.best_params_["C"],f1score))
    print("\n")
    
    C_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
    for i in C_vals:
        model3 = LogisticRegression(C=i)
        model3.fit(X1, train_labels)
        print("------------------------------------------------------------")
        for j in range(4):
            print("Sum of squared weights for label = {} and C = {} is {:.2f}"
                  .format(categories[j], i, np.sum(np.square(model3.coef_[j]))))
        print("------------------------------------------------------------")
        
P3()

###############################################################################

def P4():
    
    # First vectorize our train_data and dev_data 
    vectorizer1 = CountVectorizer()
    X1 = vectorizer1.fit_transform(train_data)  
    
    # Next a vectorizer for bigrams
    vectorizer2 = CountVectorizer(ngram_range=(1,2))
    X2 = vectorizer2.fit_transform(train_data) 
    
    # Use most optimized C reg. term
    C_val = 0.5

    # Simple logistic regression model
    regression_model1 = LogisticRegression(C=C_val)
    regression_model1.fit(X1, train_labels)
    
    # Bigram logistic regression model
    regression_model2 = LogisticRegression(C=C_val)
    regression_model2.fit(X2, train_labels)
  
    # Used for getting lists of 5 largest weight features for unigram and bigram models
    large_weights_uni = [] 
    large_weights_bi = [] 
    large_name_uni = []
    large_name_bi = []
    tmp1 = []
    tmp2 = []
    
    # The weights
    for i in range(4):
        weights_uni = np.sort(regression_model1.coef_[i])[::-1]
        weights_bi = np.sort(regression_model2.coef_[i])[::-1]
        large_weights_uni.append(weights_uni[:5])
        large_weights_bi.append(weights_bi[:5])
        
    # The feature name (word)   
    for i in range (4):
        for j in range(5):
            key = int(np.where(regression_model1.coef_ == large_weights_uni[i][j])[1])
            tmp1.append(vectorizer1.get_feature_names()[key])
            key = int(np.where(regression_model2.coef_ == large_weights_bi[i][j])[1])
            tmp2.append(vectorizer2.get_feature_names()[key])            
        large_name_uni.append(tmp1)
        large_name_bi.append(tmp2)
        tmp1 = []
        tmp2 = []
    
    # Clean decimal place for plotting
    large_weights_uni = np.round(large_weights_uni, decimals=4)
    large_weights_bi = np.round(large_weights_bi, decimals=4)
    
    # Draw tables for the data
    
    colnames = ["Weights (from 1st to 5th largest weight)", 
    "Feature Names (from 1st to 5th largest weight)"]
    
    fig, axs =plt.subplots(2,1, figsize=(10,4))
    
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].set_title("Top 5 weighted feature in uni-gram logistic regression model",
                       fontsize=14, fontweight="bold")
    axs[1].axis('tight')
    axs[1].axis('off')
    axs[1].set_title("Top 5 weighted features in bi-gram logistic regression model",
                       fontsize=14, fontweight="bold")
    
    the_table1 = axs[0].table(cellText=list(zip(large_weights_uni,large_name_uni)),
                          rowLabels=categories,
                          colLabels=colnames,
                          loc="center")
    
    the_table1 = axs[1].table(cellText=list(zip(large_weights_bi,large_name_bi)),
                          rowLabels=categories,
                          colLabels=colnames,
                          loc="center")   
    
    plt.show()

P4()

###############################################################################

def empty_preprocessor(s):
    return s

def better_preprocessor(s):
    lower_string = [x.lower() for x in s]
    
    
    
    
    
    return lower_string


  
def P5():

    processed_data = better_preprocessor(train_data)
    
    # With no pre-processing
    vectorizer1 = CountVectorizer()
    X1 = vectorizer1.fit_transform(train_data)
    
    # With pre-processing
    vectorizer2 = CountVectorizer()
    X2 = vectorizer2.fit_transform(processed_data)
    
    # Vectorize the dev data for scoring our logistic regression models
    vectorizer3 = CountVectorizer()
    Y1 = vectorizer3.fit(train_data)
    Y1 = vectorizer3.transform(dev_data)
        
    # Use most optimized C reg. term
    C_val = 0.5
    
    # Logistic regression without pre-processing
    regression_model1 = LogisticRegression(C=C_val)
    regression_model1.fit(X1, train_labels)
    preds1 = regression_model1.predict(Y1)
    score1 = metrics.f1_score(dev_labels, preds1, average="macro")
    print("F1 score for log. regression w/o pre-processing = {:.4f}".format(score1))
    
    # Logistic regression with pre-processing
    regression_model2 = LogisticRegression(C=C_val)
    regression_model2.fit(X2, train_labels)    
    preds2 = regression_model2.predict(Y1)    
    score2 = metrics.f1_score(dev_labels, preds2, average="macro")    
    print("F1 score for log. regression w/ pre-processing = {:.4f}".format(score2))    

P5()

###############################################################################

def P6():
    
    # Keep this random seed here to make comparison easier.
    np.random.seed(0)
   
    # Regularization strength term for upcoming log. reg. models
    c_vals = [0.01, 0.1, 0.5, 0.9]
    
    # First vectorize the train data
    vectorizer1 = CountVectorizer()
    X1 = vectorizer1.fit_transform(train_data)
    
    # Next vectorize the dev data for prediction
    vectorizer2 = CountVectorizer()
    Y1 = vectorizer2.fit(train_data)
    Y1 = vectorizer2.transform(dev_data)
    
    # L2 regularization
    regression_model1 = LogisticRegression(C=c_vals[2])
    regression_model1.fit(X1, train_labels)
    preds1 = regression_model1.predict(Y1)
    score1 = metrics.f1_score(dev_labels, preds1, average="macro")
    print("F1 score for log. regression with C=0.5 and L2 regularization = {:.4f}".format(score1))
    
    # Look at num of zero valued weights
    L2_zeros = np.sum(regression_model1.coef_ == 0.)
    
    #L1 regularization
    regression_model2 = LogisticRegression(penalty="l1", C=0.5)
    regression_model2.fit(X1, train_labels)
    preds2 = regression_model2.predict(Y1)
    score2 = metrics.f1_score(dev_labels, preds2, average="macro")
    print("F1 score for log. regression with C=0.5 and L1 regularization = {:.4f}".format(score2))
    
    # Look at num of zero valued weights   
    L1_zeros = np.sum(regression_model2.coef_ == 0.)
    
    print("L2 regularization gives {} weights that are equal to 0, while L1 regularization gives {} weights that are equal to 0."
          .format(L2_zeros,L1_zeros))
    
    # Some models to get a smaller vocabulary with non-zero features for a few L1 reg. values 
    regression_model3 = LogisticRegression(penalty="l1", C=c_vals[0])
    regression_model3.fit(X1, train_labels)
    
    regression_model4 = LogisticRegression(penalty="l1", C=c_vals[1])
    regression_model4.fit(X1, train_labels)
    
    regression_model5 = LogisticRegression(penalty="l1", C=c_vals[3])
    regression_model5.fit(X1, train_labels)
    
    # for extracting vocab at each C value
    all_models = [regression_model3.coef_, regression_model4.coef_, regression_model2.coef_, regression_model5.coef_]
    cut_down_vocab_part = []
    cut_down_vocab_comp = []
    temp = []
    
    for curr_model in all_models:
        for j in range(4):
            for k in range(26879):
                if (curr_model[j,k] != 0):
                    temp += [k]
            cut_down_vocab_part += list(set(temp) - set(cut_down_vocab_part))
            temp = []
        cut_down_vocab_comp.append(cut_down_vocab_part)
        cut_down_vocab_part = []

        
    # Get all the original vocab words 
    small_vocab = []
    small_vocab_comp = []
    X2 = vectorizer2.fit(train_data)    
    total_vocab = list(X2.get_feature_names())
    for i in range(4):
        for j in cut_down_vocab_comp[i]:
            small_vocab.append(total_vocab[j])
        small_vocab_comp.append(small_vocab)
        small_vocab = []

    # Get all f-scores
    f1_scores = []
    for i in range(4):
        # Vectorize the new small vocab list
        vectorizer3 = CountVectorizer()
        X2_small = vectorizer3.fit(small_vocab_comp[i])
        X2_small = vectorizer3.transform(train_data)
        
        # Vectorize the dev data on new vocab
        vectorizer4 = CountVectorizer()
        Y2_small = vectorizer3.fit(small_vocab_comp[i])
        Y2_small = vectorizer3.transform(dev_data)
        
        # New log. model with small vocab
        regression_model3 = LogisticRegression(C=c_vals[i])
        regression_model3.fit(X2_small, train_labels)
        preds3 = regression_model3.predict(Y2_small)
        
        f1_scores.append(metrics.f1_score(dev_labels, preds3, average="macro"))
    
    # Make list of vocab sizes to plot in next step
    vocab_size = []
    for i in range(4):
        vocab_size.append(len(small_vocab_comp[i]))
    
    # Plot F1 score vs. Vocab Size for Logistic Regression
    fig = plt.figure()
    fig.suptitle("Logistic Regression: F1 score for vs. Vocabulary size", fontsize=14, fontweight="bold")
    plt.ylabel("F1 Score")
    plt.xlabel("# of words in Vocabulary")
    plt.scatter(vocab_size, f1_scores, color="black") 
    plt.plot(vocab_size, f1_scores, color="red", linewidth=2)
    plt.show()
    
P6() 

###############################################################################

def P7():

    vectorizer1 = TfidfVectorizer()
    X1 = vectorizer1.fit_transform(train_data)
    
    vectorizer2 = TfidfVectorizer()
    Y1 = vectorizer2.fit(train_data)
    Y1 = vectorizer2.transform(dev_data)

    regression_model1 = LogisticRegression(C=100)
    regression_model1.fit(X1, train_labels)
   
    # Calculate R-ratio for dev data
    prob = regression_model1.predict_proba(Y1)
    preds = regression_model1.predict(Y1)
    R_vals = []

    for i in range(prob.shape[0]):
        R_vals.append(max(prob[i])/prob[i][dev_labels[i]])
    R_vals_highest = (np.sort(R_vals)[::-1])[:5]

    
    # Put the Highest R_val Messages in an array
    hard_messages = []
    hard_messages_indices = []
    for i in range(len(R_vals_highest)):
        key = int(np.where(R_vals == R_vals_highest[i])[0])
        hard_messages.append(dev_data[key])
        hard_messages_indices.append(key)



    hard_messages_pred_label = preds[hard_messages_indices]
    hard_messages_true_label = dev_labels[hard_messages_indices]
    



    for i in range(len(hard_messages)):
        print("\n")
        print("***************************")
        print("#{} Most Mis-Matched Message".format(i+1))
        print("***************************")
        print("\n")
        print("Logistic Regression thought the label was: \"{}\" ".format(categories[hard_messages_pred_label[i]]))
        print("The actual label was: \"{}\"".format(categories[hard_messages_true_label[i]]))
        print("\n\n", "----BEGIN MESSAGE TEXT---- ", "\n\n", 
              hard_messages[i], "\n\n", "----END MESSAGE TEXT----")
 
P7()

###############################################################################