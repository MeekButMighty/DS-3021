# %% [markdown]
# # Decision Tree Lab

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import graphviz 

from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz 

# %% [markdown]
# ### Steps 1-4: cleaning and prep

# %%
movie_metadata=pd.read_csv("/workspaces/DS-3021/data/movie_metadata.csv")

# %%
movie_metadata['content_rating'].unique()

# %%
def movie_cleaning(data):
    movies = data[['duration', 'genres', 'gross', 'content_rating', 'budget', 'cast_total_facebook_likes', 'imdb_score']]
    genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Horror', 'Thriller', 
              'Adventure', 'Fantasy', 'Sci-Fi', 'Animation', 'Documentary', 
              'Family', 'Mystery', 'Western', 'Crime', 'Music', 'War', 'History', 
              'Sport', 'Short', 'Biography']
    for genre in genres:
        movies[genre] = 0
    for index, row in movies.iterrows():
        for genre in genres:
            if genre in row['genres']:
                movies.loc[index, genre] = 1
    movies = movies.drop(columns=['genres'])
    movies = movies.dropna()
    movies = movies[movies['budget'] < 3500000000]
    movies = movies[movies['cast_total_facebook_likes'] < 320000]
    movies['imdb_score'] = movies['imdb_score'].apply(lambda x: '1' if x >= 7.5 else '0')
    movies['imdb_score'] = movies['imdb_score'].astype(int)
    movies_encoded = pd.get_dummies(movies, columns=['content_rating']) 
    return movies_encoded 

pd.options.mode.chained_assignment = None  # default='warn'
movs = movie_cleaning(movie_metadata)
movs

# %%
#I want to check out which genres there are the most few of.
for genre in ['Action', 'Drama', 'Comedy', 'Romance', 'Horror', 'Thriller', 
              'Adventure', 'Fantasy', 'Sci-Fi', 'Animation', 'Documentary', 
              'Family', 'Mystery', 'Western', 'Crime', 'Music', 'War', 'History', 
              'Sport', 'Short', 'Biography']:
    print(genre, movs[genre].value_counts(normalize=True) * 100)

#many of them have very small prevalences, so I will drop them.
smallprevs = ['Short', 'Documentary', 'War', 'History', 'Biography', 'Music', 'Western', 'Sport', 'Animation']
movs = movs.drop(columns=smallprevs)
movs


# %%
#I want to check out which content ratings there are the most few of.
content_rating_columns = [col for col in movs.columns if 'content_rating' in col]
# Iterate over those columns and calculate percentages
for rating in content_rating_columns:
    print(rating, movs[rating].value_counts(normalize=True) * 100)

#many of them have very small prevalences, so I will drop them.
smallprevs = ['content_rating_Approved', 'content_rating_G', 'content_rating_GP', 'content_rating_M',
              'content_rating_NC-17', 'content_rating_Passed', 'content_rating_Unrated', 'content_rating_X', 
              'content_rating_Not Rated']
movs = movs.drop(columns=smallprevs)
movs

# %% [markdown]
# ### Step 5: Prevalence

# %%
total_rows = movs.shape[0]
excellents = movs[movs['imdb_score'] == 1].shape[0]

pervalent_excellent = (excellents / total_rows) 
print(f"Prevalence of excellent movies: {pervalent_excellent}")

# %% [markdown]
# This means about 16% of the movies in our data set are classified as excellent per the imdb score. This is important because it gives us a baseline to compare our model's performance against. 

# %% [markdown]
# ### Step 6: Split the data

# %%
X = movs.drop(columns=['imdb_score'])
y = movs['imdb_score']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)
X_test, X_tune, y_test, y_tune = tts(X_test, y_test, test_size=0.5, random_state=42)


# %% [markdown]
# ### Step 7: Create the kfold object for cross validation.

# %%
kf = RepeatedStratifiedKFold(n_splits=10,n_repeats =5, random_state=42)

# %% [markdown]
# ### Step 8: Create the scoring metric you will use to evaluate your model and the max depth hyperparameter (grid search) 

# %%
scoring = ['roc_auc','recall','balanced_accuracy']

param={"max_depth" : [1,2,3,4,5,6,7,8,9,10,11]}

# %% [markdown]
# ### Step 9: Build the classifier object 

# %%
cl= DecisionTreeClassifier(random_state=1000)

# %% [markdown]
# ### Step 10: Use the kfold object and the scoring metric to find the best hyperparameter value for max depth via the grid search method.

# %%
search = GridSearchCV(cl, param, scoring=scoring, n_jobs=-1, cv=kf,refit='roc_auc', verbose=3)

# %% [markdown]
# ### Step 11: Fit the model to the training data

# %%
model = search.fit(X_train, y_train)


# %% [markdown]
# ### Step 12: What is the best depth value?

# %%
best = search.best_estimator_
print("Best parameters found: ", search.best_params_)
#best depth value is 6!

# %% [markdown]
# ### Step 13: print out the model

# %%
dot_data = export_graphviz(best, out_file =None,
               feature_names =X.columns, #feature names from dataset
               filled=True, 
                rounded=True, 
                class_names = ['ave','excellent']) #classification labels 
               
graph=graphviz.Source(dot_data)
graph

# %% [markdown]
# ### Step 14: View the results, comment on how the model performed using the metrics you selected.

# %%
#picking relevant information out of all the metrics it has from the cross-validation
auc = model.cv_results_['mean_test_roc_auc']
recall= model.cv_results_['mean_test_recall']
bal_acc= model.cv_results_['mean_test_balanced_accuracy']

SDauc = model.cv_results_['std_test_roc_auc']
SDrecall= model.cv_results_['std_test_recall']
SDbal_acc= model.cv_results_['std_test_balanced_accuracy']

#Parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

#Results of different depths as a dataframe:
final_model = pd.DataFrame(list(zip(depth, auc, recall, bal_acc,SDauc,SDrecall,SDbal_acc)),
               columns =['depth','auc','recall','bal_acc','aucSD','recallSD','bal_accSD'])


final_model.style.hide(axis='index')

# %% [markdown]
# The biggest thing that stands out to me from these results is that the recall averages around 0.2, meaning that the model isn't very good at identifying excellent movies. The one promising thing is at the ideal depth length of 6, the AUC is 0.74. This suggests that the model is at least better than random guessing.

# %% [markdown]
# ### Step 15 Which variables appear to be contributing the most (variable importance) 

# %%
varimp=pd.DataFrame(best.feature_importances_,index = X.columns,columns=['importance']).sort_values('importance', ascending=False)
print(varimp)

# %% [markdown]
# Duration seems to be contributing the most! It also looks like there are many variables (notably some genres and content ratings) not contributing at all!

# %% [markdown]
# ### Steps 16, 17, and 18: Use the predict method on the tune data and print out the results. How did the model perform on the tune data?

# %%
y_pred = model.predict(X_tune)
print(y_pred)

# %%
TPR = metrics.recall_score(y_tune, model.predict(X_tune))
print("True Positive Rate: ",TPR)
Prec = metrics.precision_score(y_tune, model.predict(X_tune))
print("Precision: ",Prec)
# the precision isn't very good!
Accuracy = metrics.accuracy_score(y_tune, model.predict(X_tune))
print("Accuracy: ",Accuracy)

# %%
#To evaluate, lets make a confustion matrix
cm = metrics.confusion_matrix(y_tune, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['average', 'excellent'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# Unsurprisingly, the model did well at correctly identifying movies that were average, because the low prevalence of excellent movies means most all of the movies are considered average. The model did not do well at identifying excellent movies, catching less than 20 percent of them. Depending on the goal, this could be a really big problem.

# %% [markdown]
# ### Step 19: What are the top 3 movies based on the tune set? Which variables are most important in predicting the top 3 movies?

# %%
probs = best.predict_proba(X_tune)
indices = X_tune.index
probs = pd.DataFrame(probs, columns=['average', 'excellent'], index=indices)
probs = probs.sort_values(by='excellent', ascending=False)
probs

# %%
#since I didn't change the indices during data cleaning, I can look up the movies
#titles in the original dataset using the indices
for i in [285, 326, 3024]:
    print(movie_metadata.loc[i]['movie_title'])

#War Horse, The Lost World: Jurassic Park, and Mrs Doubtfire are the top 3 movies

# %%
# Determine feature importances
feature_importances = model.best_estimator_.feature_importances_
features = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df.head(3))
# the most important features in predicting the top 3 movies 
# are duration, gross, and budget

# %% [markdown]
# ### Steps 20 and 21: Use a different hyperparameter for the grid search function and go through the process above again using the tune set. Did the model perform better or worse?

# %%
scoring = ['roc_auc','recall','balanced_accuracy']
new_param = {"max_leaf_nodes":[10,20,30,40,50,60,70]}
cl = DecisionTreeClassifier(random_state=1000)
search = GridSearchCV(cl, new_param, scoring=scoring, n_jobs=-1, cv=kf,refit='roc_auc', verbose=3)

# %%
model = search.fit(X_train, y_train)
best_nodes = search.best_estimator_

# %%
print("Best parameters found: ", best_nodes)
#it's best with 30 nodes

# %%
print(model.cv_results_.keys())

# %%
#picking relevant information out of all the metrics it has from the cross-validation
auc = model.cv_results_['mean_test_roc_auc']
recall= model.cv_results_['mean_test_recall']
bal_acc= model.cv_results_['mean_test_balanced_accuracy']

SDauc = model.cv_results_['std_test_roc_auc']
SDrecall= model.cv_results_['std_test_recall']
SDbal_acc= model.cv_results_['std_test_balanced_accuracy']

#Parameter:
nodes = np.unique(model.cv_results_["param_max_leaf_nodes"]).data

#Results of different depths as a dataframe:
final_model = pd.DataFrame(list(zip(nodes, auc, recall, bal_acc,SDauc,SDrecall,SDbal_acc)),
               columns =['depth','auc','recall','bal_acc','aucSD','recallSD','bal_accSD'])


final_model.style.hide(axis='index')

# %%
dot_data = export_graphviz(best_nodes, out_file =None,
               feature_names =X.columns, #feature names from dataset
               filled=True, 
                rounded=True, 
                class_names = ['ave','excellent']) #classification labels 
               
graph=graphviz.Source(dot_data)
graph

# %%
y_pred = model.predict(X_tune)
ConfusionMatrixDisplay.from_estimator(model, X_tune, y_tune, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# %%
TPR = metrics.recall_score(y_tune, model.predict(X_tune))
print("True Positive Rate: ",TPR)
Prec = metrics.precision_score(y_tune, model.predict(X_tune))
print("Precision: ",Prec)
Accuracy = metrics.accuracy_score(y_tune, model.predict(X_tune))
print("Accuracy: ",Accuracy)

# %% [markdown]
# Based off the true positive rate, the model got slightly better at identifying excellent movies with the new hyperparameter. Precision is still notably low and accuracy is about the same.

# %% [markdown]
# ### Step 22: Using the better model, predict the test data and print out the results.

# %%
y = model.predict(X_test)
print(y)

# %%
TPR = metrics.recall_score(y_test, model.predict(X_test))
print("True Positive Rate: ",TPR)
Prec = metrics.precision_score(y_test, model.predict(X_test))
print("Precision: ",Prec)
Accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
print("Accuracy: ",Accuracy)
# pretty similar results for the test set

# %% [markdown]
# ### Step 23: Summarize what you learned along the way and make recommendations to your boss on how this could be used moving forward, being careful not to over promise.

# %% [markdown]
# I learned:
# 
# a) How to implement cross validation and understand how it gives me a better evaluation of the model
# 
# b) how to use grid search to test out different hyper parameters and find the one that helps my model perform the best.
# 
# c) how to interpret what my model is doing and how I can use the information if gives me (e.g. the probabilities and equating that to top tier movies)
# 
# As for reccomendations for my boss, I would say this model still needs some work before implementation. It's good at identifying average movies, but if our goal is to pick out excellent movies, our model is not doing well. The first thing I would do to address this is change (lower) the threshold for classification as an excellent movie. Additionally, several variables could be removed to help limit noise in the model. When using this model, you need to be aware that the model is missing a good amount of excellent movies from the ones it is identifying.


