
# Disaster Response Message Classification Pipelines
Udacity Data Scientist Nanodegree Project.

### Table of Contents

1. [Project Description](#ProjectDescription)
2. [Libraries](#Libraries)
3. [Project Components](#ProjectComponents)
4. [File Description](#FileDescription)
5. [Instructions](#Instructions)
6. [Results](#Results)

## Project Description <a name="ProjectDescription"></a>
This is project assigned by Udacity data scientist nanodegree. 
In this project, i'll use disaster data from [Appen](https://appen.com/) (formally Figure 8) to build a model that classifies disaster messages.
The project will include a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data. 


## Libraries <a name="Libraries"></a>
This code runs with Python version 3.9 and requires some libraries:
```bash
pandas
numpy
sqlalchemy
plotly
NLTK
sklearn
joblib
flask
```



## Project Components <a name="ProjectComponents"></a>
There are three components in this project:
### 1. ETL Pipeline
In a Python script, data/process_data.py, contains a cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database


### 2. ML Pipeline
In a Python script, models/train_classifier.py, contains a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### 3. Flask Web App
* A web app where an emergency worker can input a new message and get classification results in several categories. 
* Display visualizations of the data.

## File Description <a name="FileDescription"></a>
```bash
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- train_classifier.py
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterClean.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

## Instructions <a name="Instructions"></a>
### ETL
Navigate to data folder and run this command: 
```bash
python process_data.py disaster_messages.csv disaster_categories.csv DisasterClean.db
```
* disaster_messages.csv: messages datset
* disaster_categories.csv: categories datset
* DisasterClean.db: output database name

Data (DisasterClean.db) will be saved to models folder.


### Train model
Navigate to models folder and run this command:
```bash
python train_classifier.py ../data/DisasterClean.db classifier.pkl False
```
* ../data/DisasterClean.db: path to database
* classifier.pkl: output model
* True or False: using GridSearchCV for tunning or not, using GridSearchCV will take time to train

Trained model (classifier.pkl) will be saved to models folder.


### Run Flask web App
Navigate to app folder and run this command:
```bash
python run.py
```
The address of web app should be something like: http://0.0.0.0:3000/


## Results <a name="Results"></a>
Plot demo:
![alt text](https://github.com/luckykid1993/Udacity/blob/main/Project2-DisasterResponsePipline/img/pic1.PNG)

Classification Demo:
![alt text](https://github.com/luckykid1993/Udacity/blob/main/Project2-DisasterResponsePipline/img/pic2.PNG)
![alt text](https://github.com/luckykid1993/Udacity/blob/main/Project2-DisasterResponsePipline/img/pic3.PNG)
