
# Starbuck's Capstone Challenge
Udacity Data Scientist Nanodegree Project.

### Table of Contents

1. [Project Description](#ProjectDescription)
2. [Project Motivation](#ProjectMotivation)
3. [Libraries](#Libraries)
4. [File Description](#FileDescription)
5. [Results](#Results)

## Project Description <a name="ProjectDescription"></a>
This project is part of Udacity's Data Scientist Nanodegree program: Data Scientist Capstone. 
Starbuck is now one of the world's largest coffee companies. 
So, what is it that makes Starbuck so well-known? 
In this project, I will use Starbuck's simulated data to see whether I can create a simple machine learning model.

## Project Motivation <a name="ProjectMotivation"></a>
I used data from Starbuck, to find answer to below questions:

1. What factors have a major impact on the use of a offer?
2. Is it possible to create a model that predicts whether or not someone will accept an offer based on demographic data?

## Libraries <a name="Libraries"></a>
This code runs with Python version 3.9 and requires some libraries:
```bash
pandas
numpy
math
json
sklearn
matplotlib
seaborn
```

## File Description <a name="FileDescription"></a>
```bash
- data
|- portfolio.json 	# portfolio data
|- profile.json		# customer's data
|- transcript.json	# transcript data

- Starbucks_Capstone_notebook.ipynb # The notebook that i used to find the anwser for the quesstions
- README.md
```


## Results <a name="Results"></a>
Based on the project's findings, I believe we can apply a machine learning model to predict whether or not a customer will accept the offer. 
The best model also tells us the most important factors that influence the likelihood of customers responding to the offer, such as membership term, age, and income.

1. What factors have a major impact on the use of a offer?
> The length of membership is the most crucial aspect in determining whether or not the offer will be accepted. 
That is, the longer a client has been a Starbucks member, the more likely he or she is to respond to an offer. 
The second and third most important criteria that influence a customer's likelihood of responding are age and income.

2. Is it possible to create a model that predicts whether or not someone will accept an offer based on demographic data?
> We can build a model that predicts whether or not someone will respond to an offer. 
The accuracy is greater than 60% for all three models, in terms of business, I think that it is a good accuracy rate, and it is acceptable in this project.

Blog post: 