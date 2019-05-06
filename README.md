https://www.linkedin.com/in/christopher-harvey-0a0304167/

For this Project I want to tackle three problems:

How to choose the best classifier?

How to deal with imbalanced data for classification?

How do models compare on both balanced and unbalanced datasets?

I wanted to answer these questions not only to help myself better understand how and why to classify data but also to be able to teach and help others understand it better.

To answer these questions I created standard data benchmarks using Iris, UCI Wine Quality, Robot IMU Sensor Data(IMUSD from now on), Credit Card Data(CCD from now on). Iris and Wine are the balanced datasets for our data. IMUSD and CCD are the imbalanced datasets.

All of the above datasets are fairly clean. I left CCD and Iris completely alone and did nothing to them. I Changed the Wine Quality dataset to only have a ‘Good’ 6-10 and ‘Bad’ 1-5 score to make it a binary classification task. I did ALOT to the IMUS Dataset. The data is set up in 9 classes with 72 sub groups making up those classes. The Data is HIGHLY imbalanced with a total of 487680 rows. Each group is made up of a number of series. Each series is 128 rows.

So the smallest class has only 1 group with only 27 series inside of it. The largest class has 15 groups with 779 series inside of it. The largest class is 28.85 times larger then the smallest!!! The series count for each class in order from largest to smallest is: 779, 732, 607, 514, 363, 308, 297, 189, 21. So even the jump from the second smallest to the smallest is quite large!

On top of all that the data itself Is not very useful and I had to do a TON of feature engineering to it to make it predict well. I had to transform the quaternion angles to Euler angles. I then created several new features such as: total linear acceleration, total angular velocity, acceleration vs velocity, total coordinate change, and the normalized and modified versions of all the sensor data. After doing all of that I saved the new data as target, test, and data so I wouldn’t have to do the feature engineering on each type of model in the code. 


Now onto the questions. For the first question I created a long list of models to play with and compare.
This is the list of models used: LSTM, Deep Dense NN, XGBoost, LightGBM, Reinforcement Learning, Tree, Logistic Regression, Naive Bayes, SVM, Random Forest, One Vs Rest, and a Hybrid.

Most of these models are overkill for the small datasets of Iris and Wine where every single one of these should reach a 100% success rate or close to it. Most of them will have a very hard time on the IMUSD and CCD as these imbalanced datasets are just too easy for the model to overfit or just classify all minority classes as the majority class.

Each model has their strengths. For the old guard of classifiers they can do things in small datasets and with less features that the new hotness can’t. Computation time and cost is important when talking about practicality of results and when it comes to the companies actually making these things. We don’t usually need a Deep Learning NN with 5 hidden layers and 128 neurons each layer!!! Sometimes Just using XGBoost or Boosted Forest is far more than enough. So at what point do we pull out the big guns?

I think most real life problems are complex and messy. So much so that we actually do need the stronger and better models to get a sustainable accuracy that consumers will be comfortable trusting for themselves and their families. No one will trust a car that only has a 70% accuracy at driving! Or trust a diagnosis of cancer with only a 60% confidence! People want at least a solid 90% accuracy or greater to be happy with the products and services they buy. They need the comfort of mind that what they are getting is the best and safest they can get. In this sense I think it is always appropriate to go for better models when you are stuck at a local maxima of only 60-70%. In these cases you will probably do better by bootstrapping some hybrid solution or going deeper into some NN architecture. In class it might be fine to only learn SKLearn and how to adjust model hyperparameters to increase basic datasets to increase accuracy to 100% every time but when it comes to making a product that people will trust their children’s lives to you better bet that you need a deeper more reliable toolkit under your belt.

So in recap I think we need to go for a better model when we hit local maxima of an unacceptable amount. We need to be able to correctly identify what type of problem we have and the tools available or to create new tools to solve the problem at hand. I wanted with this project to show how each tool has its limits and how to overcome it. In each notebook for each model I will go into all of that.

This brings us to question 2. What ways can we deal with imbalanced data to better improve our results for real life use cases?

SMOTE is a popular technique used to create artificial data that looks like real data. It does this by clustering the data and taking linear paths between points of data and creating a new data point on some random amount of distance on the linear path. This effectively makes the data clusters ‘larger’ and doesn’t take away from the features of each class for the model’s generality of future data.

Each model that succeeds more then the others will use certain things to classify the minority classes, let's go into some of those. SMOTE helps a lot in some of these use cases like the Classical ML models because they need to the data to be as balanced as possible. This improves the model’s performance to a respectable 40-65% accuracy rate but nowhere near the best model’s performance of a 88.86% rate! Most of these models will get stuck in the low 60’s to mid 70’s for accuracy rates. We will go over why that is and how to overcome it in the better models for the future.

One of the Deep Learning approaches to solving imbalanced data is to have imbalanced weights for neurons in the networks to add extra importance to correctly classifying a minority class. They deal heavy punishments to the error when you incorrectly classify a minority class and do very little to the error when you incorrectly classify a majority class. This creates a sense of ‘fear’ in the model as it never wants to mislabel a minority class again after the first training session. This stops the problem of the model overfitting the majority classes to achieve a higher score because that is easier to do. Instead the model always pays more attention to the data surrounding a minority class to extract its features so it can correctly identify the minority class in the future. This approach is highly effective in cases where the minority class has a sizable amount of data to it. In the case of IMUSD this worked for 3 of the minority classes but not the class with only 27 series entries as it had so little data that it couldn’t accurately tell which class it was. It does work VERY well in the case of binary classification though.

So you could create a hybrid model of SMOTE data and having weighted classes to make sure the minority class is large enough to have the weights work properly. This gives a higher accuracy on the IMUSD but it is still not the highest one.

There is also the method of One Vs All. This makes separate binary classifiers for each class. This is a great option for data that has a good amount for each class. For the IMUSD it does not work unless you were to SMOTE the data. Though the fake ‘real’ data will not give the best classification of the minority classes.

There is also the question of reinforcement learning. The long lost red-headed step child of the machine learning world (I’m a ginger so I feel like I can say that). It has some unique use cases. It learns from doing. How do you learn to classify data you might ask? Simple! You don’t. Instead you create a sort of ‘game’ the RL agent plays that puts numbers in buckets for each class. It can choose if it wants to put a 0 or a 1 in each bucket and based off of what buckets have 0’s and what have 1’s it classifies the data. This lets the model learn as it goes through all of the data and it naturally gets an idea of what every class ‘feels’ like. You can increase accuracy on imbalanced data by making the reward for classifying a minority class much higher like we did with Deep Learning. Only this time the agent REALLY wants to get rewarded. This makes it the most sensitive model to minority classes. This is a new sub field of RL and as it grows I see it completely taking over how we classify data and Machine Learning in general!

The highest technique is a hybrid of nearest neighbor, SVM, and sampling frequency analysis. It used sampled frequencies as features, it then found nearby series with nearest neighbor and adding them to the frequency data to create an average for each class then used an SVM to classify the data. This technique is awesome and would allow you to classify just about any group in most datasets. Based off of the number of groups and amount of data will decide what classifier you use. In this case SVM was a great choice as the data was clustered well as each class was rather different from the others. You would have to use more fine tuned models to identify difference in similar classes such as different types of Labradors.

You can also upsample and downsample data which basically means that you copy the exact same data twice for the minority classes or you decrease the amount of majority samples you have. Both cases can work depending on how much data you have and need for the model to classify correctly. I find that SMOTE and other methods create more stable and reliable models.

On to the last question. Some models just don’t work on complex or large data. The classical ML models are not effective enough at scale to compete with Reinforcement Learning, LightBGM, XGBoost, Deep Learning, or the Hybrid model. All of the later models reach 75-88% accuracy and far out performed Trees, Forests, Logistic, Naive Bayes, SVM, Nearest Neighbor, or SGD. This is about the result I expected.

I went into this project expecting to find that the stuff I was taught in school would work just fine and I didn’t need all the extra fluff that everyone on kaggle loves so much. I was wrong. I learned that sometimes you need sophisticated models to solve complex problems and that the real world is not nearly as nice and clean as the classroom. I learned that I was severally underprepared to solve the problems presented in front of me and how to overcome my shortcomings. All in all I learned what it means to be a data scientist. I learned how much time and effort and creativity you need to put into solving every problem. I hope that my time and effort into this project can possibly help some of you or to inspire you to do something hard. Something that you are not comfortable with and to make a difference in the world of machine learning!



I did not create these models. I did however want to spotlight some amazing models created by kaggle users.


Thomas Rohwer https://www.kaggle.com/trohwer64/submission-fourier-neighbour-detection-svm

He used normalized fourier amplitudes as features, found nearby series to be used as averages for the frequencies then used a support vector machine to classify the resulting clusters. Very original idea and highly recommend that you check out the kernal!!!

Abdur Rafae https://www.kaggle.com/abdurrafae/using-group-id-s-the-right-way

Another awesome kernal! He created 2 sets of bootstrapped models. The first layer of models consisted of multiple CNN layers followed by a Bi-LSTM. It then connects to a FC layer, the meta data is then added to the FC layer which is added to another FC layer which then outputs to the final model. The final model is has 2 pairs of inner and out models like the first model followed by some lambda layers that compare the output of the model and predicts if both the samples are from the same surface. He then uses this comparison and the overall output as the prediction, although you could create a prediction model for some extra overkill!!! Awesome work!

Prithvi https://www.kaggle.com/prith189/starter-code-for-3rd-place-solution
He created the most intense LSTM I have seen. Its hard to even describe everthing going on in this model so I would recommend checking it out if you want to be confused and interested by something. Also it takes a long time to run if you are curious about that!

Pedro Jofre Lora https://www.kaggle.com/pjofrelora/hybrid-classifier-solution-11th-place
He also used some exploitation of the data as the training and testing was linked. Based off of the obvious link he created a model that used TSFRESH to analyize the data and create new features to be used for standard classification techniques. He then used 6 different models to train on the data and used an ensemble method to create a final output from the output of the link and the 6 models. Great use of resources and a very unique exploit of the data!

Rajanikant Tenguria https://www.kaggle.com/algorrt/highest-scoring-public-kernel-starter-sol-29
He took the code from https://www.kaggle.com/friedchips/the-missing-link and finalized it. This was the code that found out that there was a link between training and test data and how to exploit it to come up with the correct solution effectively ending the kaggle competition lol. Still it was very clever and impressive that so many people can up with similar solutions instead of making better models!
