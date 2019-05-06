So unforunitely this area is not well researched. I am actively working on a solution for this from interpetting the research papers on the topic. Here are my findings and references.

Reinforcement Learning Algorithms for solving Classification Problems http://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/rl_classification.pdf

Reinforcement Learning as Classification: Leveraging Modern Classifiers https://users.cs.duke.edu/~parr/icml03.pdf

Deep Reinforcement Learning for Imbalanced Classification https://arxiv.org/pdf/1901.01379.pdf

The basic idea of reinforcement learning is learning from experience to do novel things.

In classification problems we have a faster method with supervised learning to classify data by updating weights on neural networks. This allows us to quickly learn and converge on features and generalizations about the data. This allows us to achieve high accuracy on new data and can be useful. This does however have great limits and takes much more time to do feature engineering to get high accuracy. Plus there is no guarantee that the network will generalize enough for complex tasks to be useful.

This is where reinforcement learning comes in!

In RL we have two things. An agent and an environment in which it learns. We use a process called Markov Decision Process for defining and creating our methods and algorithms. For classification we can create what is called a Classification Markov Decision Process, CMDP, which we define 3 aspects to the model:

1) A State - This is the current data we want to classify. It is the environment in which the agent acts.

2) An Action -  This is the process the agent can take to classify the data.

3) A Reward - This is the value that our agent gets from how well it classified all the data. Much like a loss function except this gives the agent a sense 'purpose' or 'goal' that it wants to attain.

This is called a mulit-agent RL problem to where there is a different agent for each class in the data. This allows us to have them work together to maximize overall reward and accuracy quickly.

We can set the states up in a variety of ways. The way I will talk about today is called the 'bucket method'. We set up the data as a vector of values. There are 3 buckets this vector can go into. The first bucket is the original input, the second bucket is a list of zeros and the third bucket is a copy of the original input.

The actions that each agent can take are simple. It can either chose to copy the original input into the 2nd bucket to have 3 copies of the data or it can delete the 3rd bucket and set the value to 0 effectively only keeping the original data. An agent of the correct class should want to copy the data in the second bucket. An agent of the wrong class should try to clear both of the last buckets and keep them at 0.

The reward is set up to where an agent either tries to maximize or minimize rewards. This is based off of if the agent is for the current class or not. In order to train the model to classify unseen new data we can let the agent play with the data to see if it gets a positive or negative reward. This encourages the model to learn the features behind the data to predict the reward before they recieve it.

We define the value function of the model as follows:

Let x^i be an input vector of length m and y^i be the target class for the input.

the first bucket is always just x^i. The agent is always allowed to see bucket one and therefore the input. The second bucket is initially set to all 0's and these 0's can be set by the agent to be copies of elements of x^i. The third bucket is set up as a copy of x^i, the agent can set this input to all 0's through its actions. The agent can only change one value at a time and can either choose to change a single number in bucket 2 or 3. The reward function is call independent and is only based on the number of 0's left over after the agent is finished. 0 < z < 2m is the number of zeros in the last two buckets.

The reward emitted after each action is Rt = 1 - z/m . which is therefore always between -1 and 1. Although the reward function is class independent, the agent with the same class as the training data will try to maximize its reward. While the other agents will try to minimize theirs. With a max value of 1 or -1 for 100% accurate.

After all the training is complete the agents will learn what is and what isn't there data. Then you do a one vs all update on each piece of new data and the agent which 'claims' the new data will be what the classification is.

You can balance for imbalanced data using this method naturally as each method will know what is and what isn't there own data. So the size and amount of data will not change the result. You can add noise to the data to prevent overfitting or by having the training data be 'forgotten' every state but the features or weights kept.

The paper Deep Reinforcement Learning for Imbalanced Classification has some great ideas on how to better overcome the hurdles of real world data problems. I highly recommend you to give it a read!

Reinforcement learning is actively taking over all areas of machine learning and supervised learning will not be any different. I look forward to working more on creating this process in the future!


I will add the code to this project as soon as I get it working and figured out!!!