# jonjhans-a4 Part2

By: Jonathan Hansen
===================

**Algorithm:** Simple Model (Bayesian Network) and Hidden Markov Model.

**Initial State:** The program with the models and the training and test files ready to be analyzed before anything has been run.

**Goal State:** Accurately predict the letters from noisy images.

**Observed Variables:** These are images of characters in the test image.

**Hidden Variables:** These are the actual text characters.

Overview:
---------
The goal is to make a program that will accurately predict the characters from varying levels of noisy images. The program should accurately predict the images given varying levels of noisiness. 

Model Training:
---------------
To train the model we use training characters that include A-Z, 1-9, and punctuation to evaluate what each character should look like. We also use a training file to determine the frequency of letters and surround letters. 
One of the first problems that occurred while developing the program was how to match the images. Initially we were only matching the characters that were within the character itself, but this did not give good results. So, I opted for a fuller approach of match white characters and black characters adding up all that did not match and calculating the difference from there.
Laplace smoothing was used again in this program, as was used in Part I, to assure that no probability was zero.

Simple Model:
-------------
The simple model is implemented using simple calculation of Bayes Net. Finding the likelihood of each character in the training data compared to the test data. The character that contains the highest posterior is chosen as the “correct” character.
Initially noise handling was implemented using an input number between 1-100 indicating how noisy the user determines the image is, this was then changed given the instructions in the assignment to have a more dynamic noise handling management. 

HMM:
----
The HMM Model used these probabilities to determine the images that were most likely:

1.	Initial Probabilities: are the probability that any given letter is the starting letter, which is determined from the training file.
2.	Transition Probabilities: this is the probability that any one character is followed by any given other character.
3.	Emission Probabilities: are the is the character image compared to the image of each possible character.

The Viterbi algorithm is used in the HMM model to find the best sequence of characters that were given from the test image. 
There were problems after implementation of the Viterbi algorithm and only getting the first portion of the sentence to find a solution. After debugging it was noticed that the decimals were getting so small that the program simply could not compute. So, I implemented log probabilities, and was able to get full outputs for each sentence.

Problems and Techniques:
------------------------
I tried several techniques to get a better output for the HMM model. 
1.	I made sure that I was parsing the txt file more effectively trying to make sure that it represented a more reasonable text. 
2.	I tried weighing the center of the character image where the letters would likely show up.

Both these techniques gave me a slightly better output for the HMM model.

I also tried implementing other techniques:
1.	I tried implementing the recognition of the edges of the letters so that the outline edges would be more recognizable.
2.	I tried implementing weights for the emissions probabilities. 
The first of these I was unable to fully implement and get working despite trying to rewrite much of the code. The second of these I did implement with negligible results.

Conclusion:
-----------
At times I was getting a better output for some less noisy images, but as I got better and better output for the noisier images the less noisy output degraded just a bit. So, I ended up trying to find an area where I was still getting an okay output on some of the noisier images and still a good output for the less noisy images.
Despite trying to find different techniques in many cases the HMM model did a good job at recognizing somewhat noisy images.

