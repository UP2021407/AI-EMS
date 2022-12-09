# JavaScript AI Machine Learning Model

This code creates a machine learning model using TensorFlow.js, trains it on a dataset of past emergency calls, and uses it to make predictions on the likelihood of a severe situation for incoming calls. It also uses natural language processing to extract relevant information from the caller's description, and prioritises calls for immediate dispatch if the prediction indicates a high likelihood of a severe situation.

## Libraries

I am using the following libraries in this programme:

[Natural Overview](https://www.npmjs.com/package/natural)

[TensorFlow Overview](https://www.tensorflow.org/overview)

## Code overview

The TensorFlow.js library is used in the code to create and train a machine learning model that is capable of making predictions on the likelihood of a severe emergency situation based on information provided by the caller. The library is imported at the beginning of the code using the following line:

```js
{
    const tf = require('@tensorflow/tfjs');
}
```

This line imports the TensorFlow.js library and assigns it to a variable called tf, which can be used to access the library's functions and APIs.

The code then uses the TensorFlow.js library to define the architecture of the machine learning model, using the tf.sequential and tf.layers.dense functions:

```js
{
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
    model.add(tf.layers.dense({units: 1, activation: 'linear'}));
}
```

These lines define a simple model with two dense layers, each with 100 units and a rectified linear activation function. The input shape of the model is 10, which means that the model expects to receive input data with 10 features.

The first layer is a dense, fully-connected layer with 100 units and a ReLU activation function. The input shape of this layer is 10, which means it expects 10-dimensional input data. The second layer is also a dense, fully-connected layer, but it has only 1 unit and uses a linear activation function.

Next, the code uses the TensorFlow.js library to compile the model with a loss function and an optimizer:

```js
{
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
}
```

In this case, the model is compiled with a mean squared error loss function and an SGD (stochastic gradient descent) optimizer. This specifies how the model will be trained and how it will update its weights based on the training data.

Finally, the code uses the TensorFlow.js library to train the model on a dataset of past emergency calls:

```js
{
    model.fit(emergencyCallData, emergencyOutcomeData, {epochs: 100});
}
```

This line trains the model for 100 epochs using the provided training data, which allows the model to learn the patterns in the data that are most predictive of severe emergency situations. Once the model has been trained, it can be used to make predictions on new emergency calls.

Overall, the TensorFlow.js library is used in the code to create, compile, and train a machine learning model that can be used to make predictions on the likelihood of a severe emergency situation. The library provides a convenient and powerful set of tools for building and running machine learning models in JavaScript, and is an essential part of this application.

## Natural Library

The natural library is a JavaScript library for performing NLP (Natural Language Processing) tasks. It provides functions for things like tokenization, stemming, and part-of-speech tagging, which can be useful for tasks like sentiment analysis, topic modeling, and other common NLP tasks. This library is built on top of the NLP.js library and is designed to be simple and easy to use.

I am using it in this programme because:

* It is simple and easy to use, making it a good choice for people who are new to natural language processing.

* It provides a variety of useful functions for tokenization, stemming, and part-of-speech tagging, which can be used to preprocess text data and extract features for use in other NLP tasks.

* It is built on top of the NLP.js library, which provides a solid foundation for working with natural language data.

* It is well-documented and has a supportive community, which makes it easy to find help and resources.

## TensorFlow.js Library

TensorFlow.js is a good choice for this application because it is a powerful and widely-used library for building and running machine learning models in JavaScript. It is based on TensorFlow, a popular and widely-used machine learning library developed by Google, and provides a set of APIs and tools that are specifically designed for use in JavaScript environments.

One of the key advantages of using TensorFlow.js for this application is that it allows the model to be trained and run in the browser, which makes it easier to deploy the application as a web-based app. This eliminates the need to set up a separate server to host the model, and allows the app to be accessed by users directly from their web browsers.

Additionally, TensorFlow.js is a well-supported and widely-used library, with a large and active community of developers who are working on improving the library and building new tools and applications with it. This means that there are a wealth of resources available for learning about TensorFlow.js and for getting help with any issues that may arise while using the library.


### Glossary

**Epoch**

In machine learning, an epoch is one iteration over the entire training dataset. When training a model, the data is usually divided into smaller batches that are processed independently. An epoch is a measure of the number of times each batch of data is used for training. For example, if you have a training dataset of 1000 examples and you use a batch size of 100, it would take 10 iterations to complete one epoch. After one epoch, the model would have seen all 1000 examples, but it would have only been trained on a subset of the data (i.e. 100 examples) at a time.

**ReLU Activation Function**

In neural networks, the rectified linear unit (ReLU) activation function is a type of activation function that takes the form of f(x) = max(0, x), where x is the input to the activation function. This function has a range of [0, ∞) and returns 0 for any negative input, and x for any positive input. ReLU is a popular activation function because it is simple to compute and has been shown to work well in practice for many deep learning tasks. It has also been observed to improve the training of deep neural networks by making the gradients more stable and preventing the vanishing gradient problem.

**Stochastic Gradient Descent**

Stochastic gradient descent (SGD) is an optimization algorithm used to train deep learning models. It is a variant of gradient descent, which is an iterative method for minimizing a function by following the negative of the gradient of the function. In contrast to gradient descent, which updates the model parameters using the average gradient of the entire dataset, SGD updates the model parameters using the gradient of a single training example or a small batch of examples. This makes SGD much faster and more efficient than gradient descent, but it can also be less stable and may require more careful tuning of the learning rate and other hyperparameters.

**Squared Error Loss Function**

A squared error loss function is a common type of loss function that is used in regression tasks. It is called a "squared error" loss function because it calculates the difference between the predicted value and the true value, and then squares the result. The squaring of the difference is done to ensure that the result is always positive, regardless of whether the predicted value is greater than or less than the true value. The squared error loss is then calculated as the average of the squared differences for all of the data points in the dataset.

The squared error loss function is defined as:

L(y, ŷ) = 1/n * Σi=1^n (y - ŷ)^2

Where y is the true value, ŷ is the predicted value, and n is the number of data points in the dataset.

In general, the goal when training a machine learning model is to minimize the squared error loss, so that the predicted values are as close as possible to the true values. This is typically done by adjusting the model's parameters during training to reduce the overall squared error loss on the training data.

**Architecture of the Machine Learning Model**

The architecture of a machine learning model refers to the overall structure and design of the model, including the number of layers, the types of layers, and the connections between the layers. The architecture of a model can have a significant impact on its performance and ability to learn from data, so it is an important aspect to consider when designing a machine learning model.

**Tokenization**

Tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing such as parsing or text mining. Tokenization is an essential part of many natural language processing (NLP) tasks, such as part-of-speech tagging, syntactic parsing, and text mining.

Tokenization is an important preprocessing step in natural language processing because it allows the resulting tokens to be easily indexed and counted. This makes it possible to perform many common NLP tasks, such as identifying the most frequently occurring words in a document, or finding the relative frequency of different parts of speech.

**Stemming**

In natural language processing, stemming is the process of reducing inflected words to their word stem, base, or root form—typically a written word form. This is often used as a preprocessing step in text mining and natural language processing tasks in order to reduce the dimensionality of the input data and improve the accuracy of the algorithms that process it.

Stemming is a way of normalizing text data by reducing words to their base form, regardless of the context in which they are used. This means that words with the same stem will be treated as the same word, even if they are spelled differently. For example, the words "running", "ran", and "runs" all have the same stem, "run".

Stemming is often used in natural language processing tasks because it can help to improve the performance of machine learning algorithms. By reducing words to their base form, stemming can help to reduce the dimensionality of the data, which can improve the speed and accuracy of the algorithms that process it. Additionally, stemming can help to reduce the amount of noise in the data, which can also improve the performance of machine learning algorithms.
