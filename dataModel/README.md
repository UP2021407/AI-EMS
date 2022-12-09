# JavaScript AI Machine Learning Model

This code creates a machine learning model using TensorFlow.js, trains it on a dataset of past emergency calls, and uses it to make predictions on the likelihood of a severe situation for incoming calls. It also uses natural language processing to extract relevant information from the caller's description, and prioritises calls for immediate dispatch if the prediction indicates a high likelihood of a severe situation. Of course, this is just one possible approach to creating an application like this, and the exact details of the code will depend on the specific requirements of the application.

[Natural Overview](https://www.npmjs.com/package/natural)

The natural library is a JavaScript library for performing natural language processing tasks. It provides functions for things like tokenization, stemming, and part-of-speech tagging, which can be useful for tasks like sentiment analysis, topic modeling, and other common NLP tasks. This library is built on top of the NLP.js library and is designed to be simple and easy to use.

[TensorFlow Overview](https://www.tensorflow.org/overview)

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
