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
