# AI Machine Learning Model with integration into web page

This is a group of cobbled together code to try and achieve a machine learning model that integrates with emergency medical services.

## Data Model

This is a programme that utilises the packages: *natural* and *tensorflow*. It analyses a data set to create a machine learning model using TensorFlow.js, trains it on a dataset of past emergency calls, and uses it to make predictions on the likelihood of a severe situation for incoming calls. More information can be found on this programme in the README file found in the folder.

## Data Modifier

This is a programme that runs in a html web page. It is used to add symptoms to issues. Not much use for now. Could be used to help build a dataset if the inputs could be taken from NLP on the calls and applied to the resulting issue that is found to be the case with the patient at the end of the EMS intervention.

## Identifier Webpage

This is also a programme that runs in a html web page. It is used to identify symptoms that are input against symptoms of known issues. This could be more useful if the inputs were from NLP on incoming calls and then applied to the machine learning model. The output would then be the result of the data from the machine learning model. This would be the assistance in diagnosis for the EMS worker on the call.
