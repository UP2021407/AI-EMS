// Create an object to store the medical issues and symptoms
var medicalInfo = {
  "": ""
};
// var fs = require("fs");

// Create inputs for the symptom and medical issue name
var symptomInput = document.createElement("input");
symptomInput.placeholder = "Symptom";
document.body.appendChild(symptomInput);


var medicalIssueInput = document.createElement("input");
medicalIssueInput.placeholder = "Medical Issue";
document.body.appendChild(medicalIssueInput);

// Create a button to add the symptom to the medical issue
var addButton = document.createElement("button");
addButton.innerHTML = "Add Symptom";
document.body.appendChild(addButton);

// Create a button to print the medical issues and symptoms
var printButton = document.createElement("button");
printButton.innerHTML = "Print Symptoms";
document.body.appendChild(printButton);

// Create a text area to display the medical issues and symptoms
var textArea = document.createElement("textarea");
document.body.appendChild(textArea);

// Set up a click event for the add button
addButton.addEventListener("click", function() {
  // Get the values from the inputs
  var symptom = symptomInput.value;
  var medicalIssue = medicalIssueInput.value;

  // Check if the medical issue already exists in the object
  if (medicalInfo.hasOwnProperty(medicalIssue)) {
    // If it does, add the symptom to the array of symptoms for that medical issue
    medicalInfo[medicalIssue].push(symptom);
  } else {
    // If it doesn't, create a new array for the medical issue and add the symptom to it
    medicalInfo[medicalIssue] = [symptom];
  }

  // Clear the input values

  fs.writeFile("data.json", JSON.stringify(medicalInfo), function(err) {
    if (err) {
      return console.log(err);
    }
    console.log("The file was saved!");
  });
  symptomInput.value = "";
  medicalIssueInput.value = "";
});

// Set up a click event for the print button
printButton.addEventListener("click", function() {
  // Clear the text area
  textArea.value = "";

  // Loop through the medical issues in the object
  for (var issue in medicalInfo) {
    // Add the medical issue and its symptoms to the text area
    textArea.value += issue + ":\n";
    for (var i = 0; i < medicalInfo[issue].length; i++) {
      textArea.value += " - " + medicalInfo[issue][i] + "\n";
    }
    textArea.value += "\n";
  }
});
