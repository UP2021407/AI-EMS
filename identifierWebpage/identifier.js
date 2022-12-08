document.addEventListener("DOMContentLoaded", function() {
    const input = document.getElementById("input");
    const console = document.getElementById("console");
    const button = document.getElementById("submit");

    const medicalIssues = {
        headache: {
            symptoms: ["headache", "dizziness"],
            urgency: 1
        },
        flu: {
            symptoms: ["cough", "sore throat"],
            urgency: 2
        },
        brokenbone: {
            symptoms: ["pain", "swelling"],
            urgency: 3
        }
    };

    button.addEventListener("click", function() {
        const userInput = input.value.toLowerCase().split(" ");
        let output = "";
        for (let i = 0; i < userInput.length; i++) {
            for (let issue in medicalIssues) {
                for (let j = 0; j < medicalIssues[issue].symptoms.length; j++) {
                    if (userInput[i] === medicalIssues[issue].symptoms[j]) {
                        output = output + issue + " - Urgency: " + medicalIssues[issue].urgency + "\n";
                    }
                }
            }
        }
        if (output === "") {
            output = "No medical issue found";
        }
        console.value = output;
    });

    
    });
    
    const clearButton = document.getElementById("clear");
        confirmButton.addEventListener("click", function(){
        document.getElementById("console").value = "";
        });


    const confirmButton = document.getElementById("confirm");
        confirmButton.addEventListener("click", function() {
        const userSymptom = symInput.value.toLowerCase().split(" ");
        const userName = issuesNameInput.value;

        medicalIssues[userName].symptoms.concat(userSymptom);
        clearButton.addEventListener("click", function() {
        console.value = "";
    });
});
