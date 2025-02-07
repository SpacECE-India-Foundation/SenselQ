document.addEventListener("DOMContentLoaded", function () {
    const predictionForm = document.getElementById("predictionForm");
    const predictedOutcome = document.getElementById("predictedOutcome");
    const hobbyDisplay = document.getElementById("hobbyDisplay");
    const hobbyText = document.getElementById("hobbyText");
  
    if (!predictionForm) {
      console.error("Element with ID 'predictionForm' not found.");
      return;
    }
  
    predictionForm.addEventListener("submit", function (event) {
      event.preventDefault();
  
      const formData = new FormData(predictionForm);
      const age = parseInt(formData.get("age")); // Get the age from the form
  
      // Determine the age group based on the age value
      let age_group = "";
      if (age >= 4 && age <= 5) {
        age_group = "4-5";
      } else if (age >= 6 && age <= 8) {
        age_group = "6-8";
      } else {
        console.error("Invalid age range.");
        return;
      }
  
      fetch("/proj2/predict", {
        method: "POST",
        body: new URLSearchParams(formData),
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Server response:", data);
          if (data["Predicted Hobby"]) {
            const hobby = data["Predicted Hobby"];
  
            if (hobbyText && hobbyDisplay) {
              hobbyText.textContent = "Your Child's Predicted Hobby: " + hobby;
              hobbyDisplay.classList.remove("hidden");
  
              // Redirect to videos page with hobby and age group
              setTimeout(() => {
                window.location.href = `/proj2/videos?hobby=${hobby}&age_group=${age_group}`;
              }, 3000);
            } else {
              console.error("Hobby display elements not found.");
            }
          } else if (data.error) {
            alert("Error: " + data.error);
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          if (predictedOutcome) {
            predictedOutcome.textContent = "Error: Unable to predict.";
          }
          console.error("Fetch error:", error);
        });
    });
  });