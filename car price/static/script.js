// static/script.js

// Function to load car models based on selected company
function loadCarModels() {
    const companySelect = document.getElementById('company');
    const modelSelect = document.getElementById('car_models');
    const selectedCompany = companySelect.value;

    // Clear previous models
    modelSelect.innerHTML = '<option value="" disabled selected>Select Model</option>';

    if(selectedCompany in carData) {
        carData[selectedCompany].forEach(function(model) {
            const option = document.createElement('option');
            option.value = model;
            option.text = model;
            modelSelect.add(option);
        });
    }
}

// Handle form submission
document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent default form submission

    // Get form data
    const company = document.getElementById('company').value;
    const car_model = document.getElementById('car_models').value;
    const year = document.getElementById('year').value;
    const fuel_type = document.getElementById('fuel_type').value;
    const kilo_driven = document.getElementById('kilo_driven').value;

    // Basic validation
    if(!company || !car_model || !year || !fuel_type || kilo_driven === "") {
        alert("Please fill in all the fields.");
        return;
    }

    // Prepare data to send
    const formData = new FormData();
    formData.append('company', company);
    formData.append('car_models', car_model);
    formData.append('year', year);
    formData.append('fuel_type', fuel_type);
    formData.append('kilo_driven', kilo_driven);

    // Show loading message
    document.getElementById('prediction').innerHTML = "Wait! Predicting Price.....";

    // Send POST request to Flask server
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {throw err});
        }
        return response.json();
    })
    .then(data => {
        if(data.prediction) {
            document.getElementById('prediction').innerHTML = `Prediction: ${data.prediction}`;
        } else if(data.error) {
            document.getElementById('prediction').innerHTML = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('prediction').innerHTML = error.error ? `Error: ${error.error}` : "An error occurred. Please try again.";
    });
});
