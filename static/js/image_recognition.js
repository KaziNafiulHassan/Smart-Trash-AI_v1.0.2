document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('waste-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadedImage = document.getElementById('uploaded-image');
    const predictionDiv = document.getElementById('prediction');
    const userBinSelect = document.getElementById('user-bin');
    const submitBinBtn = document.getElementById('submit-bin');
    const comparisonResult = document.getElementById('comparison-result');

    console.log('DOM loaded, elements:', {
        imageInput, analyzeBtn, uploadedImage, predictionDiv, userBinSelect, submitBinBtn, comparisonResult
    }); // Debug element existence

    let imageData = null;

    if (!analyzeBtn) {
        console.error('analyzeBtn not found in DOM');
        return;
    }

    analyzeBtn.addEventListener('click', () => {
        console.log('Analyze button clicked'); // Debug click event
        const file = imageInput.files[0];
        console.log('Selected file:', file); // Debug log
        if (!file) {
            predictionDiv.textContent = 'Please select an image!';
            return;
        }

        const formData = new FormData();
        formData.append('waste_image', file);
        console.log('FormData prepared, sending fetch request'); // Debug log

        fetch('/analyze_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Fetch response status:', response.status); // Debug log
            if (!response.ok) {
                console.log('Response text:', response.statusText); // Debug raw response
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Response from /analyze_image:', data); // Debug log
            if (data.success) {
                uploadedImage.src = `data:image/jpeg;base64,${data.image || ''}`;
                imageData = data || {}; // Fallback to empty object
                predictionDiv.textContent = 'Please select a bin to compare with AI prediction.';
                userBinSelect.style.display = 'block';
                submitBinBtn.style.display = 'block';
                uploadedImage.classList.add('scan-effect');
                comparisonResult.textContent = '';
            } else {
                predictionDiv.textContent = `Error: ${data.error || 'Unknown error'}`;
                imageData = {}; // Reset on error
            }
        })
        .catch(error => {
            console.error('Image analysis error:', error);
            predictionDiv.textContent = 'Error analyzing image! Check console for details.';
            imageData = {}; // Reset on error
        });
    });

    submitBinBtn.addEventListener('click', () => {
        const userBin = userBinSelect.value;
        console.log('Submitting bin choice:', userBin, 'with imageData:', imageData); // Debug log
        if (!userBin) {
            predictionDiv.textContent = 'Please select a bin!';
            return;
        }

        if (!imageData || !imageData.success) {
            predictionDiv.textContent = 'Please scan an image first or try again!';
            return;
        }

        const prediction = imageData.prediction || 'Unknown';
        const confidence = imageData.confidence || 'N/A';
        predictionDiv.textContent = `AI Prediction: ${prediction} (${confidence})`;

        fetch('/verify_prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ predicted_container: prediction, user_container: userBin })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            comparisonResult.textContent = data.feedback || 'No feedback available';
            comparisonResult.className = 'neon-text ' + (data.agreement ? 'correct' : 'incorrect');
        })
        .catch(error => {
            console.error('Verification error:', error);
            comparisonResult.textContent = 'Error verifying prediction! Check console for details.';
        });
    });
});