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
    let retryCount = 0;
    const MAX_RETRIES = 2;

    if (!analyzeBtn) {
        console.error('analyzeBtn not found in DOM');
        return;
    }

    // Function to handle image upload with retry logic
    function uploadImage(file, attempt = 0) {
        if (attempt > 0) {
            predictionDiv.textContent = `Retrying upload (attempt ${attempt}/${MAX_RETRIES})...`;
        } else {
            predictionDiv.textContent = 'Analyzing image...';
        }

        // Show loading indicator
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Processing...';
        
        const formData = new FormData();
        formData.append('waste_image', file);
        console.log('FormData prepared, sending fetch request'); // Debug log

        // Check file size and warn if it's large
        if (file.size > 5 * 1024 * 1024) { // 5MB
            console.warn('Large file detected:', file.size, 'bytes. This may cause upload issues.');
            predictionDiv.textContent = 'Large image detected. Processing may take longer...';
        }

        // Add a timeout to abort the request if it takes too long
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

        fetch('/analyze_image', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            console.log('Fetch response status:', response.status); // Debug log
            
            if (!response.ok) {
                console.log('Response status:', response.status, 'text:', response.statusText); // Debug raw response
                
                // If we get a 502 Bad Gateway, this is likely a server timeout
                if (response.status === 502 && attempt < MAX_RETRIES) {
                    throw new Error('RETRY');
                }
                
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Response from /analyze_image:', data); // Debug log
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
            
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
            clearTimeout(timeoutId);
            console.error('Image analysis error:', error);
            
            // Handle retry logic
            if (error.message === 'RETRY' && attempt < MAX_RETRIES) {
                console.log(`Retrying upload (${attempt + 1}/${MAX_RETRIES})`);
                setTimeout(() => uploadImage(file, attempt + 1), 2000); // Wait 2 seconds before retry
                return;
            }
            
            // Handle timeout errors
            if (error.name === 'AbortError') {
                predictionDiv.textContent = 'Request timed out. The image may be too large or the server is busy.';
            } else {
                predictionDiv.textContent = 'Error analyzing image. Please try a different image or try again later.';
            }
            
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
            imageData = {}; // Reset on error
        });
    }

    analyzeBtn.addEventListener('click', () => {
        console.log('Analyze button clicked'); // Debug click event
        const file = imageInput.files[0];
        console.log('Selected file:', file); // Debug log
        
        if (!file) {
            predictionDiv.textContent = 'Please select an image!';
            return;
        }
        
        // Check file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            predictionDiv.textContent = 'Please select a valid image file (JPEG, PNG, or GIF).';
            return;
        }
        
        // Reset retry count
        retryCount = 0;
        
        // Start upload process
        uploadImage(file, 0);
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