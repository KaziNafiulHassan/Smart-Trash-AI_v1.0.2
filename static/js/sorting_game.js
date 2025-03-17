document.addEventListener('DOMContentLoaded', () => {
    const itemDisplay = document.getElementById('item-display');
    const itemText = itemDisplay ? itemDisplay.querySelector('.item-text') : null;
    const wasteImage = document.getElementById('waste-display-image');
    const binOrbits = document.querySelectorAll('.mini-bin');
    const feedback = document.getElementById('feedback');
    const nextItemBtn = document.getElementById('next-item');
    const scoreDisplay = document.getElementById('score');
    const accuracyDisplay = document.getElementById('accuracy');

    // Debug: Log DOM elements to ensure they’re found
    console.log('Sorting game DOM elements:', {
        itemDisplay, itemText, wasteImage, binOrbits, feedback, nextItemBtn, scoreDisplay, accuracyDisplay
    });

    // Check for missing elements
    if (!itemDisplay || !itemText || !wasteImage || !feedback || !nextItemBtn || !scoreDisplay || !accuracyDisplay) {
        console.error('One or more DOM elements not found for sorting game');
        return;
    }

    let correctAnswers = 0;
    let totalAttempts = 0;

    function updatePerformance() {
        // Sync with server-side performance data
        fetch('/get_performance')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    correctAnswers = data.correct_answers || 0;
                    totalAttempts = data.total_attempts || 0;
                    scoreDisplay.textContent = `${correctAnswers}/${totalAttempts}`;
                    const accuracy = totalAttempts > 0 ? (correctAnswers / totalAttempts * 100).toFixed(2) : 0;
                    accuracyDisplay.textContent = `${accuracy}%`;
                }
            })
            .catch(error => {
                console.error('Error fetching performance:', error);
                // Fallback to local values if server fails
                scoreDisplay.textContent = `${correctAnswers}/${totalAttempts}`;
                const accuracy = totalAttempts > 0 ? (correctAnswers / totalAttempts * 100).toFixed(2) : 0;
                accuracyDisplay.textContent = `${accuracy}%`;
            });
    }

    function loadItem() {
        fetch('/get_random_item')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                itemText.textContent = data.item;
                itemDisplay.dataset.correctBin = data.container;
                wasteImage.src = data.image_path || '';
                console.log('Image path:', data.image_path);
                wasteImage.style.display = data.image_path ? 'block' : 'none';
                wasteImage.style.maxWidth = '100%'; // Matches CSS
                wasteImage.style.maxHeight = '70%'; // Matches CSS
                if (!data.image_path) {
                    console.warn('No image path returned for item:', data.item);
                    wasteImage.alt = 'Image not available';
                }
                wasteImage.onerror = () => {
                    console.error('Failed to load image:', data.image_path);
                    wasteImage.style.display = 'none';
                    wasteImage.alt = 'Failed to load image';
                };
                feedback.textContent = '';
                feedback.className = 'feedback';
                binOrbits.forEach(bin => {
                    bin.classList.remove('disabled', 'correct', 'incorrect');
                });
            })
            .catch(error => {
                console.error('Error loading item:', error);
                feedback.textContent = 'Error loading item! Please try again.';
            });
    }

    binOrbits.forEach(bin => {
        bin.addEventListener('click', () => {
            if (bin.classList.contains('disabled')) return;

            const selectedBin = bin.dataset.bin;
            const item = itemText.textContent;

            fetch('/check_answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ item, container: selectedBin })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                feedback.innerHTML = `${data.feedback}<br><span style="font-size: 0.9em; color: #ffcc00;">${data.tip}</span>`;
                feedback.className = 'feedback ' + (data.correct ? 'correct' : 'incorrect');
                binOrbits.forEach(b => b.classList.add('disabled'));
                bin.classList.add(data.correct ? 'correct' : 'incorrect');
                if (data.correct) {
                    correctAnswers++;
                }
                totalAttempts++;
                updatePerformance();
            })
            .catch(error => {
                console.error('Error checking answer:', error);
                feedback.textContent = 'Error checking answer! Please try again.';
            });
        });
    });

    nextItemBtn.addEventListener('click', loadItem);
    loadItem(); // Initial load
    updatePerformance(); // Initial performance sync
});