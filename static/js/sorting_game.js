// Global variables for DOM elements that need to be accessed outside the DOMContentLoaded event
let feedback;
let aiFeedback;
let currentStreakDisplay;
let wasteExpertise = {};

// Game state variables that need to be accessed globally
let currentItem = null;
let correctAnswers = 0;
let totalAttempts = 0;
let userLevel = 1;
let levelProgress = 0;
let currentStreak = 0;
let badgesUnlocked = {
    'recycler': false,
    'eco-warrior': false,
    'waste-master': false
};

document.addEventListener('DOMContentLoaded', function() {
    // Debug: Log that the script is loading
    console.log('Sorting game script loaded');
    
    // Get DOM elements - using the correct IDs from the HTML
    const binElements = document.querySelectorAll('.mini-bin');
    const nextItemBtn = document.getElementById('next-item');
    const itemImage = document.getElementById('waste-display-image');
    const itemText = document.querySelector('.item-text');
    const itemDisplay = document.getElementById('item-display');
    const levelProgressBar = document.getElementById('level-progress');
    const userLevelDisplay = document.getElementById('user-level');
    feedback = document.getElementById('feedback');
    aiFeedback = document.getElementById('ai-feedback');
    currentStreakDisplay = document.getElementById('current-streak');
    
    // Debug: Log DOM elements to ensure they're found
    console.log('Sorting game DOM elements:', {
        binElements,
        nextItemBtn,
        feedback,
        itemImage,
        itemText,
        itemDisplay,
        levelProgressBar,
        userLevelDisplay,
        currentStreakDisplay,
        aiFeedback
    });
    
    // Check for critical missing elements
    if (!binElements || binElements.length === 0) {
        console.error('Bin elements not found');
        return;
    }
    
    if (!nextItemBtn) {
        console.error('Next item button not found');
        return;
    }
    
    if (!itemDisplay) {
        console.error('Item display not found');
        return;
    }
    
    if (!itemText) {
        console.error('Item text not found');
        return;
    }
    
    // Initialize game elements and state
    function initializeGame() {
        console.log('Initializing game...');
        
        // Reset game variables
        currentItem = null;
        userLevel = 1;
        levelProgress = 0;
        correctAnswers = 0;
        totalAttempts = 0;
        currentStreak = 0;
        
        // Reset badges
        badgesUnlocked = {
            'recycler': false,
            'eco-warrior': false,
            'waste-master': false
        };
        
        // Reset waste expertise
        wasteExpertise = {
            'PaperWaste': 0,
            'LightweightPackaging_LVP': 0,
            'Depot_Container_Glass': 0,
            'ResidualWaste': 0,
            'BioWaste': 0
        };
        
        // Update UI elements
        if (userLevelDisplay) userLevelDisplay.textContent = userLevel;
        if (levelProgressBar) levelProgressBar.style.width = '0%';
        if (currentStreakDisplay) currentStreakDisplay.textContent = '0';
        
        // Reset bin styling
        document.querySelectorAll('.mini-bin').forEach(bin => {
            bin.classList.remove('disabled', 'correct', 'incorrect');
        });
        
        // Load the first item
        getNextItem();
        
        // Update performance data
        updatePerformance();
        
        console.log('Game initialized successfully');
    }

    // Initialize analytics components
    function initializeAnalytics() {
        // Set initial level progress
        levelProgressBar.style.width = '10%';
        
        // Initialize category progress bars
        const categoryBars = document.querySelectorAll('.category-progress');
        categoryBars.forEach(bar => {
            bar.style.width = '10%';
        });
    }

    // Update performance statistics
    function updatePerformance() {
        console.log('Updating performance stats:', {
            correctAnswers,
            totalAttempts,
            currentStreak,
            userLevel,
            levelProgress
        });
        
        // Update level progress
        if (levelProgressBar) {
            levelProgressBar.style.width = `${levelProgress}%`;
        }
        
        // Update user level
        if (userLevelDisplay) {
            userLevelDisplay.textContent = userLevel;
        }
        
        // Update current streak
        if (currentStreakDisplay) {
            currentStreakDisplay.textContent = currentStreak;
        }
        
        // Update badges
        updateBadges();
    }
    
    // Update badges display
    function updateBadges() {
        // Check for badge elements
        const recyclerBadge = document.getElementById('recycler-badge');
        const ecoWarriorBadge = document.getElementById('eco-warrior-badge');
        const wasteMasterBadge = document.getElementById('waste-master-badge');
        
        // Update badge visibility based on unlocked status
        if (recyclerBadge) {
            recyclerBadge.classList.toggle('unlocked', badgesUnlocked.recycler);
        }
        
        if (ecoWarriorBadge) {
            ecoWarriorBadge.classList.toggle('unlocked', badgesUnlocked['eco-warrior']);
        }
        
        if (wasteMasterBadge) {
            wasteMasterBadge.classList.toggle('unlocked', badgesUnlocked['waste-master']);
        }
    }

    // Update category progress bars
    function updateCategoryProgress() {
        // Get the values for our bins
        const binTypes = Object.keys(binTypeHistory);
        const maxValue = Math.max(...Object.values(binTypeHistory), 1); // Ensure we don't divide by zero
        
        // Update existing category progress bars
        binTypes.forEach(binType => {
            const value = binTypeHistory[binType];
            const progressBar = document.querySelector(`.category-progress[data-category="${binType}"]`);
            
            if (progressBar) {
                const height = maxValue > 0 ? (value / maxValue * 100) : 0;
                progressBar.style.width = `${height}%`;
            }
        });
    }

    // Update gamification elements (level, badges, streak)
    function updateGameElements(accuracy) {
        // Update streak
        currentStreakDisplay.textContent = currentStreak;
        
        // Calculate level progress based on correct answers
        // Level thresholds: Level 1: 0-9, Level 2: 10-24, Level 3: 25-49, Level 4: 50-99, Level 5: 100+
        const levelThresholds = [0, 10, 25, 50, 100];
        
        // Determine current level
        for (let i = levelThresholds.length - 1; i >= 0; i--) {
            if (correctAnswers >= levelThresholds[i]) {
                const newLevel = i + 1;
                
                // If level up occurred
                if (newLevel > userLevel) {
                    userLevel = newLevel;
                    levelProgress = 0;
                    
                    // Show level up animation
                    userLevelDisplay.classList.add('level-up');
                    setTimeout(() => {
                        userLevelDisplay.classList.remove('level-up');
                    }, 1000);
                }
                
                // Calculate progress to next level
                if (i < levelThresholds.length - 1) {
                    const currentThreshold = levelThresholds[i];
                    const nextThreshold = levelThresholds[i + 1];
                    levelProgress = ((correctAnswers - currentThreshold) / (nextThreshold - currentThreshold)) * 100;
                } else {
                    levelProgress = 100; // Max level
                }
                
                break;
            }
        }
        
        // Update level progress bar
        levelProgressBar.style.width = `${levelProgress}%`;
        
        // Check for badge unlocks
        checkBadgeUnlocks(accuracy);
    }

    // Check and update badge unlocks
    function checkBadgeUnlocks(accuracy) {
        // Recycler badge: Get 10 correct answers
        if (correctAnswers >= 10 && !badgesUnlocked.recycler) {
            unlockBadge('recycler');
            badgesUnlocked.recycler = true;
        }
        
        // Eco-Warrior badge: Achieve 75% accuracy with at least 20 attempts
        if (accuracy >= 75 && totalAttempts >= 20 && !badgesUnlocked['eco-warrior']) {
            unlockBadge('eco-warrior');
            badgesUnlocked['eco-warrior'] = true;
        }
        
        // Waste Master badge: Reach level 4
        if (userLevel >= 4 && !badgesUnlocked['waste-master']) {
            unlockBadge('waste-master');
            badgesUnlocked['waste-master'] = true;
        }
    }

    // Unlock a badge with animation
    function unlockBadge(badgeName) {
        const badge = document.querySelector(`.badge[data-badge="${badgeName}"]`);
        if (badge) {
            badge.classList.remove('locked');
            badge.classList.add('unlocked');
            
            // Show notification
            showNotification(`Badge Unlocked: ${badgeName.replace('-', ' ').toUpperCase()}!`);
        }
    }

    // Show notification
    function showNotification(message) {
        // Check if notification container exists, if not create it
        let notificationContainer = document.getElementById('notification-container');
        
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            document.body.appendChild(notificationContainer);
        }
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        
        // Add to container
        notificationContainer.appendChild(notification);
        
        // Remove after animation
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => {
                notification.remove();
            }, 500);
        }, 3000);
    }

    // Load a new waste item
    function getNextItem() {
        fetch('/get_random_item')
            .then(response => {
                if (!response.ok) {
                    console.error(`HTTP error! Status: ${response.status}`);
                    return {
                        error: true,
                        status: response.status,
                        message: `Server error (${response.status}). Please try again.`
                    };
                }
                return response.json();
            })
            .then(data => {
                if (data.error === true) {
                    console.error('Server returned an error:', data.message);
                    if (aiFeedback) {
                        aiFeedback.textContent = 'Error loading item! Please try again.';
                    }
                    return;
                }
                
                // Store current item data
                currentItem = {
                    name: data.item,
                    container: data.container,
                    image_path: data.image_path || ''
                };
                
                console.log('Loaded item:', currentItem);
                
                // Update UI
                if (itemText) {
                    itemText.textContent = data.item;
                }
                
                if (itemDisplay) {
                    itemDisplay.dataset.correctBin = data.container;
                }
                
                if (itemImage) {
                    itemImage.src = data.image_path || '';
                    console.log('Image path:', data.image_path);
                    itemImage.style.display = data.image_path ? 'block' : 'none';
                    
                    if (!data.image_path) {
                        console.warn('No image path returned for item:', data.item);
                        itemImage.alt = 'Image not available';
                    }
                    
                    itemImage.onerror = () => {
                        console.error('Failed to load image:', data.image_path);
                        itemImage.style.display = 'none';
                        itemImage.alt = 'Failed to load image';
                    };
                }
                
                // Reset feedback displays
                if (feedback) {
                    feedback.textContent = '';
                    feedback.className = 'feedback';
                    feedback.style.display = 'none';
                }
                
                // Reset AI feedback to neutral message
                if (aiFeedback) {
                    aiFeedback.textContent = 'Choose the correct bin for this item!';
                }
                
                // Reset bin styling and enable bins
                enableAllBins();
            })
            .catch(error => {
                console.error('Error loading item:', error);
                if (aiFeedback) {
                    aiFeedback.textContent = 'Error loading item! Please try again.';
                }
            });
    }

    // Process answer
    function processAnswer(selectedBin) {
        if (!currentItem) {
            console.error('No current item available');
            return;
        }
        
        console.log('Processing answer:', {
            item: currentItem.name,
            container: selectedBin
        });
        
        // Disable all bins during processing
        disableAllBins();
        
        // Increment total attempts
        totalAttempts++;
        
        // Send answer to server
        fetch('/check_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                item: currentItem.name,
                container: selectedBin // Match the parameter name expected by the server
            })
        })
        .then(response => {
            if (!response.ok) {
                console.error(`HTTP error! Status: ${response.status}`);
                throw new Error(`Server error (${response.status})`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Answer check response:', data);
            
            // Update game statistics
            if (data.correct) {
                correctAnswers++;
                currentStreak++;
                levelProgress += 10;
                
                // Check for level up
                if (levelProgress >= 100) {
                    userLevel++;
                    levelProgress = 0;
                    
                    // Show level up message
                    if (aiFeedback) {
                        aiFeedback.textContent = `Level up! You're now level ${userLevel}!`;
                    }
                }
                
                // Check for badges
                if (currentStreak >= 5 && !badgesUnlocked.recycler) {
                    badgesUnlocked.recycler = true;
                    if (aiFeedback) {
                        aiFeedback.textContent = 'Badge unlocked: Recycler! Keep up the good work!';
                    }
                }
                
                if (correctAnswers >= 15 && !badgesUnlocked['eco-warrior']) {
                    badgesUnlocked['eco-warrior'] = true;
                    if (aiFeedback) {
                        aiFeedback.textContent = 'Badge unlocked: Eco-Warrior! You\'re making a difference!';
                    }
                }
                
                if (userLevel >= 3 && !badgesUnlocked['waste-master']) {
                    badgesUnlocked['waste-master'] = true;
                    if (aiFeedback) {
                        aiFeedback.textContent = 'Badge unlocked: Waste Master! You\'re a recycling expert!';
                    }
                }
            } else {
                currentStreak = 0;
            }
            
            // Show feedback
            showFeedback(data.correct, data.correct_container, selectedBin);
            
            // Update performance stats
            updatePerformance();
            
            // Get next item after delay
            setTimeout(() => {
                getNextItem();
            }, 2000);
        })
        .catch(error => {
            console.error('Error checking answer:', error);
            enableAllBins();
            
            // Show error feedback
            if (feedback) {
                feedback.textContent = 'Error checking answer! Please try again.';
                feedback.className = 'feedback incorrect';
                feedback.style.display = 'block';
            }
        });
    }

    // Add event listeners for bin selection
    binElements.forEach(bin => {
        bin.addEventListener('click', () => {
            if (bin.classList.contains('disabled')) return;

            const selectedBin = bin.dataset.bin;
            const item = itemText.textContent;
            
            console.log('Sending answer check for:', { item, container: selectedBin });
            
            processAnswer(selectedBin);
        });
    });

    // Add event listener for next item button
    nextItemBtn.addEventListener('click', getNextItem);
    
    // Initialize the game
    initializeGame();
});

// Helper function to show feedback
function showFeedback(correct, correctContainer, selectedBin) {
    console.log('Showing feedback:', { correct, correctContainer, selectedBin });
    
    if (!feedback) {
        console.error('Feedback element not found');
        return;
    }
    
    // Format the bin names for display
    const formatBinName = (binType) => {
        const binNames = {
            'PaperWaste': 'Paper',
            'LightweightPackaging_LVP': 'Packaging',
            'Depot_Container_Glass': 'Glass',
            'ResidualWaste': 'Residual',
            'BioWaste': 'Bio Waste'
        };
        return binNames[binType] || binType;
    };
    
    const formattedCorrectBin = formatBinName(correctContainer);
    const formattedSelectedBin = formatBinName(selectedBin);
    
    feedback.textContent = correct ? 
        `Correct! ${formattedCorrectBin} is the right bin.` : 
        `Sorry, ${formattedSelectedBin} is not correct. The right bin is ${formattedCorrectBin}.`;
    
    feedback.className = 'feedback ' + (correct ? 'correct' : 'incorrect');
    feedback.style.display = 'block';
    
    // Highlight the bins
    document.querySelectorAll('.mini-bin').forEach(bin => {
        // First remove any previous highlighting
        bin.classList.remove('correct', 'incorrect');
        
        // Then add the appropriate highlighting
        if (bin.dataset.bin === selectedBin) {
            bin.classList.add(correct ? 'correct' : 'incorrect');
        } else if (!correct && bin.dataset.bin === correctContainer) {
            bin.classList.add('correct');
        }
    });
    
    // Update AI feedback
    if (aiFeedback) {
        aiFeedback.textContent = correct ? 
            'Great job! You got it right!' : 
            `That's not quite right. ${formattedCorrectBin} would be better for this item.`;
    } else {
        console.warn('AI feedback element not found');
    }
    
    // Update waste expertise for the correct category
    if (correct && wasteExpertise && wasteExpertise[correctContainer] !== undefined) {
        wasteExpertise[correctContainer] += 1;
        
        // Update the expertise progress bar if it exists
        const expertiseBar = document.querySelector(`.category-progress[data-category="${correctContainer}"]`);
        if (expertiseBar) {
            const newWidth = Math.min(100, 10 + (wasteExpertise[correctContainer] * 10));
            expertiseBar.style.width = `${newWidth}%`;
        }
    }
    
    // Update streak display
    if (currentStreakDisplay) {
        currentStreakDisplay.textContent = currentStreak;
    } else {
        console.warn('Current streak display element not found');
    }
}

// Helper function to enable all bins
function enableAllBins() {
    document.querySelectorAll('.mini-bin').forEach(bin => {
        bin.classList.remove('disabled');
    });
}

// Helper function to disable all bins
function disableAllBins() {
    document.querySelectorAll('.mini-bin').forEach(bin => {
        bin.classList.add('disabled');
    });
}