// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // Define class_names that was missing
    const class_names = [
        'Crown',
        'Implant',
        'Root Piece',
        'Filling',
        'Periapical lesion',
        'Retained root',
        'maxillary sinus',
        'Malaligned'
    ];

    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('xray-input');
    const fileName = document.getElementById('file-name');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const predictionsListDiv = document.getElementById('predictions-list');
    const annotatedImage = document.getElementById('annotated-image');
    let chart = null;

    // Store the current analysis for feedback
    let currentAnalysis = {
        imageId: null,
        originalPredictions: [],
        correctedPredictions: []
    };

    // Update the file name when a file is selected
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileName.textContent = this.files[0].name;
        } else {
            fileName.textContent = 'No file chosen';
        }
    });

    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(form);

        if (fileInput.files.length === 0) {
            alert('Please select a file to upload');
            return;
        }

        // Display loading indicator
        loadingDiv.style.display = 'flex';
        resultsDiv.style.display = 'none';

        // Send request to server
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Response data:", data);

            // Hide loading indicator
            loadingDiv.style.display = 'none';

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Store raw predictions for threshold filtering
            window.allPredictions = data.predictions;

            // Display annotated image
            annotatedImage.src = data.annotated_image + '?t=' + new Date().getTime(); // Cache busting
            annotatedImage.onload = function() {
                console.log("ai_report:", data.ai_report);
                // Display results once image is loaded
                displayResults(data.predictions, data.ai_report);

                // Setup feedback UI
                setupFeedbackUI(data.predictions, data.image_id);

                resultsDiv.style.display = 'block';

                // Add clean view toggle
                setupCleanViewToggle();

                // Scroll to results
                resultsDiv.scrollIntoView({ behavior: 'smooth' });
            };
        })
        .catch(error => {
            loadingDiv.style.display = 'none';
            alert('Error: ' + error.message);
            console.error('Error during analysis:', error);
        });
    });

    function displayResults(predictions, aiReport) {
        // Show results card
        document.getElementById('results-card').style.display = 'block';

        // Clear previous results
        predictionsListDiv.innerHTML = '';

        if (predictions.length === 0) {
            predictionsListDiv.innerHTML = '<p class="no-results">No conditions detected</p>';
            return;
        }

        // Create list of predictions
        predictions.forEach((pred, index) => {
            const probability = pred.probability * 100;
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            predItem.dataset.condition = pred.condition;
            predItem.dataset.probability = probability;

            // Add appropriate class based on probability
            if (probability > 80) {
                predItem.classList.add('high-confidence');
            } else if (probability > 50) {
                predItem.classList.add('medium-confidence');
            } else {
                predItem.classList.add('low-confidence');
            }

            // Add numbered condition with consistent numbering
            const numberSpan = document.createElement('span');
            numberSpan.className = 'condition-number';
            numberSpan.textContent = (index + 1);

            predItem.appendChild(numberSpan);

            // Add basic prediction info
            const infoSpan = document.createElement('span');
            infoSpan.className = 'condition';
            infoSpan.textContent = pred.condition;
            predItem.appendChild(infoSpan);

            const probSpan = document.createElement('span');
            probSpan.className = 'probability';
            probSpan.textContent = `${probability.toFixed(1)}%`;
            predItem.appendChild(probSpan);

            const probBar = document.createElement('div');
            probBar.className = 'probability-bar';
            probBar.style.width = `${probability}%`;
            predItem.appendChild(probBar);

            // Add details toggle if details are available
            if (pred.details) {
                const detailsToggle = document.createElement('button');
                detailsToggle.className = 'details-toggle';
                detailsToggle.innerHTML = 'Show Details';
                predItem.appendChild(detailsToggle);

                // Create details panel (initially hidden)
                const detailsPanel = document.createElement('div');
                detailsPanel.className = 'details-panel';
                detailsPanel.style.display = 'none';

                // Add details content from ontology
                detailsPanel.innerHTML = `
                    <p><strong>Definition:</strong> ${pred.details.definition || 'Not available'}</p>
                    <p><strong>Clinical Significance:</strong> ${pred.details.clinical_significance || 'Not available'}</p>
                    ${pred.details.types && pred.details.types.length ? `<p><strong>Types:</strong> ${pred.details.types.join(', ')}</p>` : ''}
                    ${pred.details.related_conditions && pred.details.related_conditions.length ? `<p><strong>Related Conditions:</strong> ${pred.details.related_conditions.join(', ')}</p>` : ''}
                    ${pred.details.differential_diagnosis && pred.details.differential_diagnosis.length ? `<p><strong>Differential Diagnosis:</strong> ${pred.details.differential_diagnosis.join(', ')}</p>` : ''}
                `;

                predItem.appendChild(detailsPanel);

                // Add toggle functionality
                detailsToggle.addEventListener('click', function() {
                    if (detailsPanel.style.display === 'none') {
                        detailsPanel.style.display = 'block';
                        detailsToggle.innerHTML = 'Hide Details';
                    } else {
                        detailsPanel.style.display = 'none';
                        detailsToggle.innerHTML = 'Show Details';
                    }
                });
            }

            predictionsListDiv.appendChild(predItem);
        });

        // Create chart
        createChart(predictions);

        // Display AI report if available
        const aiReportContainer = document.getElementById('ai-report-container');
        const aiReportDiv = document.getElementById('ai-report');

        if (aiReport) {
            // Format the report to highlight the numbers that match the conditions
            let formattedReport = aiReport;

            // Try to identify numbered sections like "1. Crown:" and add highlighting
            predictions.forEach((pred, index) => {
                const numberPattern = new RegExp(`${index+1}\\. ${pred.condition}`, 'gi');
                formattedReport = formattedReport.replace(numberPattern,
                    `<strong style="color: ${getColorForCondition(pred.condition)}">${index+1}. ${pred.condition}</strong>`);
            });

            if (typeof marked !== 'undefined') {
                aiReportDiv.innerHTML = marked.parse(formattedReport);
            } else {
                aiReportDiv.innerHTML = formattedReport.replace(/\n/g, '<br>');
            }
            aiReportContainer.style.display = 'block';
        } else {
            aiReportContainer.style.display = 'none';
        }
    }

    function setupFeedbackUI(predictions, imageId) {
        // Store current analysis data
        currentAnalysis.imageId = imageId;
        currentAnalysis.originalPredictions = JSON.parse(JSON.stringify(predictions));
        currentAnalysis.correctedPredictions = JSON.parse(JSON.stringify(predictions));

        // Use let instead of const for these elements
        let feedbackContainer = document.getElementById('feedback-container');
        let feedbackItems = document.getElementById('feedback-items');

        // If elements don't exist, create them
        if (!feedbackContainer) {
            console.log('Feedback container not found, creating it');
            const feedbackTab = document.getElementById('feedback-tab');
            if (feedbackTab) {
                feedbackContainer = document.createElement('div');
                feedbackContainer.id = 'feedback-container';
                feedbackContainer.className = 'feedback-container';
                feedbackTab.appendChild(feedbackContainer);
            } else {
                console.error('Feedback tab not found, cannot create feedback container');
                return;
            }
        }

        if (!feedbackItems) {
            console.log('Feedback items container not found, creating it');
            feedbackItems = document.createElement('div');
            feedbackItems.id = 'feedback-items';
            feedbackContainer.appendChild(feedbackItems);
        }

        feedbackItems.innerHTML = '';

        // Create feedback items for each prediction
        predictions.forEach((pred, index) => {
            const feedbackItem = document.createElement('div');
            feedbackItem.className = 'feedback-item';
            feedbackItem.dataset.index = index;

            feedbackItem.innerHTML = `
                <div class="feedback-header">
                    <span class="condition-number">${index + 1}</span>
                    <span class="condition">${pred.condition}</span>
                    <span class="probability">${(pred.probability * 100).toFixed(1)}%</span>
                </div>
                <div class="feedback-actions">
                    <label>
                        <input type="checkbox" class="correct-checkbox" checked>
                        Correct
                    </label>
                    <button class="remove-btn">Remove</button>
                    <div class="correction-options" style="display: none;">
                        <select class="condition-select">
                            ${class_names.map(c => `<option value="${c}" ${c === pred.condition ? 'selected' : ''}>${c}</option>`).join('')}
                        </select>
                    </div>
                </div>
            `;

            // Add event listeners for correction
            const correctCheckbox = feedbackItem.querySelector('.correct-checkbox');
            const correctionOptions = feedbackItem.querySelector('.correction-options');
            const conditionSelect = feedbackItem.querySelector('.condition-select');
            const removeBtn = feedbackItem.querySelector('.remove-btn');

            correctCheckbox.addEventListener('change', function() {
                correctionOptions.style.display = this.checked ? 'none' : 'block';

                // Update corrected predictions
                if (this.checked) {
                    currentAnalysis.correctedPredictions[index] = JSON.parse(JSON.stringify(currentAnalysis.originalPredictions[index]));
                }
            });

            conditionSelect.addEventListener('change', function() {
                // Update corrected predictions with new condition
                currentAnalysis.correctedPredictions[index] = {
                    ...currentAnalysis.correctedPredictions[index],
                    condition: this.value
                };
            });

            removeBtn.addEventListener('click', function() {
                // Mark this prediction for removal by setting to null
                // (We'll filter these out when submitting)
                currentAnalysis.correctedPredictions[index] = null;
                feedbackItem.style.display = 'none';
            });

            feedbackItems.appendChild(feedbackItem);
        });

        // Create feedback actions container if it doesn't exist
        let feedbackActions = feedbackContainer.querySelector('.feedback-actions-container');
        if (!feedbackActions) {
            feedbackActions = document.createElement('div');
            feedbackActions.className = 'feedback-actions-container';
            feedbackActions.style.marginTop = '20px';
            feedbackActions.style.display = 'flex';
            feedbackActions.style.justifyContent = 'space-between';
            feedbackContainer.appendChild(feedbackActions);
        }

        // Add button for adding new conditions if it doesn't exist
        let addConditionBtn = document.getElementById('add-condition-btn');
        if (!addConditionBtn) {
            addConditionBtn = document.createElement('button');
            addConditionBtn.id = 'add-condition-btn';
            addConditionBtn.className = 'secondary-button';
            addConditionBtn.textContent = 'Add Missing Condition';
            feedbackActions.appendChild(addConditionBtn);
        }

        // Add submit feedback button if it doesn't exist
        let submitFeedbackBtn = document.getElementById('submit-feedback-btn');
        if (!submitFeedbackBtn) {
            submitFeedbackBtn = document.createElement('button');
            submitFeedbackBtn.id = 'submit-feedback-btn';
            submitFeedbackBtn.className = 'primary-button';
            submitFeedbackBtn.textContent = 'Submit Feedback';
            feedbackActions.appendChild(submitFeedbackBtn);
        }

        // Set up add condition button event handler
        addConditionBtn.addEventListener('click', function() {
            const newIndex = currentAnalysis.correctedPredictions.length;

            // Create a new condition entry
            const newPrediction = {
                condition: class_names[0],
                probability: 1.0,
                bbox: [100, 100, 50, 50]  // Default position
            };

            // Add to corrected predictions
            currentAnalysis.correctedPredictions.push(newPrediction);

            // Add UI element
            const feedbackItem = document.createElement('div');
            feedbackItem.className = 'feedback-item new-condition';
            feedbackItem.dataset.index = newIndex;

            feedbackItem.innerHTML = `
                <div class="feedback-header">
                    <span class="condition-number">New</span>
                    <span class="condition">Added Condition</span>
                </div>
                <div class="feedback-actions">
                    <select class="condition-select">
                        ${class_names.map(c => `<option value="${c}">${c}</option>`).join('')}
                    </select>
                    <button class="remove-btn">Remove</button>
                </div>
            `;

            // Add event listeners
            const conditionSelect = feedbackItem.querySelector('.condition-select');
            const removeBtn = feedbackItem.querySelector('.remove-btn');

            conditionSelect.addEventListener('change', function() {
                currentAnalysis.correctedPredictions[newIndex].condition = this.value;
            });

            removeBtn.addEventListener('click', function() {
                currentAnalysis.correctedPredictions[newIndex] = null;
                feedbackItem.style.display = 'none';
            });

            feedbackItems.appendChild(feedbackItem);
        });

        // Set up submit feedback button event handler
        submitFeedbackBtn.addEventListener('click', function() {
            // Filter out null entries (removed predictions)
            const filteredPredictions = currentAnalysis.correctedPredictions.filter(p => p !== null);

            // Prepare data for submission
            const feedbackData = {
                image_id: currentAnalysis.imageId,
                original_predictions: currentAnalysis.originalPredictions,
                corrected_predictions: filteredPredictions,
                dentist_id: "test_dentist" // This could be a user ID if you implement authentication
            };

            // Send feedback to server
            fetch('/submit_feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedbackData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error submitting feedback: ' + data.error);
                } else {
                    alert('Thank you for your feedback! This will help improve the model.');
                    // Switch back to predictions tab
                    const predictionsTab = document.querySelector('.tab[data-tab="predictions"]');
                    if (predictionsTab) {
                        predictionsTab.click();
                    }
                }
            })
            .catch(error => {
                alert('Error submitting feedback: ' + error.message);
                console.error('Error submitting feedback:', error);
            });
        });

        // Show feedback container
        feedbackContainer.style.display = 'block';
    }

    // Helper function to get color for a condition
    function getColorForCondition(condition) {
        const conditionColors = {
            'Crown': '#ff0000',  // Red
            'Implant': '#00ff00',  // Green
            'Root Piece': '#0000ff',  // Blue
            'Filling': '#ffff00',  // Yellow
            'Periapical lesion': '#ff00ff',  // Magenta
            'Retained root': '#00ffff',  // Cyan
            'maxillary sinus': '#ffa500',  // Orange
            'Malaligned': '#800080',  // Purple
        };

        return conditionColors[condition] || '#000000';
    }

    function createChart(predictions) {
        // Limit to top 10 predictions
        const topPredictions = predictions.slice(0, 10);

        const chartCanvas = document.getElementById('prediction-chart');
        if (!chartCanvas) {
            console.error('Prediction chart element not found');
            return;
        }

        // Destroy previous chart if it exists
        if (chart) {
            chart.destroy();
        }

        // Prepare data for chart
        const labels = topPredictions.map(p => p.condition);
        const probabilities = topPredictions.map(p => p.probability * 100);

        // Create color array based on probabilities
        const backgroundColors = probabilities.map((p, i) => {
            const condition = topPredictions[i].condition;
            const color = getColorForCondition(condition);
            return color + '80'; // Add 50% transparency
        });

        // Create new chart
        try {
            chart = new Chart(chartCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Confidence (%)',
                        data: probabilities,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors.map(c => c.replace('80', '')),
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Confidence: ${context.raw.toFixed(1)}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Confidence (%)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Condition'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating chart:', error);
        }
    }

    // Simple clean view toggle implementation
    function setupCleanViewToggle() {
        // Check if toggle already exists
        if (document.querySelector('.clean-view-toggle')) return;

        const toggleButton = document.createElement('button');
        toggleButton.className = 'clean-view-toggle';
        toggleButton.textContent = 'Show Clean View';
        toggleButton.style.position = 'absolute';
        toggleButton.style.top = '10px';
        toggleButton.style.right = '10px';
        toggleButton.style.zIndex = '10';
        toggleButton.style.backgroundColor = 'rgba(0, 226, 200, 0.8)';
        toggleButton.style.color = '#242a33';
        toggleButton.style.border = 'none';
        toggleButton.style.padding = '8px 12px';
        toggleButton.style.borderRadius = '4px';
        toggleButton.style.fontWeight = 'bold';
        toggleButton.style.cursor = 'pointer';

        // Find the image container and add the button
        const imageContainer = document.querySelector('.image-container');
        if (imageContainer) {
            imageContainer.style.position = 'relative';
            imageContainer.appendChild(toggleButton);
        }

        // Track toggle state
        let isCleanView = false;

        // Add click handler
        toggleButton.addEventListener('click', function() {
            isCleanView = !isCleanView;

            if (isCleanView) {
                // Switch to clean view
                toggleButton.textContent = 'Show Annotations';

                // Add a temporary style to hide colored boxes
                const style = document.createElement('style');
                style.id = 'clean-view-style';
                style.textContent = `
                    .image-container img {
                        filter: contrast(120%) grayscale(80%);
                    }
                `;
                document.head.appendChild(style);
            } else {
                // Switch back to normal view
                toggleButton.textContent = 'Show Clean View';

                // Remove the temporary style
                const style = document.getElementById('clean-view-style');
                if (style) {
                    document.head.removeChild(style);
                }
            }
        });
    }

    // Handle tabs
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab
            this.classList.add('active');

            // Show corresponding content
            const tabName = this.getAttribute('data-tab');
            const tabContent = document.getElementById(`${tabName}-tab`);
            if (tabContent) {
                tabContent.classList.add('active');
            } else {
                console.error(`Tab content with id "${tabName}-tab" not found`);
            }
        });
    });

    // Confidence threshold slider with improved implementation
    const thresholdSlider = document.getElementById('confidence-threshold');
    const thresholdValue = document.getElementById('threshold-value');

    if (thresholdSlider) {
        // Keep track of previous threshold value to avoid unnecessary requests
        let previousThreshold = parseInt(thresholdSlider.value);

        thresholdSlider.addEventListener('input', function() {
            const threshold = parseInt(this.value);
            thresholdValue.textContent = `${threshold}%`;

            // Filter prediction list items based on threshold
            const predictionItems = document.querySelectorAll('.prediction-item');
            predictionItems.forEach(item => {
                const probability = parseFloat(item.querySelector('.probability').textContent);
                if (probability < threshold) {
                    item.style.display = 'none';
                } else {
                    item.style.display = 'flex';
                }
            });

            // Update AI report visibility
            updateReportWithThreshold(threshold);

            // Only update the image when the slider stops moving (debounce)
            clearTimeout(window.thresholdDebounce);
            window.thresholdDebounce = setTimeout(() => {
                // Only request if threshold changed by more than 5%
                if (Math.abs(threshold - previousThreshold) >= 5 &&
                    currentAnalysis && currentAnalysis.imageId) {

                    console.log(`Requesting filtered image with threshold: ${threshold}%`);
                    requestFilteredImage(currentAnalysis.imageId, threshold);
                    previousThreshold = threshold;
                }
            }, 300);
        });

        // Make sure we handle the "change" event which fires when the user releases the slider
        thresholdSlider.addEventListener('change', function() {
            const threshold = parseInt(this.value);
            if (threshold !== previousThreshold &&
                currentAnalysis && currentAnalysis.imageId) {

                console.log(`Slider released - requesting filtered image with threshold: ${threshold}%`);
                requestFilteredImage(currentAnalysis.imageId, threshold);
                previousThreshold = threshold;
            }
        });
    }

    // Function to request a filtered image with improved loading indicator
    function requestFilteredImage(imageId, threshold) {
        // Add loading indicator to image container
        const imageContainer = document.querySelector('.image-container');
        if (imageContainer) {
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.innerHTML = `
                <div class="spinner"></div>
                <div class="loading-text">Updating image...</div>
            `;

            // Remove existing indicator if present
            const existingIndicator = imageContainer.querySelector('.loading-indicator');
            if (existingIndicator) {
                imageContainer.removeChild(existingIndicator);
            }

            imageContainer.appendChild(loadingIndicator);
        }

        // Log the request details for debugging
        console.log(`Sending filter request: ${JSON.stringify({
            image_id: imageId,
            threshold: threshold
        })}`);

        fetch('/filter-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id: imageId,
                threshold: threshold
            })
        })
        .then(response => {
            console.log(`Filter response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Filter response data:`, data);

            if (data.error) {
                console.error('Filter error:', data.error);
                return;
            }

            // Update the image if filtered image URL provided
            if (data.filtered_image) {
                const annotatedImage = document.getElementById('annotated-image');
                if (annotatedImage) {
                    // Update the image source with a timestamp to prevent caching
                    annotatedImage.src = data.filtered_image + '?t=' + new Date().getTime();

                    console.log(`Updated image source to: ${annotatedImage.src}`);
                } else {
                    console.error('Annotated image element not found');
                }
            } else {
                console.error('No filtered image URL in response');
            }
        })
        .catch(error => {
            console.error('Error requesting filtered image:', error);
        })
        .finally(() => {
            // Remove loading indicator
            if (imageContainer) {
                const loadingIndicator = imageContainer.querySelector('.loading-indicator');
                if (loadingIndicator) {
                    setTimeout(() => {
                        imageContainer.removeChild(loadingIndicator);
                    }, 300); // Short delay to ensure the new image has time to load
                }
            }
        });
    }

    // Update AI report based on threshold
    function updateReportWithThreshold(threshold) {
        const aiReport = document.getElementById('ai-report');
        if (!aiReport) return;

        // Find all conditions in the report
        const reportItems = aiReport.querySelectorAll('p');

        reportItems.forEach(item => {
            // Look for confidence values in parentheses
            const confidenceMatch = item.textContent.match(/\((\d+\.\d+)%\)/);
            if (confidenceMatch) {
                const confidence = parseFloat(confidenceMatch[1]);

                // Hide items below threshold
                if (confidence < threshold) {
                    item.style.display = 'none';
                } else {
                    item.style.display = 'block';
                }
            }
        });
    }
});