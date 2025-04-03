document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('xray-input');
    const fileName = document.getElementById('file-name');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const predictionsListDiv = document.getElementById('predictions-list');
    const annotatedImage = document.getElementById('annotated-image');
    let chart = null;

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

            // Display annotated image
            annotatedImage.src = data.annotated_image + '?t=' + new Date().getTime(); // Cache busting
            annotatedImage.onload = function() {
                console.log("ai_report:", data.ai_report);
                // Display results once image is loaded
                displayResults(data.predictions, data.ai_report);
                resultsDiv.style.display = 'block';

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
                    <p><strong>Definition:</strong> ${pred.details.definition}</p>
                    <p><strong>Clinical Significance:</strong> ${pred.details.clinical_significance}</p>
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

            // Try to identify numbered sections e.g. "1. Crown:" and add highlighting
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

        const ctx = document.getElementById('prediction-chart').getContext('2d');

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
        chart = new Chart(ctx, {
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
    }
});