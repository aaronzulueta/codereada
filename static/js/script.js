document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsContainer = document.getElementById('results-container');
    const extractedCode = document.getElementById('extracted-code');
    const copyCodeBtn = document.getElementById('copy-code-btn');
    const scoreValue = document.getElementById('score-value');
    const scoreLabel = document.getElementById('score-label');
    const scorePoints = document.querySelectorAll('.score-point');
    const bugIndicator = document.getElementById('bug-indicator');
    const bugStatus = document.getElementById('bug-status');
    const errorTypeContainer = document.getElementById('error-type-container');
    const errorTypeBadge = document.getElementById('error-type-badge');
    const toastContainer = document.getElementById('toast-container');

    // Score labels based on Likert scale
    const scoreLabels = {
        1: 'Very Poor',
        2: 'Poor',
        3: 'Average',
        4: 'Good',
        5: 'Excellent'
    };

    // Error type labels - updated with the provided mapping
    const errorTypes = {
        0: 'No Error',
        1: 'Unknown Error',
        2: 'AttributeError',
        3: 'EOFError',
        4: 'FileNotFoundError',
        5: 'ImportError',
        6: 'IndexError',
        7: 'KeyError',
        8: 'MLE',
        9: 'ModuleNotFoundError',
        10: 'NameError',
        11: 'OverflowError',
        12: 'RecursionError',
        13: 'RuntimeError',
        14: 'SyntaxError',
        15: 'TLE',
        16: 'TypeError',
        17: 'UnboundLocalError',
        18: 'ValueError',
        19: 'ZeroDivisionError'
    };

    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    removeFileBtn.addEventListener('click', removeFile);
    uploadForm.addEventListener('submit', analyzeCode);
    copyCodeBtn.addEventListener('click', copyCodeToClipboard);

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);

    // Functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        uploadArea.classList.add('dragover');
    }

    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    }

    function handleFileSelect() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            
            // Check if file is an image
            if (!file.type.match('image.*')) {
                showToast('Please select an image file', 'error');
                return;
            }
            
            fileName.textContent = file.name;
            fileInfo.classList.remove('d-none');
            analyzeBtn.disabled = false;
        }
    }

    function removeFile() {
        fileInput.value = '';
        fileInfo.classList.add('d-none');
        analyzeBtn.disabled = true;
    }

    function analyzeCode(e) {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            showToast('Please select an image file', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing...';
        
        // Hide previous results
        resultsContainer.classList.add('d-none');
        
        // Send AJAX request
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Network response was not ok');
                });
            }
            return response.json();
        })
        .then(data => {
            // Check if data is valid
            if (!data) {
                throw new Error('No data received from server');
            }
            
            // Display results
            displayResults(data);
            
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Code';
            
            // Show success message
            showToast('Analysis completed successfully', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Code';
            
            // Show error message
            showToast(error.message || 'An error occurred during analysis', 'error');
        });
    }

    function displayResults(data) {
        // Ensure data has expected properties
        const extractedText = data.extracted_text || '';
        const readabilityScore = data.readability_score;
        const bugDetected = data.bug_detected;
        const errorType = data.error_type;
        
        // Display extracted code
        if (extractedText && extractedText.trim() !== '') {
            extractedCode.textContent = extractedText;
            try {
                hljs.highlightElement(extractedCode);
            } catch (e) {
                console.error('Error highlighting code:', e);
            }
        } else {
            extractedCode.textContent = 'No code extracted';
            showToast('No code was extracted from the image. Try a clearer image.', 'warning');
        }
        
        // Display readability score if available
        if (scoreValue && scoreLabel && scorePoints) {
            if (readabilityScore !== null && readabilityScore !== undefined) {
                // Ensure readability score is an integer between 1 and 5
                const score = parseInt(readabilityScore);
                
                if (!isNaN(score) && score >= 1 && score <= 5) {
                    scoreValue.textContent = score;
                    scoreLabel.textContent = scoreLabels[score] || 'Unknown';
                    
                    // Update score points
                    scorePoints.forEach(point => {
                        const pointScore = parseInt(point.getAttribute('data-score'));
                        if (pointScore <= score) {
                            point.classList.add('active');
                        } else {
                            point.classList.remove('active');
                        }
                    });
                } else {
                    console.warn('Invalid readability score:', readabilityScore);
                    scoreValue.textContent = '-';
                    scoreLabel.textContent = 'Invalid score';
                    scorePoints.forEach(point => point.classList.remove('active'));
                }
            } else {
                scoreValue.textContent = '-';
                scoreLabel.textContent = 'Not available';
                scorePoints.forEach(point => point.classList.remove('active'));
                if (scorePoints.length > 0) {
                    scorePoints[0].classList.add('active');
                }
            }
        }
        
        // Display bug detection results if available
        if (bugIndicator && bugStatus) {
            if (bugDetected === true) {
                bugIndicator.innerHTML = '<i class="fas fa-bug"></i>';
                bugIndicator.className = 'bug-indicator has-bug';
                bugStatus.textContent = 'Bug Detected';
                
                // Display error type if available
                if (errorType !== null && errorType !== undefined && errorTypeContainer && errorTypeBadge) {
                    const errorLabel = errorTypes[errorType] || `Error Type ${errorType}`;
                    errorTypeBadge.textContent = errorLabel;
                    
                    // Reset all error type classes
                    errorTypeBadge.className = 'badge bg-danger';
                    
                    // Apply specific error type class based on the error type
                    if (errorType === 14) { // SyntaxError
                        errorTypeBadge.classList.add('error-syntax');
                    } else if (errorType === 16) { // TypeError
                        errorTypeBadge.classList.add('error-type');
                    } else if (errorType === 18) { // ValueError
                        errorTypeBadge.classList.add('error-value');
                    } else if (errorType === 10) { // NameError
                        errorTypeBadge.classList.add('error-name');
                    } else if (errorType === 2) { // AttributeError
                        errorTypeBadge.classList.add('error-attribute');
                    } else if (errorType === 5) { // ImportError
                        errorTypeBadge.classList.add('error-import');
                    } else if (errorType === 6) { // IndexError
                        errorTypeBadge.classList.add('error-index');
                    } else if (errorType === 7) { // KeyError
                        errorTypeBadge.classList.add('error-key');
                    } else if (errorType === 9) { // ModuleNotFoundError
                        errorTypeBadge.classList.add('error-module');
                    } else if (errorType === 12) { // RecursionError
                        errorTypeBadge.classList.add('error-recursion');
                    } else if (errorType === 13) { // RuntimeError
                        errorTypeBadge.classList.add('error-runtime');
                    } else if (errorType === 19) { // ZeroDivisionError
                        errorTypeBadge.classList.add('error-zero');
                    }
                    
                    errorTypeContainer.classList.remove('d-none');
                } else if (errorTypeContainer) {
                    errorTypeContainer.classList.add('d-none');
                }
            } else if (bugDetected === false) {
                bugIndicator.innerHTML = '<i class="fas fa-check"></i>';
                bugIndicator.className = 'bug-indicator no-bug';
                bugStatus.textContent = 'No Bug Detected';
                
                // Hide error type container
                if (errorTypeContainer) {
                    errorTypeContainer.classList.add('d-none');
                }
            } else {
                bugIndicator.innerHTML = '<i class="fas fa-question"></i>';
                bugIndicator.className = 'bug-indicator';
                bugStatus.textContent = 'Not analyzed';
                
                // Hide error type container
                if (errorTypeContainer) {
                    errorTypeContainer.classList.add('d-none');
                }
            }
        }
        
        // Show results container
        resultsContainer.classList.remove('d-none');
    }

    function copyCodeToClipboard() {
        const code = extractedCode.textContent;
        
        if (!code || code === 'No code extracted') {
            showToast('No code to copy', 'error');
            return;
        }
        
        navigator.clipboard.writeText(code)
            .then(() => {
                showToast('Code copied to clipboard', 'success');
            })
            .catch(err => {
                console.error('Could not copy text: ', err);
                showToast('Failed to copy code', 'error');
            });
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center';
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        let bgColor = 'bg-info';
        let icon = 'info-circle';
        
        switch (type) {
            case 'success':
                bgColor = 'bg-success';
                icon = 'check-circle';
                break;
            case 'error':
                bgColor = 'bg-danger';
                icon = 'exclamation-circle';
                break;
            case 'warning':
                bgColor = 'bg-warning';
                icon = 'exclamation-triangle';
                break;
        }
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${icon} me-2"></i>${message}
                </div>
                <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast, {
            autohide: true,
            delay: 3000
        });
        
        bsToast.show();
        
        // Remove toast from DOM after it's hidden
        toast.addEventListener('hidden.bs.toast', function() {
            toast.remove();
        });
    }
}); 