// Wait for the DOM to be fully loaded before running the script
document.addEventListener('DOMContentLoaded', (event) => {

    // --- Function to update the file name display ---
    const setupFileInput = (inputId, displayId) => {
        const fileInput = document.getElementById(inputId);
        const fileNameDisplay = document.getElementById(displayId);

        if (fileInput) {
            fileInput.addEventListener('change', function(e) {
                // Get the file name from the input, or show default text if no file is chosen
                const fileName = e.target.files[0] ? e.target.files[0].name : 'No file chosen';
                if (fileNameDisplay) {
                    fileNameDisplay.textContent = fileName;
                }
            });
        }
    };

    // --- Setup file inputs ---
    setupFileInput('source_file', 'source-file-name');
    setupFileInput('suspect_file', 'suspect-file-name');

    // --- Handle form submission ---
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            // Show the loading spinner and hide the submit button
            const submitButton = document.getElementById('submit-button');
            const loadingSpinner = document.getElementById('loading-spinner');
            if (submitButton) submitButton.classList.add('hidden');
            if (loadingSpinner) loadingSpinner.classList.remove('hidden');

            // Hide any previous results while processing new ones
            const resultsSection = document.getElementById('results-section');
            if (resultsSection) {
                resultsSection.style.display = 'none';
            }
        });
    }
});
