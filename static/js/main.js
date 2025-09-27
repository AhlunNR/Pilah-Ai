// Setup drag and drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
uploadArea.addEventListener('drop', handleDrop, false);

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
    handleFiles(files);
}

fileInput.addEventListener('change', function(e) {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            displayImagePreview(file);
        }
    }
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        
        // Scroll to preview
        previewContainer.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(file);
}

// Integrasi dengan Flask backend yang sudah ada
async function analyzeImage() {
    const loading = document.getElementById('loading');
    const resultCard = document.getElementById('resultCard');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('fileInput');
    
    if (!fileInput.files || !fileInput.files[0]) {
        alert('Pilih gambar terlebih dahulu!');
        return;
    }
    
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Menganalisis...';
    loading.style.display = 'block';
    
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Terjadi kesalahan');
        }
        
        // Convert backend response to frontend format
        const result = {
            name: getLabelDisplayName(data.label),
            category: data.category,
            fact: data.fact
        };
        
        const confidence = Math.round(data.confidence * 100);
        
        displayResult(result, confidence);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    } finally {
        // Hide loading
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analisis Sampah';
    }
}

// Fungsi untuk mengkonversi label backend ke nama display yang lebih user-friendly
function getLabelDisplayName(label) {
    const labelMapping = {
        'plastic': 'Sampah Plastik',
        'metal': 'Sampah Logam/Kaleng',
        'paper': 'Kertas',
        'cardboard': 'Kardus/Karton',
        'trash': 'Sampah Umum'
    };
    return labelMapping[label] || label;
}

function displayResult(result, confidence) {
    const resultCard = document.getElementById('resultCard');
    const itemName = document.getElementById('itemName');
    const categoryInstruction = document.getElementById('categoryInstruction');
    const factContent = document.getElementById('factContent');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    itemName.textContent = result.name;
    categoryInstruction.innerHTML = `Masukkan ke kategori: <strong>${result.category}</strong>`;
    factContent.textContent = result.fact;
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = confidence + '%';
    }, 100);
    confidenceText.textContent = `Akurasi: ${confidence}%`;
    
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth' });
}

function resetAnalysis() {
    const uploadCard = document.getElementById('uploadCard');
    const resultCard = document.getElementById('resultCard');
    const previewContainer = document.getElementById('previewContainer');
    
    // Reset form
    fileInput.value = '';
    previewContainer.style.display = 'none';
    resultCard.style.display = 'none';
    
    // Scroll back to upload area
    uploadCard.scrollIntoView({ behavior: 'smooth' });
}

// Bottom navigation functionality
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Remove active class from all items
        document.querySelectorAll('.nav-item').forEach(navItem => {
            navItem.classList.remove('active');
        });
        
        // Add active class to clicked item
        this.classList.add('active');
    });
});