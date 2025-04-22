let showState = 0;
let currentAudio = null;
let currentImageIndex = 0;
const gridWidth = 6;
const gridHeight = 4;
let allData = [];
let visibleGenres = new Set();

window.onload = function() {
    fetch('/get_data')
        .then(response => response.json())
        .then(data => {
            allData = data.data;
            const uniqueGenres = data.genres;
            console.log(uniqueGenres);
            createGenreFilters(uniqueGenres);
            drawGrid(allData);
            drawImagesAndSquares(allData);
            setupKeyListeners(allData);
            setupToggleAllGenres();
        })
        .catch(error => console.error('Error fetching data:', error));
};

function createGenreFilters(genres) {
    const genreFilters = document.getElementById('genre-filters');
    genres.forEach(genre => {
        const label = document.createElement('label');
        label.innerHTML = `<input type="checkbox" class="genre-checkbox" value="${genre}" checked> ${genre}`;
        genreFilters.appendChild(label);
        visibleGenres.add(genre);
    });

    document.querySelectorAll('.genre-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateGenreVisibility);  // Update visibility on change
    });
}

function updateGenreVisibility() {
    const checkboxes = document.querySelectorAll('.genre-checkbox');
    visibleGenres.clear();
    checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
            visibleGenres.add(checkbox.value);
        }
    });

    // Update images based on selected genres
    const wrappers = document.querySelectorAll('.image-wrapper');
    wrappers.forEach(wrapper => {
        const trackIndex = wrapper.dataset.trackIndex;
        const trackGenre = allData[trackIndex].track_genre;
        if (visibleGenres.has(trackGenre)) {
            wrapper.style.display = 'block';
        } else {
            wrapper.style.display = 'none';
        }
    });

    // Display the count of selected genres
    document.getElementById('selected-genres').textContent = `Selected Genres: ${visibleGenres.size}`;
}

function setupToggleAllGenres() {
    const toggleAllButton = document.getElementById('toggle-all-genres');
    toggleAllButton.addEventListener('click', function() {
        const checkboxes = document.querySelectorAll('.genre-checkbox');
        const allChecked = [...checkboxes].every(checkbox => checkbox.checked);  // Check if all are selected

        // Toggle checkboxes
        checkboxes.forEach(checkbox => {
            checkbox.checked = !allChecked;
        });
        updateGenreVisibility();
    });
}

function drawGrid(data) {
    const container = document.getElementById('dot-container');

    const gridCellWidth = container.offsetWidth / gridWidth;
    const gridCellHeight = container.offsetHeight / gridHeight;

    // Initialize gridColors as an empty array with dimensions [gridHeight][gridWidth]
    let gridColors = Array.from({ length: gridHeight }, () => Array(gridWidth).fill(null));

    // Loop through data and fill gridColors based on gridWidth and gridHeight
    data.forEach(item => {
        const xGridIndex = Math.floor(item.x * gridWidth);
        const yGridIndex = Math.floor((1 - item.y) * gridHeight);
        const imageData = item.data[currentImageIndex];

        if (!gridColors[yGridIndex][xGridIndex]) {
            gridColors[yGridIndex][xGridIndex] = [];
        }

        gridColors[yGridIndex][xGridIndex].push(imageData.mean_color);
    });

    // Iterate through gridColors to calculate the average color for each grid cell
    gridColors.forEach((row, rowIndex) => {
        row.forEach((cellColors, colIndex) => {
            if (!cellColors) return;
            let avgColor = { r: 255, g: 255, b: 255 };
            if (cellColors.length) {
                const total = cellColors.reduce((acc, color) => ({
                    r: acc.r + color.r,
                    g: acc.g + color.g,
                    b: acc.b + color.b
                }), { r: 0, g: 0, b: 0 });
                avgColor = {
                    r: Math.floor(total.r / cellColors.length),
                    g: Math.floor(total.g / cellColors.length),
                    b: Math.floor(total.b / cellColors.length)
                };
            }

            // Create the grid cell and apply the calculated average color
            const gridCell = document.createElement('div');
            gridCell.classList.add('grid-cell');
            gridCell.style.width = `${gridCellWidth}px`;
            gridCell.style.height = `${gridCellHeight}px`;
            gridCell.style.left = `${colIndex * gridCellWidth}px`;
            gridCell.style.top = `${rowIndex * gridCellHeight}px`;
            gridCell.style.backgroundColor = `rgb(${avgColor.r}, ${avgColor.g}, ${avgColor.b})`;
            container.appendChild(gridCell);
        });
    });
}

function drawImagesAndSquares(data) {
    const container = document.getElementById('dot-container');
    data.forEach((item, trackIndex) => {
        const wrapper = document.createElement('div');
        wrapper.classList.add('image-wrapper');
        wrapper.dataset.trackIndex = trackIndex;
        const xPercent = item.x * 100;
        const yPercent = (1 - item.y) * 100;
        wrapper.style.left = `${xPercent}%`;
        wrapper.style.top = `${yPercent}%`;
        const image = document.createElement('img');
        image.classList.add('image', 'visible');
        image.src = item.data[currentImageIndex].image_path;
        wrapper.appendChild(image);
        wrapper.addEventListener('mouseenter', () => showPopup(trackIndex));
        wrapper.addEventListener('mouseleave', () => {
            document.getElementById('image-popup').style.display = 'none';
            if (currentAudio) {
                currentAudio.pause();
                currentAudio.currentTime = 0;
            }
        });
        wrapper.addEventListener('click', () => playAudio(item));
        container.appendChild(wrapper);
    });
    updateGenreVisibility();
}

function showPopup(trackIndex) {
    const popup = document.getElementById('image-popup');
    popup.style.display = 'block';

    const rect = document.querySelectorAll('.image-wrapper')[trackIndex].getBoundingClientRect();

    // Align the popup's top-right corner with the image's top-left corner
    popup.style.left = `${rect.left - popup.offsetWidth}px`;
    popup.style.top = `${rect.top + window.scrollY}px`;

    // Store the trackIndex in the popup element for later reference
    popup.dataset.trackIndex = trackIndex;

    updatePopupContent(trackIndex);
}

function updatePopupContent(trackIndex) {
    const currentData = allData[trackIndex];
    const imageData = currentData.data[currentImageIndex];

    document.getElementById('popup-track-name').innerHTML = `<strong>Track Name:</strong> ${currentData.track_name}`;
    document.getElementById('popup-artists').innerHTML = `<strong>Artists:</strong> ${currentData.artists}`;
    document.getElementById('popup-popularity').innerHTML = `<strong>Popularity:</strong> ${currentData.popularity}`;
    document.getElementById('popup-energy').innerHTML = `<strong>Energy:</strong> ${currentData.energy.toFixed(2)}`;
    document.getElementById('popup-track_genre').innerHTML = `<strong>Genre:</strong> ${currentData.track_genre}`;
    document.getElementById('popup-danceability').innerHTML = `<strong>Danceability:</strong> ${currentData.danceability.toFixed(2)}`;
    document.getElementById('popup-loudness').innerHTML = `<strong>Loudness:</strong> ${currentData.loudness.toFixed(2)}`;
    document.getElementById('popup-valence').innerHTML = `<strong>Valence:</strong> ${currentData.valence.toFixed(2)}`;

    document.getElementById('popup-image').src = imageData.image_path;
    document.getElementById('popup-prompt').innerHTML = `<strong>Prompt: </strong> ${imageData.prompt}`;
    document.getElementById('popup-revised-prompt').innerHTML = `<strong>Revised Prompt:</strong> ${imageData.revised_prompt}`;
}

function toggleGridAndImages() {
    const gridCells = document.querySelectorAll('.grid-cell');
    const images = document.querySelectorAll('.image-wrapper');

    showState = (showState + 1) % 2;

    if (showState === 0) {
        gridCells.forEach(cell => cell.style.display = 'none');
        images.forEach(image => image.style.display = 'block');
    } else if (showState === 1) {
        gridCells.forEach(cell => cell.style.display = 'block');
        images.forEach(image => image.style.display = 'none');
    }
}

function playAudio(item) {
    const player = document.getElementById('audio-player');
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    player.src = item.data[currentImageIndex].audio_path;
    player.play();
    currentAudio = player;
}

function setupKeyListeners(data) {
    document.addEventListener('keydown', function(event) {
        if (event.key === 't') {
            toggleGridAndImages();
        } else if (['1', '2', '3'].includes(event.key)) {
            currentImageIndex = parseInt(event.key) - 1;

            updateAllSmallImages(data);
            updateGridColors(data);

            // Ensure that if the popup is open, it shows the correct image based on trackIndex
            const popup = document.getElementById('image-popup');
            if (popup.style.display === 'block') {
                const activeTrackIndex = popup.dataset.trackIndex;

                if (activeTrackIndex) {
                    updatePopupContent(activeTrackIndex);
                }
            }
            toggleGridAndImages(); toggleGridAndImages(); // NOTE: This is a hacky solution that ensure the correct color/image toggle is shown after reload
        }
    });
}

function updateAllSmallImages(data) {
    const wrappers = document.querySelectorAll('.image-wrapper');
    wrappers.forEach(wrapper => {
        const trackIndex = wrapper.dataset.trackIndex;
        const image = wrapper.querySelector('.image');
        const imageData = data[trackIndex].data[currentImageIndex];
        if (imageData) {
            image.src = imageData.image_path;
            image.style.display = 'block';
        } else {
            image.style.display = 'none';
        }
    });
    updateGenreVisibility();
}

function updateGridColors(data) {
    const container = document.getElementById('dot-container');
    const gridCells = document.querySelectorAll('.grid-cell');
    const gridCellWidth = container.offsetWidth / gridWidth;
    const gridCellHeight = container.offsetHeight / gridHeight;

    // Populate the gridColors array based on the data
    gridCells.forEach(cell => cell.remove());
    let gridColors = Array.from({ length: gridHeight }, () => Array(gridWidth).fill(null));
    data.forEach(item => {
        const xGridIndex = Math.floor(item.x * gridWidth);
        const yGridIndex = Math.floor((1 - item.y) * gridHeight);
        const imageData = item.data[currentImageIndex];

        if (!imageData || !imageData.mean_color) {
            console.warn(`Image data missing for index ${currentImageIndex}`);
            return;
        }

        if (!gridColors[yGridIndex][xGridIndex]) {
            gridColors[yGridIndex][xGridIndex] = [];
        }

        gridColors[yGridIndex][xGridIndex].push(imageData.mean_color);
    });

    // Create grid cells and set their background colors
    gridColors.forEach((row, rowIndex) => {
        row.forEach((cellColors, colIndex) => {
            if (!cellColors) return;
            let avgColor = { r: 255, g: 255, b: 255 };
            if (cellColors.length) {
                const total = cellColors.reduce((acc, color) => ({
                    r: acc.r + color.r,
                    g: acc.g + color.g,
                    b: acc.b + color.b
                }), { r: 0, g: 0, b: 0 });
                avgColor = {
                    r: Math.floor(total.r / cellColors.length),
                    g: Math.floor(total.g / cellColors.length),
                    b: Math.floor(total.b / cellColors.length)
                };
            }

            // Create the grid cell and set its position and size
            const gridCell = document.createElement('div');
            gridCell.classList.add('grid-cell');
            gridCell.style.width = `${gridCellWidth}px`;
            gridCell.style.height = `${gridCellHeight}px`;
            gridCell.style.left = `${colIndex * gridCellWidth}px`;
            gridCell.style.top = `${rowIndex * gridCellHeight}px`;
            gridCell.style.backgroundColor = `rgb(${avgColor.r}, ${avgColor.g}, ${avgColor.b})`;
            container.appendChild(gridCell);
        });
    });
}