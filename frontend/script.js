// Display uploaded image as soon as it is selected
const fileInput = document.getElementById("file");
const uploadedImage = document.getElementById("uploaded-image");
const uploadedPlaceholder = document.querySelector(".uploaded-image-container .placeholder");

fileInput.addEventListener("change", (e) => {
    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = "block"; // Show the image
            uploadedPlaceholder.style.display = "none"; // Hide the placeholder
        };
        reader.readAsDataURL(fileInput.files[0]);
    }
});

// Handle form submission
document.getElementById("upload-form").onsubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);

    // Show loader and clear previous recolored images while processing
    const loader = document.getElementById("loader");
    const recoloredImagesContainer = document.getElementById("recolored-images");
    loader.style.display = "block";
    recoloredImagesContainer.innerHTML = ""; // Clear previous images

    // Send the image to the API
    const response = await fetch("http://127.0.0.1:8000/recolor-dress/", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    if (result.recolored_dress_images) {
        // Hide loader
        loader.style.display = "none";

        // Display all recolored images
        result.recolored_dress_images.forEach((imagePath) => {
            const imgElement = document.createElement("img");
            imgElement.src = `http://127.0.0.1:8000/${imagePath}`;
            imgElement.alt = "Recolored dress";
            imgElement.addEventListener("click", () => openModal(imgElement.src));
            recoloredImagesContainer.appendChild(imgElement);
        });
    } else {
        // Hide loader and display error
        loader.style.display = "none";
        alert(result.message || "An error occurred while processing the image.");
    }
};

// Full-screen modal functionality
const modal = document.getElementById("image-modal");
const modalImage = document.getElementById("modal-image");
const closeModal = document.querySelector(".modal .close");

function openModal(imageSrc) {
    modalImage.src = imageSrc;
    modal.style.display = "flex";
}

closeModal.onclick = () => {
    modal.style.display = "none";
};

window.onclick = (event) => {
    if (event.target === modal) {
        modal.style.display = "none";
    }
};