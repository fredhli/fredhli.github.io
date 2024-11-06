document.addEventListener("DOMContentLoaded", function () {
    const images = document.querySelectorAll(".photo-grid img");
    const overlay = document.createElement("div");
    overlay.classList.add("fullscreen-overlay");
    document.body.appendChild(overlay);

    const overlayImage = document.createElement("img");
    overlay.appendChild(overlayImage);

    images.forEach((img) => {
        img.addEventListener("click", () => {
            overlayImage.src = img.src;
            overlay.style.display = "flex";
        });
    });

    overlay.addEventListener("click", () => {
        overlay.style.display = "none";
    });
});
