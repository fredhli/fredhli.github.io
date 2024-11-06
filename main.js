document.addEventListener("DOMContentLoaded", function () {
    const siteName = document.querySelector(".md-header__title");
    if (siteName) {
        const link = document.createElement("a");
        link.href = "/";
        link.style.textDecoration = "none";
        link.style.color = "inherit";

        const boldPart = document.createElement("strong");
        boldPart.textContent = "Fred";

        const normalPart = document.createTextNode(" Houze Li");

        link.appendChild(boldPart);
        link.appendChild(normalPart);

        siteName.textContent = "";
        siteName.appendChild(link);
    }
});

document.addEventListener("DOMContentLoaded", function () {
    const images = document.querySelectorAll(".photo-grid img, .gallery img, .single-picture");
    const overlay = document.createElement("div");
    overlay.classList.add("fullscreen-overlay");
    document.body.appendChild(overlay);

    const overlayImage = document.createElement("img");
    overlay.appendChild(overlayImage);

    const closeButton = document.createElement("button");
    closeButton.classList.add("close-button");
    closeButton.innerHTML = "&times;"; // "×" 字符
    overlay.appendChild(closeButton);

    images.forEach((img) => {
        img.addEventListener("click", () => {
            overlayImage.src = img.src;
            overlay.style.display = "flex";
        });
    });

    closeButton.addEventListener("click", () => {
        overlay.style.display = "none";
    });
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            overlay.style.display = "none";
        }
    });
});

