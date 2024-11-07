document.addEventListener("DOMContentLoaded", function () {
    initializeSiteName();
    initializeImageZoom();
    observePageChanges();
});

// Initialize site name customization
function initializeSiteName() {
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
}

// Initialize image zoom functionality
function initializeImageZoom() {
    const images = document.querySelectorAll(".photo-grid img, .gallery img, .single-picture");

    if (images.length > 0) {
        let overlay = document.querySelector(".fullscreen-overlay");

        if (!overlay) {
            overlay = document.createElement("div");
            overlay.classList.add("fullscreen-overlay");
            document.body.appendChild(overlay);

            const overlayImage = document.createElement("img");
            overlay.appendChild(overlayImage);

            const closeButton = document.createElement("button");
            closeButton.classList.add("close-button");
            closeButton.innerHTML = "&times;";
            overlay.appendChild(closeButton);

            function closeOverlay() {
                overlay.style.display = "none";
            }

            closeButton.addEventListener("click", (event) => {
                event.stopPropagation();
                closeOverlay();
            });

            overlay.addEventListener("click", (event) => {
                if (event.target === overlay) {
                    closeOverlay();
                }
            });
        }

        images.forEach((img) => {
            img.addEventListener("click", () => {
                const overlayImage = overlay.querySelector("img");
                overlayImage.src = img.src;
                overlay.style.display = "flex";
            });
        });
    }
}

// Observe DOM changes to reinitialize image zoom on navigation
function observePageChanges() {
    const observer = new MutationObserver(() => {
        initializeImageZoom();
    });

    observer.observe(document.body, { childList: true, subtree: true });
}
