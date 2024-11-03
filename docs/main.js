document.addEventListener("DOMContentLoaded", function () {
    const siteName = document.querySelector(".md-header__title");
    if (siteName) {
        // 创建链接元素,并设置指向主页
        const link = document.createElement("a");
        link.href = "/";
        link.style.textDecoration = "none";
        link.style.color = "inherit";

        // 创建 "Fred" 和 "H. Li" 元素,分别设置不同的样式
        const boldPart = document.createElement("strong");
        boldPart.textContent = "Fred";

        const normalPart = document.createTextNode(" Houze Li");

        // 将加粗的 "Fred" 和正常的 "H. Li" 添加到链接中
        link.appendChild(boldPart);
        link.appendChild(normalPart);

        // 清空原始标题内容,并将新的链接插入标题中
        siteName.textContent = "";
        siteName.appendChild(link);
    }
});
