/**
 * AI SIGN LANGUAGE TRANSLATOR - MAIN UI & NAVIGATION
 * Handles mobile hamburger drawer, dark/light theme switching,
 * smooth scrolling, animations, and dictionary filtering/modals.
 */

document.addEventListener("DOMContentLoaded", () => {
    console.log("[SignTranslate AI] Main script initialized.");

    // ============================================================
    // 1. MOBILE MENU TOGGLE
    // ============================================================
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const mobileNavDrawer = document.getElementById("mobileNavDrawer");

    if (mobileMenuBtn && mobileNavDrawer) {
        mobileMenuBtn.addEventListener("click", () => {
            const isOpen = mobileNavDrawer.classList.toggle("open");
            mobileMenuBtn.innerHTML = isOpen ? "✕" : "☰";
            mobileMenuBtn.setAttribute("aria-expanded", isOpen);
        });

        // Close on link click
        const drawerLinks = mobileNavDrawer.querySelectorAll("a");
        drawerLinks.forEach(link => {
            link.addEventListener("click", () => {
                mobileNavDrawer.classList.remove("open");
                mobileMenuBtn.innerHTML = "☰";
                mobileMenuBtn.setAttribute("aria-expanded", "false");
            });
        });
    }

    // ============================================================
    // 2. THEME SWITCHER (DARK / LIGHT)
    // ============================================================
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const savedTheme = localStorage.getItem("signtranslate_theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme");
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("signtranslate_theme", newTheme);
            updateThemeIcon(newTheme);
        });
    }

    function updateThemeIcon(theme) {
        if (!themeToggleBtn) return;
        themeToggleBtn.innerHTML = theme === "dark" ? "☀️" : "🌙";
        themeToggleBtn.setAttribute("title", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
    }

    // ============================================================
    // 3. NAVBAR SCROLL EFFECT
    // ============================================================
    const navbar = document.querySelector(".navbar");
    if (navbar) {
        window.addEventListener("scroll", () => {
            if (window.scrollY > 40) {
                navbar.classList.add("navbar-scrolled");
            } else {
                navbar.classList.remove("navbar-scrolled");
            }
        });
    }

    // ============================================================
    // 4. SMOOTH SCROLLING FOR HASH LINKS
    // ============================================================
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener("click", function(e) {
            const href = this.getAttribute("href");
            if (href && href !== "#") {
                const target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({ behavior: "smooth" });
                }
            }
        });
    });

    // ============================================================
    // 5. CURRENT YEAR
    // ============================================================
    const currentYear = new Date().getFullYear();
    document.querySelectorAll(".current-year").forEach(el => {
        el.textContent = currentYear;
    });

    // ============================================================
    // 6. SIGN DICTIONARY SEARCH, FILTERS & MODAL
    // ============================================================
    const searchInput = document.getElementById("dictionarySearchInput");
    const filterTabs = document.querySelectorAll(".filter-tab");
    const signCards = document.querySelectorAll(".sign-card");
    const modalBackdrop = document.getElementById("dictionaryModal");
    const modalCloseBtn = document.getElementById("modalCloseBtn");

    // Category filter tabs
    if (filterTabs.length > 0) {
        filterTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                filterTabs.forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                filterDictionary();
            });
        });
    }

    // Search input
    if (searchInput) {
        searchInput.addEventListener("input", () => {
            filterDictionary();
        });
    }

    function filterDictionary() {
        if (!signCards || signCards.length === 0) return;

        const query = searchInput ? searchInput.value.toLowerCase().trim() : "";
        const activeTab = document.querySelector(".filter-tab.active");
        const category = activeTab ? activeTab.getAttribute("data-category") : "All";

        signCards.forEach(card => {
            const name = (card.getAttribute("data-name") || "").toLowerCase();
            const sign = (card.getAttribute("data-sign") || "").toLowerCase();
            const cardCat = card.getAttribute("data-category") || "";

            const matchesSearch = !query || name.includes(query) || sign.includes(query);
            const matchesCategory = (category === "All" || cardCat.toLowerCase() === category.toLowerCase());

            if (matchesSearch && matchesCategory) {
                card.style.display = "flex";
            } else {
                card.style.display = "none";
            }
        });
    }

    // Card click opens modal
    if (signCards.length > 0 && modalBackdrop) {
        signCards.forEach(card => {
            card.addEventListener("click", () => {
                const sign = card.getAttribute("data-sign") || "";
                const name = card.getAttribute("data-name") || sign;
                const desc = card.getAttribute("data-desc") || "";
                const cat = card.getAttribute("data-category") || "";
                const icon = card.querySelector(".sign-card-img-wrap") ? card.querySelector(".sign-card-img-wrap").innerText : "🤟";

                document.getElementById("modalSignIcon").innerText = icon;
                document.getElementById("modalSignTitle").innerText = `${name} (${sign})`;
                document.getElementById("modalSignCategory").innerText = cat;
                document.getElementById("modalSignDescription").innerText = desc;

                modalBackdrop.classList.add("open");
            });
        });
    }

    if (modalCloseBtn && modalBackdrop) {
        modalCloseBtn.addEventListener("click", () => {
            modalBackdrop.classList.remove("open");
        });

        modalBackdrop.addEventListener("click", (e) => {
            if (e.target === modalBackdrop) {
                modalBackdrop.classList.remove("open");
            }
        });
    }
});
