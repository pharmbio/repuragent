'''Injected JavaScript.

Three jobs, none of which Gradio can express declaratively:

* **The action bus.** The sidebar is one `gr.HTML` block, so clicking a card or a
  delete button has no Gradio event to fire. Clicks are serialized into a hidden
  textbox whose `change` event a Python handler listens to.
* **Downloads via fetch.** A signed link opened directly would navigate the tab on
  an error response; fetching it into a blob keeps failures in the console and gives
  the file its real name.
* **Pinning the light theme.** These components have no dark variant, and the
  browser's preference used to leak through into half-styled panels.
'''

from __future__ import annotations

CONVERSATION_SCRIPT = """
<script>
(function () {
    function findBus() {
        const element = document.getElementById("conversation-action-bus");
        if (!element) return null;
        if (element.matches && element.matches("textarea, input")) return element;
        return element.querySelector ? element.querySelector("textarea, input") : null;
    }

    function sendAction(payload) {
        const bus = findBus();
        if (!bus) return;
        // The timestamp makes each action a distinct value, so clicking the same
        // card twice still fires `change`.
        bus.value = JSON.stringify(Object.assign({ ts: Date.now() }, payload || {}));
        bus.dispatchEvent(new Event("input", { bubbles: true }));
        bus.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function enforceLightTheme() {
        const root = document.documentElement;
        if (root) {
            root.style.colorScheme = "light";
            root.classList.remove("dark");
            root.classList.add("light");
        }
        if (document.body) {
            document.body.style.colorScheme = "light";
            document.body.classList.remove("dark");
            document.body.classList.add("light");
        }
        try {
            const url = new URL(window.location.href);
            if (url.searchParams.get("__theme") !== "light") {
                url.searchParams.set("__theme", "light");
                window.history.replaceState(window.history.state, "", url.toString());
            }
        } catch (error) {
            console.warn("Unable to pin light theme", error);
        }
    }

    function observeThemeLock() {
        const root = document.documentElement;
        if (!root || root.dataset.lightThemeLocked === "1") return;
        root.dataset.lightThemeLocked = "1";
        const sync = () => window.requestAnimationFrame(enforceLightTheme);
        const observer = new MutationObserver(sync);
        observer.observe(root, { attributes: true, attributeFilter: ["class"] });
        if (document.body) {
            observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });
        }
        window.addEventListener("popstate", sync);
    }

    async function triggerDownload(anchor) {
        const url = anchor.getAttribute("href");
        if (!url) return;
        anchor.dataset.downloading = "1";
        try {
            const response = await fetch(url, { credentials: "same-origin" });
            if (!response.ok) throw new Error("HTTP " + response.status);
            const blob = await response.blob();
            const filename =
                anchor.getAttribute("data-file-name") || anchor.textContent.trim() || "download";
            const blobUrl = window.URL.createObjectURL(blob);
            const temp = document.createElement("a");
            temp.href = blobUrl;
            temp.download = filename;
            document.body.appendChild(temp);
            temp.click();
            window.setTimeout(function () {
                document.body.removeChild(temp);
                window.URL.revokeObjectURL(blobUrl);
            }, 0);
        } catch (error) {
            console.error("Download failed", error);
            window.open(url, "_blank", "noopener");
        } finally {
            delete anchor.dataset.downloading;
        }
    }

    function bindHandlers() {
        const root = document.getElementById("conversation-list-root");
        if (!root) return;

        root.querySelectorAll("summary").forEach(function (summary) {
            if (summary.dataset.repBound === "1") return;
            summary.dataset.repBound = "1";
            summary.addEventListener("click", function (event) {
                if (event.target && event.target.closest("[data-delete-thread]")) return;
                const card = summary.closest("details");
                if (!card) return;
                const threadId = card.getAttribute("data-thread-id");
                if (threadId) sendAction({ type: "activate", thread_id: threadId });
            });
        });

        root.querySelectorAll("[data-delete-thread]").forEach(function (button) {
            if (button.dataset.repBound === "1") return;
            button.dataset.repBound = "1";
            button.addEventListener("click", function (event) {
                event.preventDefault();
                event.stopPropagation();
                const threadId = button.getAttribute("data-delete-thread");
                const message = button.getAttribute("data-confirm-message");
                if (!threadId) return;
                if (message && !window.confirm(message)) return;
                sendAction({ type: "delete", thread_id: threadId });
            });
        });

        root.querySelectorAll("[data-download-link]").forEach(function (link) {
            if (link.dataset.repDownloadBound === "1") return;
            link.dataset.repDownloadBound = "1";
            link.addEventListener("click", function (event) {
                event.preventDefault();
                event.stopPropagation();
                if (link.dataset.downloading === "1") return;
                triggerDownload(link);
            });
        });
    }

    function initPartnerSlider(slider) {
        if (!slider || slider.dataset.sliderInitialized === "1") return;
        const viewport = slider.querySelector(".partner-slider__viewport");
        const track = slider.querySelector(".partner-slider__track");
        const cards = Array.from(slider.querySelectorAll(".partner-logo-card"));
        const dots = slider.querySelector(".partner-slider__dots");
        if (!viewport || !track || !cards.length || !dots) return;
        const state = { index: 0, perSlide: 1, total: 1 };

        function applyTransform() {
            const width = viewport.getBoundingClientRect().width || 1;
            track.style.transform = "translateX(-" + state.index * width + "px)";
        }

        function goTo(index) {
            state.index = Math.max(0, Math.min(index, state.total - 1));
            applyTransform();
            renderDots();
        }

        function renderDots() {
            dots.innerHTML = "";
            if (state.total <= 1) {
                dots.style.display = "none";
                return;
            }
            dots.style.display = "flex";
            for (let i = 0; i < state.total; i += 1) {
                const dot = document.createElement("button");
                dot.type = "button";
                dot.className = "partner-slider__dot" + (i === state.index ? " is-active" : "");
                dot.setAttribute("aria-label", "Show partner group " + (i + 1));
                dot.addEventListener("click", goTo.bind(null, i));
                dots.appendChild(dot);
            }
        }

        function recalc() {
            const width = viewport.getBoundingClientRect().width || 1;
            const sample = cards[0].getBoundingClientRect().width || 1;
            const styles = window.getComputedStyle(track);
            const gap = parseFloat(styles.columnGap || styles.gap || "16") || 16;
            state.perSlide = Math.max(1, Math.floor((width + gap) / (sample + gap)));
            state.total = Math.max(1, Math.ceil(cards.length / state.perSlide));
            state.index = Math.min(state.index, state.total - 1);
            renderDots();
            applyTransform();
        }

        const requestRecalc = function () { window.requestAnimationFrame(recalc); };
        if (window.ResizeObserver) {
            new ResizeObserver(requestRecalc).observe(viewport);
        } else {
            window.addEventListener("resize", requestRecalc);
        }
        requestRecalc();
        slider.dataset.sliderInitialized = "1";
    }

    function ensureReady() {
        enforceLightTheme();
        observeThemeLock();
        bindHandlers();
        document.querySelectorAll("[data-partner-slider]").forEach(initPartnerSlider);
    }

    ensureReady();
    // Gradio replaces DOM subtrees on every update, so handlers must be re-bound.
    // The `data-*Bound` guards keep this idempotent.
    new MutationObserver(function () {
        window.requestAnimationFrame(ensureReady);
    }).observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

__all__ = ["CONVERSATION_SCRIPT"]
