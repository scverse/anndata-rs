/**
 * Fit the Read the Docs addons into the theme.
 */

/**
 * Add CSS to the shadow roots of an addon’s elements, existing and future
 * (e.g. `<readthedocs-search>` is only created once the addon data arrives).
 * @param {string} name
 * @param {string} css
 */
function restyle(name, css) {
    const sheet = new CSSStyleSheet()
    sheet.replaceSync(css)
    /** @param {ShadowRoot} root */
    const adopt = (root) => {
        root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet]
    }
    customElements.whenDefined(name).then((cls) => {
        for (const { shadowRoot } of document.querySelectorAll(name)) {
            adopt(shadowRoot)
        }
        const { createRenderRoot } = cls.prototype
        cls.prototype.createRenderRoot = function () {
            const root = createRenderRoot.call(this)
            adopt(root)
            return root
        }
    })
}

// The RTD logos are <img>s of single-colour SVGs (light in the flyout, dark in the search),
// so they vanish on one of our backgrounds: filter both to a mid grey (like RTD’s icons) that works on either.

// flyout.css hardcodes things no part/variable can override:
// `.container { position: fixed }`, the version icon’s grey, and the logo
restyle(
    "readthedocs-flyout",
    `
    .container { position: static; max-width: none; max-height: none; }
    header > span svg.icon { color: var(--readthedocs-flyout-color); }
    header > img.logo { filter: brightness(0.5); }
    `,
)
restyle(
    "readthedocs-search",
    `.credits img { filter: invert(1) brightness(0.6); }`,
)

// Search: see https://docs.readthedocs.com/platform/stable/addons.html#integrate-with-search-as-you-type
document.addEventListener("readthedocs-addons-data-ready", (event) => {
    const { addons } = event.detail.data()
    /** @type {HTMLDialogElement | null} */
    const theme_dialog = document.querySelector("dialog#pst-search-dialog")

    if (!addons.search?.enabled || !theme_dialog) {
        return // RTD option or `disable_search` theme option
    }

    // pydata’s search button and Ctrl/⌘+K both open its dialog via `showModal()`: show RTD’s search instead.
    theme_dialog.showModal = () => {
        const shown = document.querySelector("readthedocs-search")?.show
        document.dispatchEvent(
            new CustomEvent(`readthedocs-search-${shown ? "hide" : "show"}`),
        )
    }
})
