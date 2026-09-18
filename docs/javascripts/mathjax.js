// MathJax configuration for the documentation site.
//
// `pymdownx.arithmatex` runs in generic mode, so it has already rewritten the
// `$...$` and `$$...$$` of the Markdown sources into `\(...\)` and `\[...\]`
// inside `<span class="arithmatex">`. The dollar delimiters are deliberately
// not configured here: GitHub renders those itself, and the extension is what
// translates them for this site, so the same source renders in both places.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Typeset only what arithmatex marked up, and nothing else on the page.
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// The theme swaps pages in without a reload, so the new content has to be
// typeset again. `document$` is the theme's observable for that; when it is
// absent, MathJax's own load-time typesetting is all that is needed.
if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
