BASE.SIZE <- 10
INCH.PER.CM <- 1 / 2.54
TEXTWIDTH <- 13.07245 * INCH.PER.CM
HEIGHT <- 19.42224 * INCH.PER.CM
OUTDIR_PDF <- "img-pdf/"
OUTDIR_TIKZ <- "img-tikz/"
COLOURS.LIST <- c(
  "black",
  "#E69F00",
  "#009371",
  "#999999",
  "#beaed4",
  "#ed665a",
  "#1f78b4",
  "#CC79A7"
)
POINT.SIZE <- 1
LINE.SIZE <- 1

theme_paper <- function() {
  return(
    theme_bw(base_size = BASE.SIZE) +
      theme(
        strip.background = element_rect(colour = "black", fill = "white"),
        axis.title.x = element_text(size = BASE.SIZE),
        axis.title.y = element_text(size = BASE.SIZE),
        legend.title = element_text(size = BASE.SIZE),
        legend.position = "top",
        plot.margin = unit(c(0, 0, 0, 0), "cm"),
        legend.margin = margin(b = -2)
      )
  )
}

create_plot <- function(g, save_name, w, h, use_tikz = TRUE) {
  if (use_tikz) {
    if (!dir.exists(OUTDIR_TIKZ)) {
      dir.create(OUTDIR_TIKZ, recursive = TRUE)
    }
    tikz(
      str_c(OUTDIR_TIKZ, save_name, ".tex"),
      width = w,
      height = h
    )
  } else {
    if (!dir.exists(OUTDIR_PDF)) {
      dir.create(OUTDIR_PDF, recursive = TRUE)
    }
    pdf(
      str_c(OUTDIR_PDF, save_name, ".pdf"),
      width = w,
      height = h
    )
  }
  print(g)
  dev.off()
  if (!use_tikz) {
    print(str_c(
      "Created plot in ",
      getwd(),
      "/",
      OUTDIR_PDF,
      save_name,
      ".pdf"
    ))
  }
}
