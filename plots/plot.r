#!/usr/bin/env Rscript

library(tidyverse)
library(tikzDevice)
library(scales)
source("layout.r")

options(
  tikzLatexPackages = c(
    getOption("tikzLatexPackages"),
    "\\usepackage{amsmath}"
  )
)

use_tikz <- TRUE
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1) {
  use_tikz <- FALSE
  POINT.SIZE <- 0.5
  LINE.SIZE <- 0.5
}

d <- read_csv("results_paper/zhou_trajectories.csv") |>
  mutate(deviation = abs(occupation_hw - occupation_theory)) |>
  pivot_longer(
    c(occupation_hw, occupation_theory),
    names_to = "model",
    values_to = "occupation",
  ) |>
  filter(!(model == "occupation_theory" & setting == "free knobs"))

d_dig <- read_csv("results_paper/zhou_digital_trajectories.csv") |>
  filter(steps %in% c(10, 50, 100)) |>
  mutate(
    deviation = abs(occupation_hw - occupation_theory),
    occupation = occupation_hw,
    setting = steps,
    model = "digital"
  )

common <- intersect(colnames(d), colnames(d_dig))
d_joined <- rbind(d[common], d_dig[common])

g <- ggplot(
  d_joined,
  aes(x = time_ms, y = occupation, colour = interaction(model, setting))
) +
  theme_paper() +
  geom_line(linewidth = LINE.SIZE) +
  scale_colour_manual(
    "Operator",
    values = COLOURS.LIST,
    breaks = c(
      "occupation_theory.prescribed knobs",
      "occupation_hw.free knobs",
      "occupation_hw.prescribed knobs",
      "digital.10",
      "digital.50",
      "digital.100"
    ),
    labels = c(
      "$\\hat{H}_\\text{IR2}$",
      "$\\hat{H}_\\text{sim}$ (free $\\Theta_\\text{sim}$)",
      "$\\hat{H}_\\text{sim}$ (prescribed $\\Theta_\\text{sim}$)",
      "$\\hat{U}_\\approx (n = 10)$",
      "$\\hat{U}_\\approx (n = 50)$",
      "$\\hat{U}_\\approx (n = 100)$"
    )
  ) +
  scale_x_continuous("$t$ [ms]") +
  scale_y_continuous("$\\langle \\hat{n}_\\text{matter}\\rangle$") +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

create_plot(
  g,
  "trajectory_occupation",
  1 * TEXTWIDTH,
  0.25 * HEIGHT,
  use_tikz
)

d <- d |>
  filter(model != "occupation_theory") |>
  pivot_longer(
    c(deviation, leakage, violation),
    names_to = "error_type",
    values_to = "error"
  )

d$error_type <- factor(
  d$error_type,
  levels = c("deviation", "leakage", "violation"),
  labels = c("Deviation", "Leakage", "Violation $\\eta$")
)

g <- ggplot(
  d,
  aes(x = time_ms, y = error, colour = setting)
) +
  theme_paper() +
  geom_line(linewidth = LINE.SIZE) +
  scale_colour_manual(
    "Hamiltonian",
    values = c(COLOURS.LIST[2], COLOURS.LIST[3]),
    breaks = c(
      "free knobs",
      "prescribed knobs"
    ),
    labels = c(
      "$\\hat{H}_\\text{sim}$ (free $\\Theta_\\text{sim}$)",
      "$\\hat{H}_\\text{sim}$ (prescribed $\\Theta_\\text{sim}$)"
    )
  ) +
  facet_grid(error_type ~ ., scales = "free_y") +
  scale_x_continuous("$t$ [ms]") +
  scale_y_continuous("Error") +
  guides(colour = guide_legend(nrow = 1))

create_plot(
  g,
  "trajectory_error",
  1 * TEXTWIDTH,
  0.35 * HEIGHT,
  use_tikz
)
