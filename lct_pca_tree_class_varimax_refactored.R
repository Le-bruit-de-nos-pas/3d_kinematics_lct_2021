

library(readr)
library(ggplot2)
library(caret)
library(C50)
library(ggfortify)
library(dplyr)
library(tidyr)
library(stringr)
library(viridis)

# --- CONFIG ----------------------------------------------------------------
CFG <- list(
  input_best_on = "Best_ON_Control.csv",
  input_all = "allnewpatients.csv",
  input_best_on_off = "BestON_OFF_DEC_8.csv",
  out_dir = "../out",
  pca_cols = 1:56,
  group_col = "Group",
  plot_width = 8,
  plot_height = 6,
  save_plots = TRUE,
  run_stats = TRUE,
  seed = 4
)

if (!dir.exists(CFG$out_dir)) dir.create(CFG$out_dir, recursive = TRUE)

# --- Helpers ----------------------------------------------------------------
load_csv <- function(path, sep = ";", dec = ",", na.strings = c("", " ", "NA")) {
  df <- tryCatch(read.csv(path, sep = sep, dec = dec, na.strings = na.strings, header = TRUE, stringsAsFactors = FALSE),
                 error = function(e) stop("Failed to read: ", path, " -> ", e$message))
  return(df)
}

run_pca <- function(df, cols = CFG$pca_cols, center = TRUE, scale = TRUE) {
  mat <- df[, cols]
  matnum <- suppressWarnings(as.data.frame(lapply(mat, as.numeric)))
  pca <- prcomp(matnum, center = center, scale. = scale)
  return(pca)
}

save_pca_autoplot <- function(pca, df, colour_col = CFG$group_col, filename = "pca_autoplot.svg") {
  plot_obj <- autoplot(pca, data = df, colour = colour_col, shape = colour_col) + theme_minimal()
  if (CFG$save_plots) ggsave(filename = file.path(CFG$out_dir, filename), plot = plot_obj, width = 7, height = 6)
  return(plot_obj)
}

train_c50 <- function(train_df, label_col = "label") {
  formula <- as.formula(paste(label_col, "~ ."))
  model <- C5.0(formula, data = train_df)
  return(model)
}

evaluate_model <- function(model, valid_df, label_col = "label") {
  preds <- predict(model, valid_df)
  acc <- mean(preds == valid_df[[label_col]], na.rm = TRUE)
  return(list(preds = preds, accuracy = acc))
}

varimax_rotation <- function(pca, ncomp = 12) {
  rot <- varimax(pca$rotation[, 1:ncomp])
  return(rot)
}

# Templated plotting: saves one plot per variable index/column with consistent theme
plot_variable_by_group <- function(df, group_col, var_index, var_label = NULL, filename_prefix = "varplot") {
  var_label <- var_label %||% paste0("V", var_index)
  p <- ggplot(df, aes_string(x = group_col, y = sprintf("df[, %d]", var_index), fill = group_col)) +
    geom_violin(alpha = 0.7, position = position_dodge(width = .75), size = 0.6, color = "black", show.legend = FALSE) +
    geom_boxplot(notch = TRUE, outlier.size = NA, color = "black", width = 0.12, alpha = 0.7, show.legend = FALSE) +
    geom_point(shape = 21, position = position_jitterdodge(), size = 2, color = "black", alpha = 0.8, show.legend = FALSE) +
    theme_minimal() +
    xlab(NULL) + ylab(var_label) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

  fname <- file.path(CFG$out_dir, paste0(filename_prefix, "_col", var_index, ".svg"))
  if (CFG$save_plots) ggsave(fname, plot = p, width = CFG$plot_width, height = CFG$plot_height)
  return(p)
}

# run repeated non-parametric tests simple wrappers
run_kruskal_loop <- function(df, start_col = 1, end_col = 56, group_col = CFG$group_col) {
  results <- list()
  for (i in seq(start_col, end_col)) {
    formula <- as.formula(paste0(sprintf("df[, %d]", i), " ~ ", group_col))
    # using kruskal.test on the column
    res <- tryCatch(kruskal.test(stats::as.formula(paste0("V~G")), data = (function(){G <- df[[group_col]]; V <- df[,i]; data.frame(G,V)})()), error = function(e) e)
    results[[paste0("col", i)]] <- res
  }
  return(results)
}

# --- Main pipeline ---------------------------------------------------------
main <- function() {
  message("Loading data files...")
  best_on <- load_csv(CFG$input_best_on)
  all_pat <- load_csv(CFG$input_all)
  best_off <- load_csv(CFG$input_best_on_off)

  # ensure group factor ordering if present
  if (CFG$group_col %in% names(all_pat)) {
    all_pat[[CFG$group_col]] <- factor(all_pat[[CFG$group_col]], levels = unique(all_pat[[CFG$group_col]]))
  }

  # PCA and autoplot
  message("Running PCA on selected columns...")
  pca <- run_pca(best_on, CFG$pca_cols)
  save_pca_autoplot(pca, best_on, filename = "pca_autoplot_best_on.svg")

  # preProcess PCA via caret used in original: create principal components and split
  set.seed(CFG$seed)
  pp <- preProcess(best_on[, CFG$pca_cols], method = c("pca"))
  PC <- predict(pp, best_on[, CFG$pca_cols])
  tr <- cbind(PC, label = best_on[, ncol(best_on)])
  tr$label <- as.factor(tr$label)

  idx <- createDataPartition(tr$label, p = 0.7, list = FALSE)
  traindata <- tr[idx, ]
  validdata <- tr[-idx, ]

  # Train C5.0 model
  message("Training C5.0 model...")
  model <- train_c50(traindata, label_col = "label")
  saveRDS(model, file = file.path(CFG$out_dir, "c50_model.rds"))

  eval <- evaluate_model(model, validdata, label_col = "label")
  cat("Validation accuracy:", eval$accuracy, "\n")
  write.csv(data.frame(accuracy = eval$accuracy), file = file.path(CFG$out_dir, "c50_accuracy.csv"), row.names = FALSE)

  # varimax rotation and saving top loadings
  message("Computing varimax rotation on first 12 PCs...")
  rot <- varimax_rotation(pca, ncomp = 12)
  loadings_df <- as.data.frame(rot$loadings)
  write.csv(loadings_df, file = file.path(CFG$out_dir, "varimax_loadings.csv"), row.names = TRUE)

  # Templated plotting for repeated ggplot blocks (BestONvsOFF17patients style) -- user files may vary
  message("Generating templated variable-by-group plots for all_pat and best_off datasets...")
  # choose a dataset and column ranges to plot; here we infer number of columns
  datasets <- list(all_pat = all_pat, best_off = best_off)
  for (dname in names(datasets)) {
    d <- datasets[[dname]]
    ncol_d <- ncol(d)
    # choose range to plot (safe guard)
    endcol <- min(ncol_d, 64)
    for (i in seq(1, endcol)) {
      try({
        plot_variable_by_group(d, CFG$group_col, i, var_label = names(d)[i], filename_prefix = paste0(dname, "_var"))
      }, silent = TRUE)
    }
  }

  # Optionally run non-parametric tests (kruskal/wilcox) similar to original loops
  if (CFG$run_stats) {
    message("Running quick Kruskal tests for all_pat (columns 1..56)...")
    kruskal_res <- run_kruskal_loop(all_pat, start_col = 1, end_col = min(56, ncol(all_pat)), group_col = CFG$group_col)
    saveRDS(kruskal_res, file = file.path(CFG$out_dir, "kruskal_results.rds"))
  }

  message("Finished refactored pipeline. Outputs in:", CFG$out_dir)
}

# Run when executed directly
if (identical(environment(), globalenv())) {
  tryCatch(main(), error = function(e) message("Pipeline failed: ", e$message))
}
