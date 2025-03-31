time_taken <- system.time({
  


# Load necessary libraries
library(data.table)
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)


setwd("/home/hastib/Breast_Cancer_Analysis")


# Wisconsin Data
wisconsin_test_data <- fread("wisconsin_data_filtered.csv")[-createDataPartition(fread("wisconsin_data_filtered.csv")$y, p = 0.8, list = FALSE), ]
wisconsin_rf_model <- readRDS("rf_model_wisconsin.rds")
wisconsin_features <- names(wisconsin_rf_model$forest$xlevels)
cat("Wisconsin model features:", paste(wisconsin_features, collapse = ", "), "\n")
cat("Columns in wisconsin_test_data:", paste(colnames(wisconsin_test_data), collapse = ", "), "\n")
required_wisconsin <- setdiff(wisconsin_features, "y")
for (col in required_wisconsin) {
  if (!col %in% colnames(wisconsin_test_data)) {
    wisconsin_test_data[[col]] <- NA
    cat("Added dummy column to wisconsin_test_data:", col, "\n")
  }
}
wisconsin_probs <- predict(wisconsin_rf_model, newdata = wisconsin_test_data, type = "prob")[, "M"]

# Histopathology Data
histo_features <- readRDS("histopathology_features.rds")
histo_test_data <- histo_features[-createDataPartition(histo_features$label, p = 0.8, list = FALSE), ]
histo_rf_model <- readRDS("rf_model_histopathology.rds")
histo_features_required <- names(histo_rf_model$forest$xlevels)
cat("Histopathology model features:", paste(histo_features_required, collapse = ", "), "\n")
cat("Columns in histo_test_data:", paste(colnames(histo_test_data), collapse = ", "), "\n")
required_histo <- setdiff(histo_features_required, "label")
for (col in required_histo) {
  if (!col %in% colnames(histo_test_data)) {
    histo_test_data[[col]] <- NA
    cat("Added dummy column to histo_test_data:", col, "\n")
  }
}
histo_probs <- predict(histo_rf_model, newdata = histo_test_data, type = "prob")[, "1"]

# Gene Data
gene_test_data <- readRDS("gene_test_data.rds")
gene_rf_model <- readRDS("rf_model_gene.rds")
gene_features <- names(gene_rf_model$forest$xlevels)
cat("Gene model features:", paste(gene_features, collapse = ", "), "\n")
cat("Columns in gene_test_data:", paste(colnames(gene_test_data), collapse = ", "), "\n")
original_colnames <- colnames(gene_test_data)
cleaned_colnames <- make.names(original_colnames, unique = TRUE)
colnames(gene_test_data) <- cleaned_colnames
required_gene <- setdiff(gene_features, "type")
gene_test_data_subset <- gene_test_data[, intersect(colnames(gene_test_data), c("type", required_gene)), drop = FALSE]
for (col in required_gene) {
  if (!col %in% colnames(gene_test_data_subset)) {
    gene_test_data_subset[[col]] <- NA
    cat("Added dummy column to gene_test_data_subset:", col, "\n")
  }
}
gene_probs <- predict(gene_rf_model, newdata = gene_test_data_subset, type = "prob")
cat("Gene model class probabilities:", paste(colnames(gene_probs), collapse = ", "), "\n")


aggressive_subtypes <- c("Basal", "HER")
available_subtypes <- intersect(aggressive_subtypes, colnames(gene_probs))
if (length(available_subtypes) == 0) {
  cat("Warning: None of 'Basal', 'HER' found in gene_probs. Using all probabilities.\n")
  gene_malignancy_prob <- rowMeans(gene_probs)
} else {
  cat("Using aggressive subtypes:", paste(available_subtypes, collapse = ", "), "\n")
  gene_malignancy_prob <- rowSums(gene_probs[, available_subtypes, drop = FALSE])
}

combined_results <- data.table(
  Wisconsin_Malignant_Prob = wisconsin_probs[1:min(length(wisconsin_probs), length(histo_probs), length(gene_malignancy_prob))],
  Histo_Malignant_Prob = histo_probs[1:min(length(wisconsin_probs), length(histo_probs), length(gene_malignancy_prob))],
  Gene_Aggressive_Prob = gene_malignancy_prob[1:min(length(wisconsin_probs), length(histo_probs), length(gene_malignancy_prob))],
  Actual_Wisconsin = wisconsin_test_data$y[1:min(length(wisconsin_probs), length(histo_probs), length(gene_malignancy_prob))],
  Actual_Histo = histo_test_data$label[1:min(length(wisconsin_probs), length(histo_probs), length(gene_malignancy_prob))],
  Actual_Gene = gene_test_data_subset$type[1:min(length(wisconsin_probs), length(histo_probs), length(gene_malignancy_prob))]
)

# Combine Predictions
combined_results[, Combined_Malignant_Prob := rowMeans(.SD), .SDcols = c("Wisconsin_Malignant_Prob", "Histo_Malignant_Prob", "Gene_Aggressive_Prob")]
combined_results[, Prediction := ifelse(Combined_Malignant_Prob > 0.5, "Malignant/Aggressive", "Benign/Non-Aggressive")]

# Evaluate Combined Performance
conf_matrix_combined <- confusionMatrix(factor(combined_results$Prediction, levels = c("Benign/Non-Aggressive", "Malignant/Aggressive")), 
                                        factor(ifelse(combined_results$Actual_Wisconsin == "M", "Malignant/Aggressive", "Benign/Non-Aggressive"), 
                                               levels = c("Benign/Non-Aggressive", "Malignant/Aggressive")))
print(conf_matrix_combined)

# Check class distribution and compute ROC-AUC if possible
cat("Unique Actual_Wisconsin values:", paste(unique(combined_results$Actual_Wisconsin), collapse = ", "), "\n")
if (length(unique(combined_results$Actual_Wisconsin)) > 1) {
  roc_curve_combined <- roc(combined_results$Actual_Wisconsin, combined_results$Combined_Malignant_Prob, levels = c("B", "M"), direction = "<")
  auc_value <- auc(roc_curve_combined)
} else {
  cat("Warning: Only one class in Actual_Wisconsin. ROC-AUC cannot be computed.\n")
  auc_value <- NA
}
cat("Combined AUC:", auc_value, "\n")

# Visualize Results
ggplot(combined_results, aes(x = Wisconsin_Malignant_Prob, y = Histo_Malignant_Prob, color = Prediction)) +
  geom_point(alpha = 0.6) +
  labs(title = "Combined Malignancy Predictions", x = "Wisconsin Probability", y = "Histopathology Probability", color = "Final Prediction") +
  scale_color_manual(values = c("Malignant/Aggressive" = "red", "Benign/Non-Aggressive" = "blue")) +
  theme_minimal()
ggsave("combined_malignancy_plot.png")

# Summary Table
summary_table <- data.table(
  Metric = c("Combined AUC", "Accuracy", "Sensitivity (Malignant)", "Specificity (Benign)"),
  Value = c(auc_value, 
            conf_matrix_combined$overall["Accuracy"], 
            conf_matrix_combined$byClass["Sensitivity"], 
            conf_matrix_combined$byClass["Specificity"])
)
print(summary_table)
fwrite(summary_table, "combined_summary.csv")
fwrite(combined_results, "combined_results.csv")

cat("Combined analysis complete. Results saved as 'combined_results.csv', 'combined_summary.csv', and 'combined_malignancy_plot.png'.\n")

})  # End the system.time() block
print(time_taken)  # Display the time taken
