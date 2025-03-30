# Load necessary libraries
library(data.table)
library(caret)
library(ggplot2)
library(randomForest)

# Read the Wisconsin dataset
wisconsin_data <- fread("brca.csv")

# View the first few rows
head(wisconsin_data)

# View the dataset structure
str(wisconsin_data)

# Check for missing values
colSums(is.na(wisconsin_data))

# Check class distribution
table(wisconsin_data$y)

# Plot class distribution
ggplot(wisconsin_data, aes(x = y, fill = y)) +
  geom_bar() +
  labs(title = "Distribution of Benign and Malignant Cases", x = "Class", y = "Count") +
  scale_fill_manual(values = c("B" = "skyblue", "M" = "salmon"))

# Summary statistics of the dataset
summary(wisconsin_data)

# Save summary statistics to a CSV file
summary_stats <- summary(wisconsin_data)
write.csv(as.data.frame(summary_stats), "wisconsin_summary_statistics.csv", row.names = TRUE)

# First boxplot
# Boxplot for outlier detection
ggplot(melt(wisconsin_data, id.vars = "y"), aes(x = variable, y = value, fill = y)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of Numeric Features", x = "Features", y = "Values")

# Second boxplot
library(reshape2) 

# Exclude non-numeric columns before melting
numeric_cols <- names(wisconsin_data)[sapply(wisconsin_data, is.numeric)]
wisconsin_melted <- melt(wisconsin_data, id.vars = "y", measure.vars = numeric_cols)

# Boxplot for outlier detection
ggplot(wisconsin_melted, aes(x = variable, y = value, fill = y)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplot of Numeric Features", x = "Features", y = "Values")

# Count outliers using the IQR method
outlier_count <- sapply(wisconsin_data[, ..numeric_cols], function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR))
})

# Display number of outliers per column
outlier_count

# Function to extract actual outlier values (only for features that have outliers)
outliers_detail <- lapply(numeric_cols, function(col) {
  Q1 <- quantile(wisconsin_data[[col]], 0.25)
  Q3 <- quantile(wisconsin_data[[col]], 0.75)
  IQR <- Q3 - Q1
  outlier_values <- wisconsin_data[[col]][wisconsin_data[[col]] < (Q1 - 1.5 * IQR) | wisconsin_data[[col]] > (Q3 + 1.5 * IQR)]
  
  if (length(outlier_values) > 0) {  # Only create a dataframe if outliers exist
    return(data.frame(Feature = col, Outlier_Values = outlier_values))
  } else {
    return(NULL)  # Skip features with no outliers
  }
})

# Combine results into a single table (removing NULL entries)
outliers_table <- do.call(rbind, outliers_detail)

# Print the outliers table
print(outliers_table)

# Load correlation plot library
library(corrplot)

# Compute correlation matrix (only numeric columns)
cor_matrix <- cor(wisconsin_data[, ..numeric_cols])

# Plot the correlation heatmap
corrplot(cor_matrix, method = "color", tl.cex = 0.6, cl.cex = 0.6, type = "lower")

# Find highly correlated features (above 0.9)
high_corr <- findCorrelation(cor_matrix, cutoff = 0.9, names = FALSE)  # Get column indices

# Convert indices to actual feature names
high_corr_names <- colnames(cor_matrix)[high_corr]

# Print the features that will be removed
print(high_corr_names)

# Remove highly correlated features
wisconsin_data_filtered <- wisconsin_data[, !high_corr_names, with = FALSE]

# Check structure after removal
str(wisconsin_data_filtered)

# Save the cleaned dataset to a CSV file
write.csv(wisconsin_data_filtered, "wisconsin_data_filtered.csv", row.names = FALSE)

# Convert target variable to factor
wisconsin_data_filtered$y <- as.factor(wisconsin_data_filtered$y)

# Check the levels of 'y' to confirm
levels(wisconsin_data_filtered$y)

set.seed(123)   # Set seed for reproducibility

# Split the dataset into 80% training and 20% testing
train_index <- createDataPartition(wisconsin_data_filtered$y, p = 0.8, list = FALSE)
train_data <- wisconsin_data_filtered[train_index, ]
test_data <- wisconsin_data_filtered[-train_index, ]

# Check the dimensions of the split data
dim(train_data)
dim(test_data)

# Train the Random Forest model
set.seed(123)  # Set seed for reproducibility
rf_model <- randomForest(y ~ ., data = train_data, ntree = 100, mtry = 5, importance = TRUE)

# Print model summary
print(rf_model)

# Predict on the test set
rf_predictions <- predict(rf_model, newdata = test_data)

# Create confusion matrix
conf_matrix <- confusionMatrix(rf_predictions, test_data$y)

# Print the results
print(conf_matrix)

# Plot feature importance
importance(rf_model)
# Adjust margins and text size for better visibility
par(mar = c(10, 5, 2, 2))  # Increases bottom margin to fit labels
varImpPlot(rf_model, cex = 0.5)  # Reduces text size (cex = character expansion)

# Generate predictions using the trained model
predictions <- predict(rf_model, test_data)

# Compute precision, recall, and F1-score
precision <- posPredValue(predictions, test_data$y, positive = "B")
recall <- sensitivity(predictions, test_data$y, positive = "B")
f1_score <- (2 * precision * recall) / (precision + recall)

# Print results
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")

# Load necessary libraries
library(pROC)

# Convert predictions to probabilities
rf_probs <- predict(rf_model, test_data, type = "prob")[,2]  # Probability of "M" (Malignant)

# Compute ROC curve
roc_curve <- roc(test_data$y, rf_probs, levels = c("B", "M"), direction = "<")

# Plot ROC curve
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Random Forest")

# Compute AUC
auc_value <- auc(roc_curve)
cat("AUC Score:", auc_value, "\n")

# Perform 10-fold Cross-Validation
set.seed(123)
cv_results <- train(y ~ ., data = wisconsin_data_filtered, method = "rf",
                    trControl = trainControl(method = "cv", number = 10))

# Print results
print(cv_results)

# Save the Random Forest model
saveRDS(rf_model, "rf_model_wisconsin.rds")