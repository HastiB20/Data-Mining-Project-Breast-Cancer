time_taken <- system.time({
  

# Load necessary libraries
library(data.table)
library(caret)
library(randomForest)

# Load dataset efficiently and convert to data.frame
gene_data <- fread("Breast_GSE45827.csv", data.table = FALSE)

# Keep only numeric columns (gene expression values)
gene_data <- gene_data[, sapply(gene_data, is.numeric), drop = FALSE]

# Check dimensions to see the actual number of rows and columns
dim(gene_data)

# Display first few rows
head(gene_data)

# Check total number of missing values
sum(is.na(gene_data))

# Remove rows with any missing values (if any exist)
gene_data_clean <- na.omit(gene_data)

# Check dimensions after cleaning
dim(gene_data_clean)

# Normalize data with log2 transformation
gene_data_normalized <- log2(gene_data_clean + 1)

# Check the first few rows of normalized data (first 5 columns)
head(gene_data_normalized[, 1:5])

# Load the original dataset again to get the 'type' column
original_data <- fread("Breast_GSE45827.csv", data.table = FALSE)

# Extract the 'type' column (assuming it’s the target variable)
type_column <- original_data$type

# Attach 'type' to the normalized dataset (ensure row order matches)
gene_data_normalized <- cbind(type = type_column, gene_data_normalized)

# Filter out low-variance genes (keep top 25% most variable)
variances <- apply(gene_data_normalized[, -1], 2, var)  # Exclude 'type' column
top_genes <- names(variances)[variances > quantile(variances, 0.75)]
gene_data_filtered <- gene_data_normalized[, c("type", top_genes)]

# Check dimensions of the filtered dataset
dim(gene_data_filtered)

# Check first few rows (limit to first 5 columns)
head(gene_data_filtered[, 1:5])

# Set seed for reproducibility
set.seed(123)

# Create train-test split (80% training, 20% testing)
train_index <- createDataPartition(gene_data_filtered$type, p = 0.8, list = FALSE)
train_data <- gene_data_filtered[train_index, ]
test_data <- gene_data_filtered[-train_index, ]

# Save the test data for later use in combine_results.R
saveRDS(test_data, "gene_test_data.rds")

# Check the dimensions of the training and testing sets
dim(train_data)
dim(test_data)

# Ensure that the 'type' column is a factor (target variable)
train_data$type <- as.factor(train_data$type)

# Check if '1405_i_at' exists in train_data
if ("1405_i_at" %in% colnames(train_data)) {
  print("Column '1405_i_at' is present in train_data")
} else {
  print("Column '1405_i_at' is NOT present in train_data")
}

# Subset to top 50 genes + 'type', ensuring valid column names
top_50_genes <- top_genes[1:min(50, length(top_genes))]  # Ensure we don’t exceed available genes
train_data_small <- train_data[, c("type", top_50_genes), drop = FALSE]

# Clean column names to ensure they are valid R variable names
colnames(train_data_small) <- make.names(colnames(train_data_small), unique = TRUE)

# Check column names in the smaller dataset
print("Column names in train_data_small after make.names:")
print(colnames(train_data_small))

# Check if '1405_i_at' exists or has been renamed
if ("1405_i_at" %in% colnames(train_data_small)) {
  print("Column '1405_i_at' is present in train_data_small")
} else {
  print("Column '1405_i_at' might have been renamed, checking:")
  print(colnames(train_data_small)[grep("1405", colnames(train_data_small))])
}

# Ensure there are no missing values in the dataset
sum(is.na(train_data_small))

# Train the Random Forest model with the smaller dataset
rf_model <- randomForest(type ~ ., data = train_data_small, 
                         ntree = 50, 
                         mtry = 10, 
                         maxnodes = 30, 
                         importance = TRUE)

# Print the model summary
print(rf_model)

# Ensure all necessary libraries are loaded
library(ggplot2)  # For confusion matrix heatmap
library(reshape2) # For melting the confusion matrix

# 1. Confusion Matrix Heatmap
# Extract the confusion matrix from the model
conf_matrix <- rf_model$confusion[, 1:6]  # Exclude the class.error column
rownames(conf_matrix) <- c("Basal", "Cell Line", "HER", "Luminal A", "Luminal B", "Normal")
colnames(conf_matrix) <- c("Basal", "Cell Line", "HER", "Luminal A", "Luminal B", "Normal")

# Melt the matrix for ggplot2
conf_matrix_melted <- melt(conf_matrix)
colnames(conf_matrix_melted) <- c("Actual", "Predicted", "Count")

# Create the heatmap
heatmap_plot <- ggplot(conf_matrix_melted, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() + 
  geom_text(aes(label = Count), color = "white", size = 4) +  # Add numbers on tiles
  scale_fill_gradient(low = "lightblue", high = "darkblue", name = "Count") +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Subtype", y = "Actual Subtype") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

# Display the plot
print(heatmap_plot)

# 2. Variable Importance Plot
# Plot the importance of variables (Mean Decrease in Gini)
varImpPlot(rf_model, main = "Variable Importance (Mean Decrease in Gini)", 
           n.var = 10)  # Show top 10 most important variables for brevity

# Optional: Extract and print the top 10 most important variables for reference
importance_scores <- importance(rf_model, type = 2)  # Type 2 = Mean Decrease in Gini
top_10_genes <- head(sort(importance_scores[, 1], decreasing = TRUE), 10)
print("Top 10 most important genes:")
print(top_10_genes)

# Save the Random Forest model
saveRDS(rf_model, "rf_model_gene.rds")

})  # End the system.time() block
print(time_taken)  # Display the time taken
