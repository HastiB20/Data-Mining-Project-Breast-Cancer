time_taken <- system.time({
  


# Load libraries
library(data.table)
library(randomForest)
library(caret)
library(png) 

# Confirm current working directory
cat("Current working directory:", getwd(), "\n")

# List all image files from the archive folder (assuming PNG format)
image_files <- list.files(path = "archive", recursive = TRUE, full.names = TRUE, 
                          pattern = "\\.png$")  # Using PNG only; adjust if JPEG needed

# Create a data table to store file paths and labels
image_data <- data.table(file = image_files, label = NA_character_)

# Assign labels based on folder structure (0 = benign, 1 = malignant)
image_data[grep("/0/", file), label := "0"]
image_data[grep("/1/", file), label := "1"]

# Function to preprocess an image using png package
preprocess_image <- function(image_path) {
  # Load image
  img <- readPNG(image_path)  # Returns a 3D array (height, width, channels) or 2D if grayscale
  
  # If color image (3 channels), convert to grayscale
  if (length(dim(img)) == 3 && dim(img)[3] == 3) {
    img_gray <- 0.2989 * img[,,1] + 0.5870 * img[,,2] + 0.1140 * img[,,3]  # Standard RGB to grayscale
  } else {
    img_gray <- img  # Already grayscale
  }
  
  # Resize to 100x100 (simple downsampling by averaging)
  h <- dim(img_gray)[1]
  w <- dim(img_gray)[2]
  img_resized <- matrix(0, 100, 100)
  for (i in 1:100) {
    for (j in 1:100) {
      x_start <- floor((i - 1) * h / 100) + 1
      x_end <- floor(i * h / 100)
      y_start <- floor((j - 1) * w / 100) + 1
      y_end <- floor(j * w / 100)
      img_resized[i, j] <- mean(img_gray[x_start:x_end, y_start:y_end])
    }
  }
  
  # Normalize pixel values to 0-1
  img_normalized <- img_resized / max(img_resized)
  
  return(img_normalized)
}

# Preprocess all images and store as a list
cat("Preprocessing", length(image_files), "images...\n")
preprocessed_images <- list()
for (i in 1:nrow(image_data)) {
  preprocessed_images[[i]] <- preprocess_image(image_data[i, file])
  if (i %% 50 == 0) cat("Processed", i, "images\n")
}

# Add preprocessed images to the data table (as a list column)
image_data[, preprocessed := preprocessed_images]

# Check for any missing labels or failed preprocessing
image_data_clean <- image_data[!is.na(label) & !is.na(preprocessed)]

# Save the preprocessed data for later use
saveRDS(image_data_clean, "preprocessed_histopathology.rds")

# Summary of loaded data
cat("Total images preprocessed:", nrow(image_data_clean), "\n")
cat("Benign (0):", sum(image_data_clean$label == "0"), "\n")
cat("Malignant (1):", sum(image_data_clean$label == "1"), "\n")


# Load the preprocessed data
image_data_clean <- readRDS("preprocessed_histopathology.rds")

# Function to extract simple features from a preprocessed image
extract_features <- function(img) {
  pixels <- as.vector(img)  # Flatten the 100x100 matrix to a vector
  features <- c(
    mean_intensity = mean(pixels),      # Average pixel value
    sd_intensity = sd(pixels),          # Standard deviation of intensity
    sum_intensity = sum(pixels),        # Total intensity (proxy for brightness)
    max_intensity = max(pixels),        # Maximum pixel value
    min_intensity = min(pixels)         # Minimum pixel value
  )
  return(features)
}

# Extract features for all images
cat("Extracting features from", nrow(image_data_clean), "images...\n")
feature_list <- lapply(image_data_clean$preprocessed, extract_features)

# Convert list of features to a data table
feature_data <- as.data.table(do.call(rbind, feature_list))

# Combine features with labels
image_features <- cbind(image_data_clean[, .(label)], feature_data)

# Ensure label is a factor for classification
image_features[, label := as.factor(label)]

# Check for missing values and remove if any
image_features_clean <- na.omit(image_features)

# Save the feature dataset
saveRDS(image_features_clean, "histopathology_features.rds")

# Summary of feature data
cat("Total images with features:", nrow(image_features_clean), "\n")
cat("Benign (0):", sum(image_features_clean$label == "0"), "\n")
cat("Malignant (1):", sum(image_features_clean$label == "1"), "\n")
head(image_features_clean)  # Preview the first few rows



# Load the feature dataset
image_features_clean <- readRDS("histopathology_features.rds")

# Split into training and testing sets (80% train, 20% test)
set.seed(123)  # For reproducibility
train_index <- createDataPartition(image_features_clean$label, p = 0.8, list = FALSE)
train_data <- image_features_clean[train_index]
test_data <- image_features_clean[-train_index]

# Train Random Forest model
rf_model <- randomForest(label ~ ., data = train_data, 
                         ntree = 100,  # Simple number of trees
                         mtry = 2,     # Number of features to try at each split (sqrt(5) ≈ 2)
                         importance = TRUE)

# Print model summary
print(rf_model)

# Evaluate on test set
test_predictions <- predict(rf_model, newdata = test_data)
confusionMatrix(test_predictions, test_data$label)

# Feature importance
importance_scores <- importance(rf_model, type = 2)  # Mean Decrease in Gini
cat("Feature Importance (Mean Decrease in Gini):\n")
print(importance_scores)

# Save the model
saveRDS(rf_model, "rf_model_histopathology.rds")


# Load the trained model
rf_model <- readRDS("rf_model_histopathology.rds")

# Feature importance (already calculated, just for plotting)
importance_scores <- importance(rf_model, type = 2)  # Mean Decrease in Gini

# Simple barplot of feature importance
barplot(importance_scores[, 1], names.arg = rownames(importance_scores), 
        main = "Feature Importance for Malignancy Prediction", 
        xlab = "Features", ylab = "Mean Decrease in Gini", 
        las = 2, col = "skyblue")  # las=2 rotates labels for readability


# Summary table of performance metrics
performance_summary <- data.table(
  Metric = c("OOB Accuracy", "Test Accuracy", "Sensitivity (Benign)", "Specificity (Malignant)", 
             "Top Feature", "Top Feature Score"),
  Value = c(1 - 0.132, 0.8697, 0.9302, 0.6916, "mean_intensity", 209.3369)
)

# Print summary
cat("Performance Summary:\n")
print(performance_summary)

# Save summary to a CSV file
fwrite(performance_summary, "performance_summary.csv")

# Final message
cat("Analysis complete. Model, plot, and summary saved in working directory.\n")


saveRDS(rf_model, "rf_model_histopathology.rds")

})  # End the system.time() block
print(time_taken)  # Display the time taken

