install.packages(c("data.table", "randomForest", "caret"), dependencies = TRUE)
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("EBImage")
