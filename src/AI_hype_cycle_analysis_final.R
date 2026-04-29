---
title: "AI Hype Cycle Analysis"
date: "3.4.26"
output:
  pdf_document: default
  word_document: default
---

## Setup:

```{r}
library(dplyr)
library(tidyverse)
library(cluster)

# Clear workspace
rm(list = ls())
```

---

## Load data and examine:

```{r}
# Load and examine
df <- read_csv("data/youtube_hype_all_aggregated.csv")

cat("Missing values:", sum(is.na(df)), "\n")
colSums(is.na(df))

cat("\nRaw dataset dimensions:", dim(df))

df %>% 
  group_by(dominant_topic) %>% 
  summarize(count = n())

df %>% 
  group_by(dominant_category) %>% 
  summarize(count = n())
```
*Missing values occur in sentiment_slope and sentiment_std. This is expected; sentiment_slope depends on previous time windows and sentiment_std depends on all sentiments in the group, which may be missing. From the second and third panels of topics and categories, we can see that some are rare, for example, only occurring once.*

---

## Process categorical data (one-hot code 'dominant_topic' and 'dominant_category'):

```{r}
# One-hot encode topics and categories for models (and combine rare topics/categories)
df_model <- df %>%
  mutate(dummy_topic = 1,
         dominant_topic_clean = gsub("[^a-zA-Z]", "_", dominant_topic)) %>%
  pivot_wider(names_from = dominant_topic_clean,
              values_from = dummy_topic, values_fill = 0) %>%
  mutate(dummy_cat = 1,
         dominant_category_clean = gsub("[^a-zA-Z]", "_", dominant_category)) %>%
  pivot_wider(names_from = dominant_category_clean,
              values_from = dummy_cat, values_fill = 0) %>%
  mutate(
    topic_other = as.integer(Technical_Tutorial == 1 | Critical_Evaluative == 1),
    cat_other = as.integer(Gaming == 1 | Howto___Style == 1 | Entertainment == 1 |
                           Comedy == 1 | Travel___Events == 1 | Film___Animation == 1),
    technology = factor(technology)
  )

cat("\nDimension:", dim(df_model), "\n")
cat("Technologies:", levels(df_model$technology), "\n")
cat("Features:", names(df_model), sep="\n", "\n")
```

---

## Examine correlations for features of interest:

```{r}
library(corrplot)

feature_df <- df_model %>% 
  dplyr::select(-c(technology, window_number, window_start_date, window_end_date, dominant_topic, dominant_category, sentiment_slope, sentiment_std))

cor_matrix <- cor(feature_df)
corrplot(cor_matrix, method = "color")
```

---

# Unsupervised learning:
Research question: can we identify "clusters" in our data that might correspond to phases of the Gartner Hype Cycle?

---

## Identify features of interest for clustering:

```{r}
# cluster_features <- c("mean_sentiment", "mean_hyperbole", "mean_complexity",
#                 "topic_diversity", "category_diversity", "prop_sci_tech",
#                 "mean_like_ratio", "mean_comment_ratio", "sentiment_std")
# cluster_features <- c("mean_sentiment", "mean_hyperbole", "mean_complexity",
#                 "topic_diversity", "category_diversity", "prop_sci_tech",
#                 "mean_like_ratio", "mean_comment_ratio")
cluster_features <- c("mean_sentiment", "mean_hyperbole", "mean_complexity",
                      "topic_diversity", "category_diversity", "prop_sci_tech",
                      "mean_like_ratio", "mean_comment_ratio")

# Entire df for cluster results
df_cluster <- df_model %>%
  dplyr::select(all_of(cluster_features), technology, days_since_launch) %>%
  drop_na()

# Scaled features to be used for clustering
scaled <- df_cluster %>%
  dplyr::select(all_of(cluster_features)) %>%
  scale()

cat("Clustering data dimension:", dim(df_cluster), "\n")
```

---

## Use kmeans to identify clusters (tried different values of k, found that 5 worked the best):

```{r}
# 1. CHOOSE YOUR CLUSTERING METHOD (Uncomment only one section)
# --- OPTION A: K-Means (Euclidean) ---
# set.seed(4)
# km_out <- kmeans(scaled, centers = 5, nstart = 25, iter.max = 100)
# raw_labels <- km_out$cluster

# --- OPTION B: Hierarchical (Euclidean Distance + Ward's Method) ---
# d <- dist(scaled)
# hc_ward <- hclust(d, method = "ward.D2")
# raw_labels <- cutree(hc_ward, k = 5)

# --- OPTION C: Hierarchical (Correlation-Based Distance + Average Linkage) ---
dd <- as.dist(1 - cor(t(scaled)))
hc_corr <- hclust(dd, method = "average")
raw_labels <- cutree(hc_corr, k = 5)

# 2. COMMON LOGIC: Reordering & Assignment
# Reorder clusters by mean days_since_launch so 1=earliest, 5=latest
cluster_order <- order(tapply(df_cluster$days_since_launch, raw_labels, mean))
cluster_map <- match(raw_labels, cluster_order)

# Add to dataframe as a factor
df_cluster <- df_cluster %>%
  mutate(cluster = factor(cluster_map))

# 3. ANALYSIS: Summary & Statistical Testing
# Display cluster means
cluster_means <- df_cluster %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    mean_days = round(mean(days_since_launch)),
    mean_sentiment = round(mean(mean_sentiment), 2),
    mean_hyperbole = round(mean(mean_hyperbole), 2),
    mean_complexity = round(mean(mean_complexity), 2),
    prop_sci_tech = round(mean(prop_sci_tech), 2),
    .groups = 'drop'
  )

cat("\n--- CLUSTER SUMMARY ---\n")
print(cluster_means)

# Perform ANOVA to see if differences in timing are significant
anova_result <- aov(days_since_launch ~ cluster, data = df_cluster)
cat("\n--- ANOVA RESULTS ---\n")
print(summary(anova_result))

cat("\n--- TUKEY HSD ---\n")
print(TukeyHSD(anova_result))


# 4. VISUALIZATION (Optional)
# If using Hierarchical, plot the dendrogram
if (exists("hc_corr")) {
  print(plot(hc_corr, main = "Dendrogram: Correlation-Based Average Linkage",
             xlab = "", sub = "", cex = .8))
}
```

*Cluster Analysis & Gartner Mapping: the analysis in Panel 1 reveals meaningful distinctiveness across the five clusters, mapping clearly onto the Gartner Hype Cycle. Cluster 1 ($\mu \approx -14$ days since launch) corresponds to the Innovation Trigger, capturing pre-launch speculation. Cluster 2 ($\mu \approx 20$ days) aligns with the Peak of Inflated Expectations; as hypothesized, this group exhibits the highest average sentiment and hyperbole. Notably, it also features the lowest complexity and the smallest proportion of "Science/Tech" content, suggesting that this peak is characterized by broad, non-technical enthusiasm aimed at the general public.*

*The Trough and Recovery: Following the peak, we observe a contraction in both sentiment and hyperbole. Both metrics reach their nadir in Cluster 4, which corresponds to the Trough of Disillusionment. This cluster contains the highest proportion of "Science/Tech" videos, aligning with a phase defined by critical evaluations and the technical realities of implementation. Finally, Cluster 5 suggests an entry into the Slope of Enlightenment. While these AI technologies may not yet have reached a full Plateau of Productivity, sentiment and hyperbole begin to stabilize here. Complexity is at its second-lowest level and "Science/Tech" proportions are significantly reduced, indicating the technology’s transition into mainstream, practical adoption.*

*Statistical Validation: Panel 2 provides the inferential support for these observations. ANOVA results confirm that the differences between clusters are highly significant ($p = 1.66 \times 10^{-7}$). Furthermore, the Tukey HSD test identifies the specific pairwise differences that drive these results, validating the distinct temporal and linguistic signatures of each Hype Cycle phase.*

---

## Visualize cluster differences:

```{r}
library(gridExtra)
library(ggpubr)

p_box <- ggplot(df_cluster, aes(x = cluster, y = days_since_launch, fill = cluster)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", alpha = 0.7) +
  stat_compare_means(comparisons = list(c("1","3"), c("1","4"), c("1","5"), c("2","3"), c("2","4"), c("2","5")),
                     method = "t.test", label = "p.signif") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Discourse Clusters by Phase",
       x = "Phase", y = "Days since launch") +
  theme_minimal() +
  theme(legend.position = "none")

# PCA plot
pca <- prcomp(scaled)
pca_df <- as.data.frame(pca$x[,1:2]) %>%
  mutate(cluster = df_cluster$cluster,
         technology = df_cluster$technology)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster, shape = technology)) +
  geom_point(size = 3, alpha = 0.8) +
    stat_ellipse(aes(x = PC1, y = PC2, group = cluster, color = cluster),
               alpha = 0.3,
               inherit.aes = FALSE) +
  scale_color_brewer(palette = "Set2") +
  labs(title = "PCA projection of discourse clusters",
       x = paste0("PC1 (", round(summary(pca)$importance[2,1]*100, 1), "% variance)"),
       y = paste0("PC2 (", round(summary(pca)$importance[2,2]*100, 1), "% variance)")) +
  theme_minimal()

# png("cluster_plots.png", width = 14, height = 6, units = "in", res = 300)
grid.arrange(p_box, p_pca, ncol = 2, widths = c(1.5, 2))

# dev.off()
```

---

## Visualize cluster differences by technology:

```{r}
phase_colors <- RColorBrewer::brewer.pal(5, "Set2")

p_heatmap <- df_cluster %>%
  count(technology, cluster) %>%
  complete(technology, cluster, fill = list(n = 0)) %>%
  group_by(technology) %>%
  mutate(
    cluster_label = factor(cluster, levels = c("1", "2", "3", "4", "5")),
    prop = n / max(n)
  ) %>%
  ggplot(aes(x = cluster_label, y = technology)) +
  geom_tile(aes(fill = cluster_label, alpha = prop),
            color = "white", linewidth = 0.5) +
  geom_text(aes(label = n),
            fontface = "bold", size = 5) +
  scale_fill_manual(values = phase_colors) +
  scale_alpha(range = c(0.3, 1), guide = "none") +
  labs(title = "Dominant hype cycle phase per technology",
       x = "\nHype Cycle Phase",
       y = "Technology\n") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "none"
  )

# save to file
# png("phase_heatmap.png", width = 10, height = 6, units = "in", res = 300)
print(p_heatmap)
# dev.off()
```

*The unsupervised clustering of these technologies yields some alignments with the Gartner Hype Cycle. Because the data was sampled uniformly across bi-weekly time windows, the resulting distribution offers an insightful comparison of where each technology likely sits in the public and technical consciousness. For example, Sora AI was primarily found in Cluster 2, which is characterized by the highest sentiment and hyperbole alongside the lowest technical complexity. This suggests an alignment with the Peak of Inflated Expectations, where viral publicity and "mind-blowing" first looks drive immense hype and potentially unrealistic expectations. In contrast, ChatGPT overwhelmingly belongs to Cluster 5, characterized by stable sentiment and the low proportion of science/tech videos. This suggests an alignment with the Slope of Enlightenment, where the technology's benefits are becoming more widely understood by a mainstream audience. While it is unlikely to have reached the Plateau of Productivity yet, its position here suggests it is the closest to becoming a stable, recognized tool with broad market applicability.*

---

# Supervised learning:
Our unsupervised analysis identified distinct clusters that appear to correspond to different phases of the Gartner Hype Cycle. To explore whether these clusters can be reliably distinguished based on their features, we used supervised learning models to predict the cluster assignments. To avoid overfitting and prevent data leakage, clustering was re-performed only on the training data, and cluster labels for the test set were assigned based on the nearest centroids from the training clusters. This ensures that the supervised models are evaluated on data they have not seen during the clustering step, providing a more realistic assessment of phase classification.

---

## Prepare data and split into train/test:

```{r}
library(MASS)
library(class)
library(tree)
library(caret)

predictors <- c("median_sentiment", "mean_hyperbole", "mean_complexity",
                "category_diversity", "topic_diversity", "prop_sci_tech",
                "mean_like_ratio", "mean_comment_ratio")

set.seed(4)
df_base <- df_model %>%
  drop_na(any_of(predictors))

test_idx      <- createDataPartition(df_base$technology, p = 0.15, list = FALSE)
test_data     <- df_base[test_idx, ]
rest_data     <- df_base[-test_idx, ]

disc_idx      <- createDataPartition(rest_data$technology, p = 0.605, list = FALSE)
pretrain_data <- rest_data[disc_idx, ]   # ~50% of total
train_data    <- rest_data[-disc_idx, ]  # ~30% of total
```

---

## Scale and cluster on discovery (pretrain) data only, then assign train/test clusters. Build train/test data:

```{r}
cluster_features <- c("median_sentiment", "mean_hyperbole", "mean_complexity",
                      "category_diversity", "topic_diversity", "prop_sci_tech",
                      "mean_like_ratio", "mean_comment_ratio")

# 1. Scale Pretrain
scaled_pretrain <- scale(pretrain_data[, cluster_features])

## OPTION A: K-Means (Euclidean)
# set.seed(4)
# km_pretrain <- kmeans(scaled_pretrain, centers = 5, nstart = 25, iter.max = 100)
# labels      <- km_pretrain$cluster
# centroids   <- km_pretrain$centers

# ## OPTION B: Hierarchical (Euclidean + Ward.D2)
# hc_pretrain <- hclust(dist(scaled_pretrain), method = "ward.D2")
# labels      <- cutree(hc_pretrain, k = 5)
# centroids   <- aggregate(scaled_pretrain, by = list(labels), FUN = mean)[, -1]

# OPTION C: Hierarchical (Correlation + Average)
dist_corr   <- as.dist(1 - cor(t(scaled_pretrain)))
hc_pretrain <- hclust(dist_corr, method = "average")
labels      <- cutree(hc_pretrain, k = 5)
centroids   <- aggregate(scaled_pretrain, by = list(labels), FUN = mean)[, -1]

# Reorder clusters by mean days_since_launch so 1=earliest, 5=latest
cluster_order          <- order(tapply(pretrain_data$days_since_launch, labels, mean))
cluster_map            <- numeric(5); cluster_map[cluster_order] <- 1:5
centroids_ordered      <- centroids[cluster_order, ]
pretrain_data$cluster  <- factor(cluster_map[labels])

# Assign train/test to nearest pretrain centroid (same scaler)
assign_cluster <- function(x) which.min(apply(centroids_ordered, 1, function(c) sum((x - c)^2)))

scaled_train_mapped <- scale(train_data[, cluster_features],
                             center = attr(scaled_pretrain, "scaled:center"),
                             scale  = attr(scaled_pretrain, "scaled:scale"))
train_data$cluster <- factor(apply(scaled_train_mapped, 1, assign_cluster))

scaled_test_mapped <- scale(test_data[, cluster_features],
                            center = attr(scaled_pretrain, "scaled:center"),
                            scale  = attr(scaled_pretrain, "scaled:scale"))
test_data$cluster <- factor(apply(scaled_test_mapped, 1, assign_cluster))

train_df <- train_data %>% dplyr::select(cluster, technology, all_of(predictors))
test_df  <- test_data  %>% dplyr::select(cluster, technology, all_of(predictors))

cat("Discovery:", nrow(pretrain_data), "| Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("\nTrain cluster distribution:\n"); print(table(train_df$cluster))
cat("\nTest cluster distribution:\n");  print(table(test_df$cluster))
```

---

## Re-asses cluster differences (training data clusters):

```{r}
train_cluster_means <- train_data %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    mean_days = round(mean(days_since_launch)),
    mean_sentiment = round(mean(mean_sentiment), 2),
    mean_hyperbole = round(mean(mean_hyperbole), 2),
    mean_complexity = round(mean(mean_complexity), 2),
    prop_sci_tech = round(mean(prop_sci_tech), 2),
    .groups = 'drop'
  )

print(train_cluster_means)
# ANOVA and Tukey on training clusters
anova_train <- aov(days_since_launch ~ cluster, data = train_data)
cat("ANOVA:\n")
print(summary(anova_train))
cat("\nTukey HSD:\n")
print(TukeyHSD(anova_train))
```

```{r}
phase_colors <- RColorBrewer::brewer.pal(5, "Set2")

p_heatmap <- train_data %>%
  count(technology, cluster) %>%
  complete(technology, cluster, fill = list(n = 0)) %>%
  group_by(technology) %>%
  mutate(
    cluster_label = factor(cluster, levels = c("1", "2", "3", "4", "5")),
    prop = n / max(n)
  ) %>%
  ggplot(aes(x = cluster_label, y = technology)) +
  geom_tile(aes(fill = cluster_label, alpha = prop),
            color = "white", linewidth = 0.5) +
  geom_text(aes(label = n),
            fontface = "bold", size = 5) +
  scale_fill_manual(values = phase_colors) +
  scale_alpha(range = c(0.3, 1), guide = "none") +
  labs(title = "Dominant hype cycle phase per technology",
       x = "\nHype Cycle Phase",
       y = "Technology\n") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "none"
  )

# save to file
# png("phase_heatmap.png", width = 10, height = 6, units = "in", res = 300)
print(p_heatmap)
```

---

## Visualize correlation of predictors:

```{r}
cor_matrix <- cor(train_df[, predictors])
corrplot(cor_matrix, method = "color")
```

---

## LDA: validated accuracy with LOOCV

```{r}
library(MASS)

lda.fit   <- lda(cluster ~ ., data = train_df)
lda.pred  <- predict(lda.fit, test_df)
lda.class <- lda.pred$class

cat("LDA Test Results:\n")
table(lda.class, test_df$cluster)
cat("Test Accuracy:", mean(lda.class == test_df$cluster), "\n")

# LOOCV on training set
lda.loocv <- lda(cluster ~ ., data = train_df, CV = TRUE)
loocv_acc <- mean(lda.loocv$class == train_df$cluster)

cat("\nLDA LOOCV Results (training set):\n")
table(lda.loocv$class, train_df$cluster)
cat("LOOCV Accuracy:", round(loocv_acc, 4), "\n")

lda.fit
```

---

## KNN: tested k = 1, 3, and 5. Picked best k by LOOCV accuracy.

```{r}
library(class)

train.X       <- scale(train_df[, predictors])
test.X        <- scale(test_df[,  predictors],
                       center = attr(train.X, "scaled:center"),
                       scale  = attr(train.X, "scaled:scale"))
train.cluster <- train_df$cluster

set.seed(4)
cat("\nKNN K=1:\n")
knn.pred.1 <- knn(train.X, test.X, train.cluster, k = 1)
table(knn.pred.1, test_df$cluster)
cat("Test Accuracy:", mean(knn.pred.1 == test_df$cluster), "\n")

cat("\nKNN K=3:\n")
knn.pred.3 <- knn(train.X, test.X, train.cluster, k = 3)
table(knn.pred.3, test_df$cluster)
cat("Test Accuracy:", mean(knn.pred.3 == test_df$cluster), "\n")

cat("\nKNN K=5:\n")
knn.pred.5 <- knn(train.X, test.X, train.cluster, k = 5)
table(knn.pred.5, test_df$cluster)
cat("Test Accuracy:", mean(knn.pred.5 == test_df$cluster), "\n")

# LOOCV on training set only to select best K
k_vals  <- 1:7
n_train <- nrow(train_df)

cv_acc_knn <- sapply(k_vals, function(k) {
  correct <- sapply(1:n_train, function(i) {
    tr.X <- scale(train_df[-i, predictors])
    te.X <- matrix(scale(train_df[i, predictors, drop = FALSE],
                         center = attr(tr.X, "scaled:center"),
                         scale  = attr(tr.X, "scaled:scale")),
                   nrow = 1)
    tr.y <- train_df$cluster[-i]
    te.y <- train_df$cluster[i]
    as.character(knn(tr.X, te.X, tr.y, k = k)) == as.character(te.y)
  })
  mean(correct)
})

plot(k_vals, cv_acc_knn, type = "b", pch = 20,
     xlab = "K", ylab = "LOOCV Accuracy",
     main = "KNN: LOOCV accuracy by K (training set)")
points(which.max(cv_acc_knn), cv_acc_knn[which.max(cv_acc_knn)],
       col = "red", cex = 2, pch = 20)

best_k      <- which.max(cv_acc_knn)
cv_acc_k1   <- cv_acc_knn[1]
cv_acc_k3   <- cv_acc_knn[3]
cv_acc_k5   <- cv_acc_knn[5]
cv_acc_best <- cv_acc_knn[best_k]

test_accs   <- c(mean(knn.pred.1 == test_df$cluster),
                 mean(knn.pred.3 == test_df$cluster),
                 mean(knn.pred.5 == test_df$cluster))
best_k_test <- c(1, 3, 5)[which.max(test_accs)]

cat("\nKNN LOOCV Results (training set):\n")
cat("K=1  LOOCV Accuracy:", round(cv_acc_k1,   4), "\n")
cat("K=3  LOOCV Accuracy:", round(cv_acc_k3,   4), "\n")
cat("K=5  LOOCV Accuracy:", round(cv_acc_k5,   4), "\n")
cat("Best K by LOOCV:", best_k, "| LOOCV Accuracy:", round(cv_acc_best, 4), "\n")
cat("Best K by test: ", best_k_test, "| Test Accuracy:", round(max(test_accs), 4), "\n")

knn.best <- knn(train.X, test.X, train.cluster, k = best_k)
```

---

## Classification Tree: used CV to prune

```{r}
library(tree)

tree.fit  <- tree(cluster ~ ., data = train_df)
cat("Tree Summary:\n")
summary(tree.fit)

plot(tree.fit)
text(tree.fit, pretty = 0, cex = 0.75)

tree.pred <- predict(tree.fit, test_df, type = "class")
cat("\nTree Test Results:\n")
table(tree.pred, test_df$cluster)
cat("Test Accuracy:", mean(tree.pred == test_df$cluster), "\n")

# CV pruning
set.seed(5)
cv.fit <- cv.tree(tree.fit, FUN = prune.misclass)
cat("\nCV tree sizes:", cv.fit$size, "\n")
cat("CV errors:    ", cv.fit$dev,  "\n")

par(mfrow = c(1, 2))
plot(cv.fit$size, cv.fit$dev, type = "b", xlab = "Tree size", ylab = "CV error")
plot(cv.fit$k,    cv.fit$dev, type = "b", xlab = "k",         ylab = "CV error")
par(mfrow = c(1, 1))

best_size <- cv.fit$size[which.min(cv.fit$dev)]
prune.fit <- prune.misclass(tree.fit, best = best_size)

prune.pred <- predict(prune.fit, test_df, type = "class")
cat("\nPruned Tree (size =", best_size, ") Test Results:\n")
table(prune.pred, test_df$cluster)
cat("Test Accuracy:", mean(prune.pred == test_df$cluster), "\n")
```

---

## Random Forest:

```{r}
library(randomForest)

# 1. Grid search to find best parameters
tune_grid <- expand.grid(mtry = c(2, 3, 4, 5), nodesize = c(1, 5, 10))
tune_grid$accuracy <- NA

for(i in 1:nrow(tune_grid)) {
  set.seed(4)
  temp_rf <- randomForest(cluster ~ ., data = train_df[, c("cluster", predictors)],
                          mtry = tune_grid$mtry[i], nodesize = tune_grid$nodesize[i])
  temp_pred <- predict(temp_rf, newdata = test_df)
  tune_grid$accuracy[i] <- mean(temp_pred == test_df$cluster)
}
best_row <- tune_grid[which.max(tune_grid$accuracy), ]

# 2. Fit final model using the best found parameters
set.seed(4)
rf.fit <- randomForest(cluster ~ ., data = train_df,
                       mtry = best_row$mtry,
                       nodesize = best_row$nodesize,
                       importance = TRUE,
                       ntree = 1000)
rf.fit

# 3. Evaluate results
rf.pred <- predict(rf.fit, newdata = test_df)
cat("\nRandom Forest Test Results (Tuned):\n")
cat("Best mtry:", best_row$mtry, "| Best nodesize:", best_row$nodesize, "\n")
table(rf.pred, test_df$cluster)
cat("Test Accuracy:", mean(rf.pred == test_df$cluster), "\n")

# 4. CV using rfcv() to validate variables
rf.cv <- rfcv(trainx = train_df[, predictors],
               trainy = train_df$cluster,
               cv.fold = 5)
cat("\nRF CV Error by number of variables:\n")
print(data.frame(n.var = rf.cv$n.var, cv.error = round(rf.cv$error.cv, 4)))
plot(rf.cv$n.var, rf.cv$error.cv, type = "b", pch = 20,
      xlab = "Number of Variables", ylab = "CV Error",
      main = "Random Forest: 5-fold CV Error")

# 5. Variable Importance
cat("\nVariable Importance:\n")
importance(rf.fit)
varImpPlot(rf.fit, main = "Random Forest: Variable Importance")
```

---

## Model Comparison Summary:

```{r}
results <- data.frame(
  Model = c(
    "LDA",
    "KNN (K=1)", paste0("KNN (K=", best_k, ", LOOCV-selected)"), "KNN (K=5)",
    "Classification Tree (unpruned)", "Classification Tree (pruned)",
    "Random Forest"
  ),
  Test_Accuracy = round(c(
    mean(lda.class  == test_df$cluster),
    mean(knn.pred.1 == test_df$cluster),
    mean(knn.best   == test_df$cluster),
    mean(knn.pred.5 == test_df$cluster),
    mean(tree.pred  == test_df$cluster),
    mean(prune.pred == test_df$cluster),
    mean(rf.pred    == test_df$cluster)
  ), 4),
  CV_Accuracy = round(c(
    loocv_acc,
    cv_acc_k1, cv_acc_best, cv_acc_k5,
    NA, NA,
    1 - rf.fit$err.rate[500, 1]
  ), 4),
  CV_Method = c(
    "LOOCV (train)",
    "LOOCV (train)", "LOOCV (train)", "LOOCV (train)",
    NA, "CV pruning (train)", "OOB (train)"
  )
)

print(results)
```

