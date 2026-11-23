# %%
library(brms)
library(ape)

# 1. Load data and fit brms
phylo        <- read.nexus("phylo.nex")
data_fisher  <- read.table("../data_effect.txt", header = TRUE)
data_fisher$obs <- 1:nrow(data_fisher)
data_fisher$se  <- sqrt(1 / (data_fisher$N - 3))
A <- vcv.phylo(phylo)

cat("\nFitting brms meta-analysis model...\n")
model_meta <- brm(
  Zr | se(se) ~ 1 + (1 | gr(phylo, cov = A)) + (1 | obs),
  data = data_fisher, family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 10), "Intercept"),
    prior(student_t(3, 0, 10), "sd")
  ),
  chains = 2, cores = 2, iter = 4000, warmup = 1000,
  refresh = 0
)

# 2. Load BF posteriors
BF_post   <- read.csv("BF_post_meta.csv")
post_brms <- as.data.frame(model_meta)

# 3. Compare posterior means -> log.txt
cat("\n--- Comparison of Posterior Means (Meta-Analysis) ---\n")
comparison <- data.frame(
  Parameter = c("Intercept", "sd_phylo", "sd_obs"),
  brms_Mean = c(
    mean(post_brms$b_Intercept),
    mean(post_brms$sd_phylo__Intercept),
    mean(post_brms$sd_obs__Intercept)
  ),
  BF_Mean = c(
    mean(BF_post$Intercept),
    mean(BF_post$sd_phylo),
    mean(BF_post$sd_obs)
  )
)
comparison$Difference <- comparison$BF_Mean - comparison$brms_Mean
print(comparison)
write.csv(comparison, "comparison_meta.csv", row.names = FALSE)
cat("\n--- META-ANALYSIS MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
write.table(comparison, file = "log.txt", append = TRUE,
            row.names = FALSE, sep = "\t", quote = FALSE)

# 4. Combined density panel (all parameters)
cat("\nGenerating combined density comparison panel...\n")
draw_density <- function(label, brms_s, BF_s) {
  brms_s <- as.numeric(na.omit(as.vector(brms_s)))
  BF_s   <- as.numeric(na.omit(BF_s))
  d1 <- density(brms_s); d2 <- density(BF_s)
  xlim <- range(c(d1$x, d2$x)); ylim <- range(c(d1$y, d2$y))
  plot(d1, col = "red", lwd = 2, main = label,
       xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density")
  lines(d2, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("brms", "BF"),
         col = c("red", "blue"), lwd = 2, lty = c(1, 2), cex = 0.8)
}

png("density_comparison.png", width = 1800, height = 600, res = 120, type = "cairo")
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
draw_density("Intercept", post_brms$b_Intercept,         BF_post$Intercept)
draw_density("sd_phylo",  post_brms$sd_phylo__Intercept, BF_post$sd_phylo)
draw_density("sd_obs",    post_brms$sd_obs__Intercept,   BF_post$sd_obs)
dev.off()
cat("Panel saved to density_comparison.png\n")
# %%
