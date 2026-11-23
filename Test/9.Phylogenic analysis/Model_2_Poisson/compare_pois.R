# %%
library(brms)
library(ape)

# 1. Load data and fit brms
phylo     <- read.nexus("phylo.nex")
data_pois <- read.table("data_pois.txt", header = TRUE)
data_pois$obs <- 1:nrow(data_pois)
A <- vcv.phylo(phylo)

cat("\nFitting brms Poisson model...\n")
model_pois <- brm(
  phen_pois ~ cofactor + (1 | gr(phylo, cov = A)) + (1 | obs),
  data = data_pois, family = poisson("log"),
  data2 = list(A = A),
  chains = 2, cores = 2, iter = 4000, warmup = 2000,
  control = list(adapt_delta = 0.95),
  refresh = 0
)

# 2. Load BF posteriors
BF_post   <- read.csv("BF_post_pois.csv")
post_brms <- as.data.frame(model_pois)

# 3. Compare posterior means -> log.txt
cat("\n--- Comparison of Poisson Model Results ---\n")
comparison <- data.frame(
  Parameter = c("Intercept (uncentered)", "cofactor", "sd_phylo", "sd_obs"),
  brms_Mean = c(
    mean(post_brms$b_Intercept),
    mean(post_brms$b_cofactor),
    mean(post_brms$sd_phylo__Intercept),
    mean(post_brms$sd_obs__Intercept)
  ),
  BF_Mean = c(
    mean(BF_post$b_Intercept),
    mean(BF_post$b_cofactor),
    mean(BF_post$sd_phylo),
    mean(BF_post$sd_obs)
  )
)
comparison$Difference <- comparison$BF_Mean - comparison$brms_Mean
print(comparison)
write.csv(comparison, "comparison_pois.csv", row.names = FALSE)
cat("\n--- POISSON MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
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

png("density_comparison.png", width = 1200, height = 800, res = 120, type = "cairo")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
draw_density("Intercept (uncentered)", post_brms$b_Intercept,         BF_post$b_Intercept)
draw_density("cofactor",               post_brms$b_cofactor,          BF_post$b_cofactor)
draw_density("sd_phylo",               post_brms$sd_phylo__Intercept, BF_post$sd_phylo)
draw_density("sd_obs",                 post_brms$sd_obs__Intercept,   BF_post$sd_obs)
dev.off()
cat("Panel saved to density_comparison.png\n")
# %%
