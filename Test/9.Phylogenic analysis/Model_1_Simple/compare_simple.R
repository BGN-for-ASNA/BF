# %%
library(brms)
library(ape)

# 1. Load data and fit brms
phylo      <- read.nexus("phylo.nex")
data_simple <- read.table("data_simple.txt", header = TRUE)
A <- vcv.phylo(phylo)

cat("\nFitting brms Simple model...\n")
model_simple <- brm(
  phen ~ cofactor + (1 | gr(phylo, cov = A)),
  data = data_simple, family = gaussian(),
  data2 = list(A = A),
  prior = c(
    prior(normal(0, 50), "Intercept"),
    prior(normal(0, 10), "b"),
    prior(student_t(3, 0, 20), "sd"),
    prior(student_t(3, 0, 20), "sigma")
  ),
  chains = 2, cores = 2, iter = 3000, warmup = 1000,
  refresh = 0
)

# 2. Load BF posteriors
BF_post   <- read.csv("BF_post_simple.csv")
post_brms <- as.data.frame(model_simple)

# 3. Compare posterior means -> log.txt
cat("\n--- Comparison of Posterior Means ---\n")
comparison <- data.frame(
  Parameter = c("Intercept (uncentered)", "cofactor", "sd_phylo", "sigma"),
  brms_Mean = c(
    mean(post_brms$b_Intercept),
    mean(post_brms$b_cofactor),
    mean(post_brms$sd_phylo__Intercept),
    mean(post_brms$sigma)
  ),
  BF_Mean = c(
    mean(BF_post$b_Intercept),
    mean(BF_post$b_cofactor),
    mean(BF_post$sd_phylo),
    mean(BF_post$sigma)
  )
)
comparison$Difference <- comparison$BF_Mean - comparison$brms_Mean
print(comparison)
write.csv(comparison, "comparison_simple.csv", row.names = FALSE)
cat("\n--- SIMPLE MODEL COMPARISON ---\n", file = "log.txt", append = TRUE)
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
draw_density("sigma",                  post_brms$sigma,               BF_post$sigma)
dev.off()
cat("Panel saved to density_comparison.png\n")
# %%
