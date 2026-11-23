# %%
library(brms)
library(ape)

# 1. Load data and fit brms
data_slopes <- read.table("data_slopes.txt", header = TRUE)
data_slopes$x <- data_slopes$x - mean(data_slopes$x)
phylo <- read.nexus("phylo_slopes.nex")
A <- vcv.phylo(phylo)
A <- A / max(A)

cat("\nFitting brms Varying Slopes model...\n")
model_slopes <- brm(
  y ~ x + (1 + x | gr(phylo, cov = A)),
  data = data_slopes,
  data2 = list(A = A),
  family = gaussian(),
  prior = c(
    prior(normal(0, 10), class = Intercept),
    prior(normal(0, 10), class = b),
    prior(student_t(3, 0, 10), class = sd),
    prior(lkj(2), class = cor)
  ),
  chains = 2, cores = 1, iter = 3000, warmup = 1000,
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99),
  refresh = 0
)

# 2. Load BF posteriors
BF_post   <- read.csv("BF_post_slopes.csv")
post_brms <- as.data.frame(model_slopes)

# 3. Compare posterior means -> log.txt
cat("\n--- Comparison of Posterior Means (Varying Slopes) ---\n")
brms_intercept <- if ("b_Intercept" %in% names(post_brms)) post_brms$b_Intercept else post_brms$Intercept
brms_bx        <- if ("b_x" %in% names(post_brms)) post_brms$b_x else post_brms$x
rho_col        <- "cor_phylo__Intercept__x"
brms_rho       <- if (rho_col %in% names(post_brms)) post_brms[[rho_col]] else rep(NA, nrow(post_brms))

comparison <- data.frame(
  Parameter = c("Intercept", "b_x", "sigma", "sd_intercept", "sd_slope", "rho"),
  brms_Mean = c(
    mean(brms_intercept, na.rm = TRUE),
    mean(brms_bx, na.rm = TRUE),
    mean(post_brms$sigma),
    mean(post_brms$sd_phylo__Intercept),
    mean(post_brms$sd_phylo__x),
    mean(brms_rho, na.rm = TRUE)
  ),
  BF_Mean = c(
    mean(BF_post$Intercept,    na.rm = TRUE),
    mean(BF_post$b_x,          na.rm = TRUE),
    mean(BF_post$sigma,        na.rm = TRUE),
    mean(BF_post$sd_intercept, na.rm = TRUE),
    mean(BF_post$sd_slope,     na.rm = TRUE),
    mean(BF_post$rho,          na.rm = TRUE)
  )
)
comparison$Difference <- comparison$BF_Mean - comparison$brms_Mean
comparison$Diff_Pct   <- abs(comparison$Difference) / abs(comparison$brms_Mean) * 100
print(comparison)
write.csv(comparison, "comparison_slopes.csv", row.names = FALSE)
cat("\n--- MODEL 5 COMPARISON ---\n", file = "log.txt", append = TRUE)
write.table(comparison, file = "log.txt", append = TRUE,
            row.names = FALSE, sep = "\t", quote = FALSE)

# 4. Combined density panel (all parameters)
cat("\nGenerating combined density comparison panel...\n")
draw_density <- function(label, brms_s, BF_s) {
  brms_s <- as.numeric(na.omit(as.vector(brms_s)))
  BF_s   <- as.numeric(na.omit(BF_s))
  if (length(brms_s) < 2 || length(BF_s) < 2) {
    plot.new(); title(main = paste(label, "(no data)")); return()
  }
  d1 <- density(brms_s); d2 <- density(BF_s)
  xlim <- range(c(d1$x, d2$x)); ylim <- range(c(d1$y, d2$y))
  plot(d1, col = "red", lwd = 2, main = label,
       xlim = xlim, ylim = ylim, xlab = "Value", ylab = "Density")
  lines(d2, col = "blue", lwd = 2, lty = 2)
  legend("topright", legend = c("brms", "BF"),
         col = c("red", "blue"), lwd = 2, lty = c(1, 2), cex = 0.8)
}

png("density_comparison.png", width = 1800, height = 800, res = 120, type = "cairo")
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
draw_density("Intercept",    brms_intercept,              BF_post$Intercept)
draw_density("b_x",          brms_bx,                     BF_post$b_x)
draw_density("sigma",        post_brms$sigma,             BF_post$sigma)
draw_density("sd_intercept", post_brms$sd_phylo__Intercept, BF_post$sd_intercept)
draw_density("sd_slope",     post_brms$sd_phylo__x,       BF_post$sd_slope)
draw_density("rho",          brms_rho,                    BF_post$rho)
dev.off()
cat("Panel saved to density_comparison.png\n")
# %%
