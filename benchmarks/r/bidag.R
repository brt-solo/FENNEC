# Bayesian structure learning with BiDAG (Kuipers, Suter & Moffa): posterior DAG samples.
# Usage: Rscript bench_bidag.R <in.csv> <out.csv> [n_samples]
# Pipeline (BiDAG's recommended one): iterativeMCMC to find the search space, then partition MCMC
# (Kuipers & Moffa 2017) with the BGe score (linear-Gaussian marginal likelihood). The last
# `n_samples` thinned samples of the chain are written, one DAG per row (d*d entries, row-major,
# entry [i, j] = 1 means i -> j).
args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; outfile <- args[2]
ns <- if (length(args) >= 3) as.integer(args[3]) else 50
suppressPackageStartupMessages(library(BiDAG))
X <- as.matrix(read.csv(infile, header = TRUE))
d <- ncol(X)
set.seed(1)
sp <- scoreparameters("bge", as.data.frame(X))
it <- iterativeMCMC(sp, verbose = FALSE)
pf <- partitionMCMC(sp, startspace = it$endspace, iterations = 30000, stepsave = 100, verbose = FALSE)
inc <- pf$traceadd$incidence
keep <- tail(seq_along(inc), ns)
rows <- t(sapply(keep, function(k) as.vector(t(as.matrix(inc[[k]])))))
write.table(rows, outfile, sep = ",", row.names = FALSE, col.names = FALSE)
