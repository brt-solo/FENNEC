# Run an R structure-learning method on a CSV and write the adjacency matrix.
# Usage: Rscript bench_r_methods.R <method: cam|mmhc_bnlearn|prune> <in.csv> <out.csv> [<dag.csv> for prune]
#   cam           CAM (Buehlmann, Peters & Ernest 2014), standard configuration used in the
#                 GraN-DAG / NOTEARS comparisons: preliminary neighbourhood selection (boosting)
#                 + greedy additive-model search + significance pruning.
#   mmhc_bnlearn  bnlearn's reference MMHC (Gaussian), to cross-check our Python MMHC.
#   prune         CAM's significance pruning (additive-model p-values, cut-off 0.001) of a given DAG:
#                 the post-processing used by SCORE, NoGAM and GraN-DAG.
args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; infile <- args[2]; outfile <- args[3]
X <- as.matrix(read.csv(infile, header = TRUE))
d <- ncol(X)
if (method == "cam") {
  suppressPackageStartupMessages(library(CAM))
  res <- CAM(X, scoreName = "SEMGAM", numCores = 1, output = FALSE,
             variableSel = TRUE, variableSelMethod = selGamBoost,
             pruning = TRUE, pruneMethod = selGam, pruneMethodPars = list(cutOffPVal = 0.001, numBasisFcts = 10))
  A <- as.matrix(res$Adj)
} else if (method == "mmhc_bnlearn") {
  suppressPackageStartupMessages(library(bnlearn))
  df <- as.data.frame(X)
  colnames(df) <- paste0("V", seq_len(d))
  bn <- mmhc(df)
  A <- matrix(0, d, d)
  arcs <- bnlearn::arcs(bn)
  for (r in seq_len(nrow(arcs))) {
    A[as.integer(sub("V", "", arcs[r, 1])), as.integer(sub("V", "", arcs[r, 2]))] <- 1
  }
} else if (method == "prune") {
  suppressPackageStartupMessages(library(CAM))
  G <- as.matrix(read.csv(args[4], header = FALSE))
  A <- as.matrix(CAM:::pruning(X, G, output = FALSE, pruneMethod = CAM:::selGam,
                         pruneMethodPars = list(cutOffPVal = 0.001, numBasisFcts = 10)))
} else stop("unknown method")
write.table(A, outfile, sep = ",", row.names = FALSE, col.names = FALSE)
