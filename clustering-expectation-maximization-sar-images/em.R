
###### Histogram Clustering with Expectation-Maximization on SAR Images ######

H <- matrix(readBin("histograms.bin", "double", 640000), 40000, 16)
H[,] <-  H[,] + 0.01  # add a small constant (such as 0.01) to the input histograms.

# Implement the EM algorithm
MultinomialEM <- function(H, K, tau){
  n <- dim(H)[1]
  d <- dim(H)[2]
  c <- rep(1,K) / K;
  # Choose K of the histograms at random and normalize each. These are our initial centroids t1, ..., tK.
  centroid = H[sample(1:n,K),]
  for (i in (1:K)){
    centroid[i,] <- centroid[i,] / sum(centroid[i,])
  }
  A <- matrix(0, nrow = n, ncol = K)
  # Iterate
  while (TRUE){
    # E-step
    Phi <- exp(H %*% t(log(centroid*121)))
    A_old <- A
    A <- (Phi * c) * matrix(rep(1/rowSums(Phi * c), K), nrow = n, ncol = K)
    # M-step
    c <- colSums(A) / n
    b <- t(A) %*% H
    centroid <-  b * matrix(rep(1/rowSums(b), d), nrow = K, ncol = d)
    # measure for terminating the iteration
    if (norm(A-A_old, type="o")<tau){
      break
    }
  }
  # Turn the soft assignments into a vector m of hard assignments
  m <- rep(0,n)
  for (i in (1:n)){
    m[i] <- which.max(A[i,])
  }
  return(m)
}

# Run for K=3,4,5. You may have to try different values of tau to obtain a reasonable result. Visualize the results as an image.
VisualizeMEM <- function(m, K, tau){
  M <- matrix(m, nrow = 200, ncol = 200)
  # image(x=1:200, y=1:200, M, col=grey((0:2^4)/2^4), xlab="row", ylab="col") # 4-bit image
  # rotate the axes
  img <- matrix(0, nrow = 200, ncol = 200)
  for (i in (1:200)){
    img[,i] <- M[,200+1-i]
  }
  image(x=1:200, y=1:200, img, col=grey((0:2^4)/2^4), xlab="", ylab="", axes = FALSE) 
}

tau <- c(1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001) # try different values of tau
for (K in (3:5)){
  for (i in (1:length(tau))){
    m <- MultinomialEM(H, K, tau[i])
    VisualizeMEM(m)
  }
}
