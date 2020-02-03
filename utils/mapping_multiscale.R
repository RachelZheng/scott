#!/usr/bin/env Rscript
library(gmra)
library(mop)
args = commandArgs(trailingOnly=TRUE)

if(length(args) == 0){
	stop("Input folder must be supplied.n", call.=FALSE)
} else{
	folder_data <- args[1]
	X1 <- read.table(file.path(folder_data, 'X1.txt'), header = FALSE)
	X2 <- read.table(file.path(folder_data, 'X2.txt'), header = FALSE)
	gmra1 = gmra.create.ipca(X=X1, eps=0, d=2, maxKids=2)
	gmra2 = gmra.create.ipca(X=X2, eps=0, d=2, maxKids=2)
	trp.lp <- multiscale.transport.create.lp(oType=30)
	icprop <- multiscale.transport.create.iterated.capacity.propagation.strategy(1, 0)
	multiscale.transport.set.propagation.strategy.1(trp.lp, icprop)
	trp <- multiscale.transport.solve(trp.lp, gmra1, gmra2, p = 2, nType=0, dType=1)
	idx <- length(trp$from)
	# from <- trp$from[idx][[1]]
	from_idx <- trp$fromIndex[idx][[1]]
	# to <- trp$to[idx][[1]]
	to_idx <- trp$toIndex[idx][[1]]
	map <- trp$map[idx][[1]]
	write.csv(from_idx, file.path(folder_data, 'X1_idx.csv'), row.names = FALSE)
	write.csv(to_idx, file.path(folder_data, 'X2_idx.csv'), row.names = FALSE)
	write.csv(map, file.path(folder_data, 'map.csv'), row.names = FALSE)
}

