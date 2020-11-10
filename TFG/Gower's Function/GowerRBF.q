GowerRBF <- function (df, ords = c(), bins = c()) #, cycs)
{
  if(length(dx <- dim(df)) != 2 || !is.data.frame(df)) stop("df is not a dataframe!")
  
  variables <- dx[2]
  records <- dx[1]
  
  if(!is.null(bins))
  {
    databin <- df[, bins]
    lenB <- sapply(lapply(databin, function(y) levels(as.factor(y))), length)
    if(any(lenB > 2)) stop("at least one binary variable has more than 2 levels.")
    if(any(lenB < 2)) warning("at least one binary variable has not 2 different levels.")
    #Are the binary variables in a range of (0,1,NA)?
    if(any(is.f <- sapply(databin, is.factor)))
      databin[is.f] <- lapply(databin[is.f], function(f) as.integer(as.character(f)))
    if(!all(sapply(databin, function(y)is.logical(y) || all(sort(unique(as.numeric(y[!is.na(y)])))%in% 0:1))))
      stop("at least one binary variable has values not in {0,1,NA}")
  }
  
  type <- sapply(df, data.class)
  x <- data.matrix(df)
  
  if(!is.null(ords))
  {
    x[, names(type[ords])] <- unclass(as.ordered(x[, names(type[ords])]))
    type[ords] <- "O"
  }
  
  type[type == "numeric"] <- "I"
  type[type == "integer"] <- "I"
  type[type == "ordered"] <- "O"
  type[type == "factor"] <- "C"
  type[type == "logical"] <- "B"
  type[bins] <- "B"
  
  type <- paste(type, sep = ",")
  typeCodes <- c("B","C","O","I")
  type2 <- match(type,typeCodes);
  
  colR <- apply(x, 2, range, na.rm = TRUE) #forma de extraer maximos y minimos de las variables.
  colmin <- colR[1,] #elementos mínimos de las variables
  colmax <- colR[2,] #elementos máximos de las variables
  scale <- colmax - colmin
  
  md <- 0
  if(any(is.na(x))) {md <- 1}
  
  if(!is.null(comp))
  {
    dataset <- df[comp,]
    x <- data.matrix(dataset)
    md <- 0
    if(any(is.na(x))) {md <- 1}
    records <- length(comp)
  }
  diss <- GowerDist(x,type2,variables,records,scale,md,(records*(records-1))/2)
  full <- matrix(0,records,records)
  full[!lower.tri(full, diag = TRUE)] <- diss
  return(full)
}