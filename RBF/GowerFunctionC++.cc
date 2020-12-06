#include <Rcpp.h>
using namespace Rcpp;

double abs (double x)
{
  if (x<0) x = x*(-1.0);
  return x;
}

// [[Rcpp::export]]
NumericVector GowerDist (NumericMatrix x, NumericVector Types, int variables, int records, NumericVector scale, int md, int outsize)
{
  NumericVector Output (outsize);
  bool hasNA = md == 1;
  int outindex = 0;
  
  for (int i = 1; i < records;++i)
  {
    for (int j = 0; j < i; ++j)
    {
      double varnums = 0;
      double score = 0;
      
      for (int k = 0; k < variables; ++k)
      {
        double val1 = x(i,k);
        double val2 = x(j,k);
        
        if(hasNA)
        {
          if (ISNAN(val1)) continue;
          if (ISNAN(val2)) continue;
        }
        
        varnums += 1.0;
        
        if (Types[k] <= 2)
        {
          if (val1 != val2) score += 1.0;
        }
        
        else
        {
          double sc = scale[k];
          score += (abs(val1 - val2))/sc; 
        }
      }
      Output[outindex] = score/varnums;
      ++outindex;
    }
  }
  return Output;
}