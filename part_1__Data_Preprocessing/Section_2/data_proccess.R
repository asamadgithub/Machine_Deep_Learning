# Import data set
df = read.csv("Data.csv")

df$Age <- ifelse(is.na(df$Age),
          ave(df$Age, FUN = function(x) mean(x, na.rm = TRUE)),       
          df$Age       )

df$Age


help(ave)
help(ifelse)

