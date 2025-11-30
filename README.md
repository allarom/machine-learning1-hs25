# machine-learning1-hs25
To install libraries you need to run file 00_load_packages.R.
To compile markdown files in R-Studio: in RMD files you have a button called "Knit", there you can specify the format.

In my setup (Mac, Visual Code Studio) I needed to specify Pandoc location in .Renviron file (placed in my root folder ~, so path was ~/.Renviron)

There I added:

```RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/x86_64```


You can complile analysis.html file by running this command in the R:

```source("src/00_load_packages.R")
compile_analysis_report()```

Or on Mac by running ```Cmd+Shift+K```, K stands for "knitting"

