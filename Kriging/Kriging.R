library(gstat)
library(moments)
library(raster)

# Define data folder
dataFolder <- "E:\\Kriging\\coordinate_31\\"
train <- read.csv(paste0(dataFolder, "Camk2n1.txt"), header=TRUE, sep='\t')
train
class(train)
coordinates(train) = ~x+y
class(train)
is(train, "Spatial")
longlat = !is.projected(train)
if(is.na(longlat)) longlat = FALSE
diagonal = spDists(t(bbox(train)), longlat = longlat)[1,2]
bbox(train)
train
boundaries = seq(2,100,by=2) * diagonal * 0.4/100
boundaries


# GLS.model_1 <- vgm(psill = 1, model = "Sph", range = 300, nugget = 0.1)
# g = gstat(NULL, "bla", formula, train, model = GLS.model, set = list(gls=1))
# experimental_variogram = variogram(g, boundaries = boundaries, ...)
formula = z~1
input_data = train
GLS.model = NA
# experimental_variogram = variogram(formula, input_data,boundaries = boundaries, ...)
verbose = FALSE

if(!is(GLS.model, "variogramModel")) {
  experimental_variogram = variogram(formula, input_data,boundaries = boundaries)
} else {
  if(verbose) cat("Calculating GLS sample variogram\n")
  g = gstat(NULL, "bla", formula, input_data, model = GLS.model, set = list(gls=1))
  experimental_variogram = variogram(g, boundaries = boundaries)
}

while(TRUE) {
  if(length(experimental_variogram$np[experimental_variogram$np < miscFitOptions[["min.np.bin"]]]) == 0 | length(boundaries) == 1) break
  boundaries = boundaries[2:length(boundaries)]			
  if(!is(GLS.model, "variogramModel")) {
    experimental_variogram = variogram(formula, input_data,boundaries = boundaries, ...)
  } else {
    experimental_variogram = variogram(g, boundaries = boundaries, ...)
  }
}

experimental_variogram

miscFitOptionsDefaults = list(merge.small.bins = TRUE, min.np.bin = 5)
miscFitOptions = list()
miscFitOptions = modifyList(miscFitOptionsDefaults, miscFitOptions)
miscFitOptions[["merge.small.bins"]]

start_vals = c(NA,NA,NA)
min(experimental_variogram$gamma)
if(is.na(start_vals[1])) {  # Nugget
  initial_nugget = min(experimental_variogram$gamma)
  initial_nugget_up = initial_nugget
  initial_nugget_down = quantile(experimental_variogram$gamma,probs = c(0.01))
  #initial_nugget_seq = seq(initial_nugget_up,initial_nugget_down, by=abs(initial_nugget_down-initial_nugget_up/10) )
  initial_nugget_seq = seq(0,initial_nugget_down, by=abs(initial_nugget_down-0/50))
  
} else {
  initial_nugget = start_vals[1]
}
if(is.na(start_vals[2])) { # Range
  initial_range = 0.1 * diagonal   # 0.10 times the length of the central axis through the area
  initial_range_central = initial_range
  initial_range_up = 0.05 * diagonal
  initial_range_down = 0.15 * diagonal
  initial_range_seq = seq(initial_range_up, initial_range_down, by=abs(initial_range_up-initial_range_central)/10)
  
} else {
  initial_range = start_vals[2]
}
if(is.na(start_vals[3])) { # Sill
  initial_sill = mean(c(max(experimental_variogram$gamma), median(experimental_variogram$gamma)))
  initial_sill_mean = mean(experimental_variogram$gamma)
  initial_sill_up = quantile(experimental_variogram$gamma,probs = c(0.5))
  initial_sill_down = quantile(experimental_variogram$gamma,probs = c(0.95))
  initial_sill_seq = c(initial_sill,initial_sill_mean,seq(initial_sill_up,initial_sill_down, by=abs(initial_sill_down-initial_sill_up)/10))
  
} else{
  initial_sill = start_vals[3]
}
print(paste("initial_nugget",initial_nugget,sep=":"))
print(paste("initial_range",initial_range,sep=":"))
print(paste("initial_sill",initial_sill,sep=":"))

fix.values = c(NA,NA,NA)

if(!is.na(fix.values[1]))
{
  fit_nugget = FALSE
  initial_nugget = fix.values[1]
} else
  fit_nugget = FALSE
#fit_nugget = TRUE

# Range
if(!is.na(fix.values[2]))
{
  fit_range = FALSE
  initial_range = fix.values[2]
} else
  fit_range = FALSE
#fit_range = TRUE

# Partial sill
if(!is.na(fix.values[3]))
{
  fit_sill = FALSE
  initial_sill = fix.values[3]
} else
  fit_sill = FALSE
#fit_sill = TRUE

getModel = function(psill, model, range, kappa, nugget, fit_range, fit_sill, fit_nugget, verbose)
{#initial_sill - initial_nugget, m, initial_range, kappa = 0, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose
  # 这个是干什么用的
  if(verbose) debug.level = 1 else debug.level = 0
  if(model == "Pow") {
    warning("Using the power model is at your own risk, read the docs of autofitVariogram for more details.")
    if(is.na(start_vals[1])) nugget = 0
    if(is.na(start_vals[2])) range = 1    # If a power mode, range == 1 is a better start value
    if(is.na(start_vals[3])) sill = 1
  }
  obj = try(fit.variogram(experimental_variogram,
                          model = vgm(psill=psill, model=model, range=range,
                                      nugget=nugget,kappa = kappa),
                          fit.ranges = FALSE, fit.sills = FALSE,
                          #fit.ranges = c(fit_range), fit.sills = c(fit_nugget, fit_sill),
                          debug.level = 0), 
            TRUE)
  print(obj)
  if("try-error" %in% class(obj)) {
    #print(traceback())
    warning("An error has occured during variogram fitting. Used:\n", 
            "\tnugget:\t", nugget, 
            "\n\tmodel:\t", model, 
            "\n\tpsill:\t", psill,
            "\n\trange:\t", range,
            "\n\tkappa:\t",ifelse(kappa == 0, NA, kappa),
            "\n  as initial guess. This particular variogram fit is not taken into account. \nGstat error:\n", obj)
    return(NULL)
  } else return(obj)
}
model = c('Sph')
test_models = model
SSerr_list = c()
vgm_list = list()
counter = 1
for(m in test_models) {
  if(m != "Mat" && m != "Ste") {        # If not Matern and not Stein
    for(initial_nugget in initial_nugget_seq){
      for(initial_sill in initial_sill_seq){
        for(initial_range in initial_range_seq){
          
          #model_fit = getModel(initial_sill - initial_nugget, m, initial_range, kappa = 0, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose)
          #print(paste("initial_range:",initial_range,sep=""))
          model_fit = getModel(initial_sill, m, initial_range, kappa = 0, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose)
          if(!is.null(model_fit)) {	# skip models that failed
            vgm_list[[counter]] = model_fit
            SSerr_list = c(SSerr_list, attr(model_fit, "SSErr"))
            #experimental_variogram = variogram(formula, input_data,boundaries = boundaries,cloud=T, ...)
            print(paste(Sys.time()," model",model_fit$model[-1],"psill:",model_fit$psill[-1],"range:",model_fit$range[-1],"nugget",model_fit$psill[1], sep=" "))
          }
          counter = counter + 1
          print(attr(model_fit, "SSErr"))
        }}}
  } else {                 # Else loop also over kappa values
    for(k in kappa) {
      model_fit = getModel(initial_sill - initial_nugget, m, initial_range, k, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose)
      if(!is.null(model_fit)) {
        vgm_list[[counter]] = model_fit
        SSerr_list = c(SSerr_list, attr(model_fit, "SSErr"))}
      counter = counter + 1
    }
  }
}
vgm_list[[1]]
SSerr_list
class(vgm_list[[1]])

SSerr_list

cor_list = c()
nmax_list = c()
pvalue_list = c()
print(paste(Sys.time(),"Start cross validation....."))
counter = 1
# 为什么cv这跑的这么慢
for(model_fit in vgm_list){
  print(length(vgm_list))
  cor_result = c()    
  pvalue_result = c()
  for(nmax in seq(10,14,2)){
    print(nmax)
    cv <- krige.cv(formula, locations = input_data, model=vgm(psill=model_fit$psill[-1], model=model_fit$model[-1], range=model_fit$range[-1], nugget=model_fit$psill[1]), nmin=5, nmax=nmax)
    print(cv)
    cor_test = cor.test(cv$var1.pred, cv$observed, method=c("pearson"))
    cor_result = c(cor_result,cor_test$estimate)
    print(cor_result)
    pvalue_result = c(pvalue_result,cor_test$p.value)
  }
  cor_list = c(cor_list, max(cor_result))
  nmax_list = c(nmax_list, seq(10,14,2)[which.max(cor_result)])
  print(nmax_list)
  pvalue_list = c(pvalue_list,pvalue_result[which.max(cor_result)])
  counter = counter + 1
}


# Reduce the number of points in the variogram cloud by subsetting the data
set.seed(42) # for reproducibility
train_subset <- train[sample(1:nrow(train), size = floor(0.1 * nrow(train))), ]

v.cloud <- variogram(z ~ 1, data = train_subset, cloud=T)
head(v.cloud)

# Save plot to file
png(filename = "E:\\Kriging\\variogram_cloud_reduced.png")
plot(v.cloud, main = "Variogram cloud (reduced points)", xlab = "Separation distance (m)")
dev.off()

v.cut<-variogram(z ~ 1, train, cutoff=800, width=800/20)
plot(v.cut, main = "Variogram with cutoff and fixed width", xlab = "Separation distance (m)")


v.map<-variogram(z ~ 1, train, map = TRUE, cutoff=800, width=800/20)
plot(v.map, col.regions = bpy.colors(64),
     main="Variogram Map",
     xlab="x",
     ylab="y")

plot(variogram(z ~ 1, train, 
               alpha = c(0, 30, 45, 60, 90, 105, 135, 120, 150),
               cutoff = 800),  
     main = "Directional Variograms, z",
    # sub = "Azimuth 30N (left), 120N (right)", 
     pch = 20, col = "blue")