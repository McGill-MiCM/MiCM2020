# double
class(5); is.double(5)

class(5.0);
is.double(5.0)

5L/2L

# integer
class(5L); is.double(5L)

# How precise is double precision?
options(digits = 22) # show more digits in output
print(1/3)
options(digits = 7) # back to the default

object.size(rep(5, 5e6))
object.size(rep(5L, 5e6))

# logical
class(TRUE); class(F)

# character
class("TRUE")

# Not important for this workshop
fac <- as.factor(c(1, 5, 11, 3))
fac

class(fac)

# R has an algorithm to decide the order of the levels
fac.ch <- as.factor(c("B", "a", "1", "ab", "b", "A"))
fac.ch

# Scalar - a vector of length 1
myscalar <- 5
myscalar

class(myscalar)

# Vector
myvector <- c(1, 1, 2, 3, 5, 8)
myvector

class(myvector)

# Matrix - a 2d array
mymatrix <- matrix(c(1, 1, 2, 3, 5, 8), nrow = 2, byrow = FALSE)
mymatrix

class(mymatrix)

str(mymatrix)

# Array - not important for this workshop
myarray <- array(c(1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144), dim = c(2, 2, 3))
print(myarray) # print() is not needed if run in R or Rstudio.

class(myarray)

# List - very important for the workshop
mylist <- list(Title = "R Beyond the Basics",
               Duration = c(2, 2),
               sections = as.factor(c(1, 2, 3, 4)),
               Date = as.Date("2020-03-06"),
               Lunch_provided = FALSE,
               Feedbacks = c("Amazing!", "Great workshop!", "Yi is the best!", "Wow!")
)
print(mylist) # No need for print if running in R or Rstudio

class(mylist)

# Access data stored in lists
mylist$Title

# or
mylist[[6]]

# Further
mylist$Duration[1]
mylist[[6]][2]

# Elements in lists can have different data types
lapply(mylist, class) # We will talk about lapply() later

# Elements in list can have different lengths
lapply(mylist, length)

# Data frames - most commonly used for analyses
head(mtcars)

# Access a column (variable) in data frames
mtcars$mpg

# Vectorized operation
# system.time(operation)  returns the time needed to run the 'operation'
t <- system.time( x1 <- sqrt(1:1000000) )
head(x1)

# For loop with memory pre-allocation
x2 <- rep(NA, 1000000)
t0 <- proc.time()
for (i in 1:1000000) {
    x2[i] <- sqrt(i)
}
t1 <- proc.time()

identical(x1, x2) # Check whether results are the same

# For loop without memory pre-allocation
x3 <- NULL
t2 <- proc.time()
for (i in 1:1000000) {
    x3[i] <- sqrt(i)
}
t3 <- proc.time()

identical(x1, x2) # Check whether results are the sa

# As we can see, R is not very fast with loops.
t; t1 - t0; t3 - t2
# ?proc.time

# microbenchmark runs the code multiple times and take a summary
# Use well-developped R function
library(microbenchmark)
result <- microbenchmark(sqrt(500),
                         500^0.5,
                         unit = "ns", times = 1000
                        )
summary(result)
# Result in nanoseconds

data <- read.csv("https://raw.githubusercontent.com/ly129/MiCM2020/master/sample.csv", header = TRUE)
head(data, 8)

summary(data)

mean(data$Wr.Hnd)

mean(data$Height)

mean(data$Height, na.rm = TRUE)

# Choose the continuous variables
var.cts <- sapply(data, FUN = is.numeric)
var.cts <- c("Wr.Hnd", "NW.Hnd", "Pulse", "Height", "Age")

cts.data <- data[, var.cts]
head(cts.data)

# Calculate the mean
apply(cts.data, MARGIN = 2, FUN = mean)

apply(cts.data, MARGIN = 2, FUN = mean, na.rm = TRUE)

sex.tab <- table(data$Sex)
sex.tab

prop.table(sex.tab)

table(data$Smoke)

table(data$Sex, data$Smoke)

ss.tab <- table(data[, c("Sex", "Smoke")])
ss.tab

prop.table(ss.tab, margin = 2)

prop.table(ss.tab, margin = 1)

sample(letters[1:3], size = 100, replace = T)

tab3d <- table(data$Sex, data$Smoke,
               sample(letters[1:3], size = 100, replace = T))
tab3d

prop.table(tab3d, margin = c(1,3))

# Subset if we don't know these functions yet..
data.f <- data[data$Sex == "Female", ]
head(data.f)
sd(data.f$Wr.Hnd)

# aggregate() syntax 1
aggregate(data$Wr.Hnd, by = list(sex = data$Sex), FUN = sd)

# aggregate() syntax 2
aggregate(Wr.Hnd~Sex, data = data, FUN = sd)

# by()
by(data = data$Wr.Hnd, INDICES = list(sex = data$Sex), FUN = sd)

# tapply()
tapply(X = data$Wr.Hnd, INDEX = list(data$Sex), FUN = sd)

# Return a list using tapply()
tapply(X = data$Wr.Hnd, INDEX = list(data$Sex),
       FUN = sd, simplify = FALSE)

aggregate(data$Wr.Hnd,
          by = list(sex = data$Sex, smoke = data$Smoke),
          FUN = sd)

aggregate(Wr.Hnd~Sex+Smoke, data = data, FUN = sd)

cbind(1:5, 5:1); rbind(1:5, 5:1)

aggregate(cbind(data$Wr.Hnd, data$NW.Hnd),
          by = list(data$Sex, data$Smoke),
          FUN = sd)

aggregate(cbind(Wr.Hnd, NW.Hnd)~Sex+Smoke, data = data, FUN = sd)

aggregate(Wr.Hnd~Smoke, data = data, FUN = print)

aggregate(Wr.Hnd~Smoke, data = data, FUN = length)

par(mfrow = c(2,2))
aggregate(Wr.Hnd~Smoke, data = data, FUN = hist,
          main = "hist", breaks = 3)

vec <- 1:5
vec

ifelse(vec>3, yes = "big", no = "small")

data$Adult <- ifelse(data$Age >= 18, yes = "Yes", no = "No")

if (data$Age >= 18) {
    data$Adult2 = "Yes"
} else {
    data$Adult2 = "No"
}
head(data)

# Delete Adult2
data <- subset(data, select=-c(Adult2))

cut.points <- c(-Inf, 16, 18, 20, 22, Inf)

data$Hn.Grp <- cut(data$Wr.Hnd, breaks = cut.points, right = T)

head(data, 12)
# labels as default

# Set labels to false
data$Hn.Grp <- cut(data$Wr.Hnd, breaks = cut.points, labels = F)
head(data)

# Customized labels
custom.label <- c("TP/XS", "P/S", "M/M", "G/L", "TG/XL")
data$Hn.Grp <- cut(data$Wr.Hnd, breaks = cut.points,
                   right = T, labels = custom.label)
head(data)

aggregate(Wr.Hnd~Hn.Grp, data = data, FUN = mean)

numbers <- 1:10
groups <- sample(letters[1:3], size = 10, replace = T)
rbind(numbers, groups)

split(x = numbers, f = groups)

wr.hnd.grp <- split(data$Wr.Hnd, f = data$Hn.Grp)
wr.hnd.grp

# lapply
la <- lapply(X = wr.hnd.grp, FUN = mean);
la

# sapply
sapply(X = wr.hnd.grp, FUN = mean, simplify = T)

sapply(X = wr.hnd.grp, FUN = mean, simplify = F)

summary(1:10)

# vapply *
# Safer than sapply(), and a little bit faster
# because FUN.VALUE has to be specified that length and type should match

va <- vapply(wr.hnd.grp, summary, FUN.VALUE = c("Min." = numeric(1),
                                                "1st Qu." = numeric(1),
                                                "Median" = numeric(1),
                                                "Mean" = numeric(1),
                                                "3rd Qu." = numeric(1),
                                                "Max." = numeric(1)))
va

# aggregate(Wr.Hnd~Smoke, data = data, FUN = ...)
# tapply(X = data$Wr.Hnd, INDEX = list(data$Smoke), FUN = ...)

sample.mean <- aggregate(Wr.Hnd~Smoke, data = data, FUN = mean)[,2]
sample.var <- aggregate(Wr.Hnd~Smoke, data = data, FUN = var)[,2]
sample.size <- aggregate(Wr.Hnd~Smoke, data = data, FUN = length)[,2]

# sample.mean; sample.var; sample.size

t <- qt(p = 0.025, df = sample.size - 1, lower.tail = FALSE)

lb <- sample.mean - t * sqrt(sample.var/sample.size); lb
ub <- sample.mean + t * sqrt(sample.var/sample.size); ub


# How many times did we aggregate according to the group?
# Can we aggregate only once?

# The structure
func_name <- function(argument){
    statement
}

# Build the function
times2 <- function(x) {
    fx <- 2 * x
    return(fx)
}
# Use the function
times2(x = 5)
# or
times2(3)

# R has operators that do this
9 %/% 2
9 %% 2

int.div <- function(a, b){
    int <- floor(a/b)
    mod <- a - int*b
    return(list(integer = int, modulus = mod))
}

# class(result)
# Recall: how do we access the modulus?
result <- int.div(21, 4)
result

int.div <- function(a, b){
    int <- a%/%b
    mod <- a%%b
    return(cat(a, "%%", b, ": \n integer =", int,"\n ------------------", " \n modulus =", mod, "\n"))
}
int.div(33,5)

int.div <- function(a, b){
    int <- a%/%b
    mod <- a%%b
    out <- rbind(a, b, int, mod)
    out
}
int.div(21:25, 1:5)

# No need to worry about the details here.
# Just want to show that functions do not always have to return() something.
AIcanadian <- function(who, reply_to) {
    system(paste("say -v", who, "Sorry!"))
}
# AIcanadian("Alex", "Sorry I stepped on your foot.")

# Train my chatbot - AlphaGo style.
# I'll let Alex and Victoria talk to each other.
# MacOS has their voices recorded.
# chat_log <- rep(NA, 8)
# for (i in 1:8) {
#     if (i == 1) {
#         chat_log[1] <- "Sorry I stepped on your foot."
#         system("say -v Victoria Sorry, I stepped on your foot.")
#     } else {
#         if (i %% 2 == 0)
#             chat_log[i] <- AIcanadian("Alex", chat_log[i - 1])
#         else
#             chat_log[i] <- AIcanadian("Victoria", chat_log[i - 1])
#     }
# }
# chat_log

data_summary <- function(func) {
    data <- read.csv("https://raw.githubusercontent.com/ly129/MiCM2020/master/sample.csv", header = TRUE)
    by(data = data$Wr.Hnd, INDICES = list(data$Smoke), FUN = func)
}
data_summary(mean)

a_times_2_unless_you_want.something.else.but.I.refuse.3 <- function(a, b=2){
    if (b == 3) {
        stop("I refuse 3!")
    }
    
    if (b == 4) {
        warning("4 sucks too.")
    }
    
    a*b
}

a_times_2_unless_you_want.something.else.but.I.refuse.3(a = 5)

a_times_2_unless_you_want.something.else.but.I.refuse.3(a = 5, b = 4)

a_times_2_unless_you_want.something.else.but.I.refuse.3(a = 5, b = 3)

# Multiple optional arguments fed to different functions called in our own function
# Still needs a lot of refinement

fancy.mean <- function(vec, ...) {
    args <- list(...)
    print(args)
#     print(as.list(match.call(expand.dots = FALSE)))
    mean.args <- c("trim", "na.rm")
    sample.args <- c("size", "replace", "prob")
    
# Could use a for loop to go over all args.
    
#     cat("arguments in mean()", mean.args, "\n")
    
    args.names <- names(args)
    
#     cat("argument names in ...", args.names, "\n")
    
    m.args <- args.names[args.names %in% mean.args]
    s.args <- args.names[args.names %in% sample.args]
    
#     cat("args names that will be used in mean", m.args, "\n")
    
    print(args[["m.args"]])
    m <- list(trim = 0, na.rm = args[[m.args]])
    s <- list(args[[s.args]], replace = FALSE, prob = NULL)
    
#     print(m)
#     print("args that will be used in mean", m, "\n")
    
    fancy.mean <- mean(vec, trim = 0, na.rm = args[[m.args]])
    fancy.sample <- sample(vec, size = args[[s.args]])
    return(list(fancy.mean,fancy.sample))
}

fancy.mean(c(1:5, NA, NA, 234,123,4,123,41,234),
           na.rm = T, size = 1)

sample.ci <- function(x, digits = 2) {
    mu <- mean(x)
    variance <- var(x)
    size <- length(x)
    t.stat <- qt(p = 0.025, df = size - 1, lower.tail = FALSE)
    
    lb <- mu - t.stat * sqrt(variance/size)
    ub <- mu + t.stat * sqrt(variance/size)
    
    return(round(c(lower = lb, upper = ub), digits))
    
    # Think carefully whether this output can be put
    # inside of a data.frame
}

aggregate(Wr.Hnd~Smoke, data = data, FUN = sample.ci)

set.seed(20200306)
N <- 200
height <- round(rnorm(n = N, mean = 180, sd = 10)) # in centimeter
weight <- round(rnorm(n = N, mean = 80, sd = 10)) # in kilograms
age <- round(rnorm(n = N, mean = 50, sd = 10))
treatment <- sample(c(TRUE, FALSE), size = N,
                    replace = T, prob = c(0.3,0.7))
HF <- sample(c(TRUE, FALSE), size = N, replace = T,
             prob = c(0.1,0.9))

fake <- data.frame(height, weight, age, treatment, HF)
head(fake)

names(fake)
fake$BMI <- fake$weight/(fake$height/100)^2
head(fake)

cut.pts <- c(-Inf, 18.5, 25, 30, Inf)
labs <- c("Underweight", "Normal weight", "Overweight", "Obesity")
fake$BMI.cat <- cut(fake$BMI, breaks = cut.pts, labels = labs, right = F)
head(fake)

# aggregate()
aggregate(BMI~BMI.cat, data = fake, FUN = mean)

# split() and lapply()
BMI.grp <- split(fake$BMI, f = fake$BMI.cat)
lapply(BMI.grp, FUN = mean)

# Trick:
FALSE+TRUE+TRUE

aggregate(HF~BMI.cat+treatment, data = fake, FUN = sum)

tab2by2 <- function(data, treatment, outcome){
    sub <- data[, c(treatment, outcome)]
    return(table(sub))
}

tab2by2(fake, treatment = "treatment", outcome = "HF")

tab2by2.pro <- function(data, treatment, outcome, treatment.threshold, outcome.threshold){
    tx <- data[, treatment]
    rx <- data[, outcome]
    
    if (length(table(tx))>2) {
        if (missing(treatment.threshold)) {
            stop("Non-binary treatment. Please provide a threshold.")
        } else {
            binary.treatment <- ifelse(tx<=treatment.threshold,
                                       yes = paste("<=", treatment.threshold),
                                       no = paste(">", treatment.threshold))
        }
    } else {
        binary.treatment <- tx
    }
    
    if (length(table(rx))>2) {
        if (missing(outcome.threshold)) {
            stop("Non-binary outcome. Please provide a threshold.")
        } else {
            binary.outcome <- ifelse(rx<=outcome.threshold,
                                     yes = paste("<=", outcome.threshold),
                                     no = paste(">", outcome.threshold))
        }
    } else {
        binary.outcome <- rx
    }
    
    
    return(table(treatment = binary.treatment, outcome = binary.outcome))
}

tab2by2.pro(fake, treatment = "age", outcome = "BMI")

tab2by2.pro(fake, treatment = "age", outcome = "BMI", treatment.threshold = 50)

tab2by2.pro(fake, treatment = "age", outcome = "BMI", treatment.threshold = 50, outcome.threshold = 20)

tab2by2.pro(fake, treatment = "age", outcome = "HF")

# HF is binary, so it is OK if "outcome.threshold" is missing.
tab2by2.pro(fake, treatment = "age", outcome = "HF", treatment.threshold = 50)


