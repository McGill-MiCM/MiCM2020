# double
class(5); is.double(5)

# integer
class(5L); is.double(5L)

# How precise is double precision?
options(digits = 22) # show more digits in output
print(1/3)
options(digits = 7) # back to the default

object.size(rep(5, 10))
object.size(rep(5L, 10))

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

# For loop
x2 <- rep(NA, 1000000)
t0 <- proc.time()
for (i in 1:1000000) {
    x2[i] <- sqrt(i)
}
t1 <- proc.time()

identical(x1, x2) # Check whether results are the same

# As we can see, R is not very fast with loops.
t; t1 - t0
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







# Choose the continuous variables

# Calculate the mean









# aggregate() syntax 1

# aggregate() syntax 2

# by()

# tapply()

# Return a list using tapply()
















vec <- 1:5
vec

ifelse(vec>3, yes = "big", no = "small")



if (data$Age >= 18) {
    data$Adult2 = "Yes"
} else {
    data$Adult2 = "No"
}
head(data)

# Delete Adult2
data <- subset(data, select=-c(Adult2))

cut.points <- c(0, 16, 18, 20, 22, Inf)

head(data)
# labels as default

# Set labels to false


# Customized labels
label <- c("TP/XS", "P/S", "M/M", "G/L", "TG/XL")




# cut.points <- c(0, 16, 18, 20, 22, Inf)


# lapply


# sapply




# vapply *
# Safer than sapply(), and a little bit faster
# because FUN.VALUE has to be specified that length and type should match

# va <- vapply(Wr.Hnd.Grp, summary, FUN.VALUE = c("Min." = numeric(1),
#                                                 "1st Qu." = numeric(1),
#                                                 "Median" = numeric(1),
#                                                 "Mean" = numeric(1),
#                                                 "3rd Qu." = numeric(1),
#                                                 "Max." = numeric(1)))
# va

# aggregate(Wr.Hnd~Smoke, data = data, FUN = ...)
# tapply(X = data$Wr.Hnd, INDEX = list(data$Smoke), FUN = ...)


# How many times did we aggregate according to the group? Can on aggregate only once?

# The structure
func_name <- function(argument){
    statement
}

# Build the function
times2 <- function(x) {
    fx = 2 * x
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
result$integer

int.div <- function(a, b){
    int <- a%/%b
    mod <- a%%b
    return(cat(a, "%%", b, ": \n integer =", int,"\n ------------------", " \n modulus =", mod, "\n"))
}
int.div(21,4)

int.div <- function(a, b){
    int <- a%/%b
    mod <- a%%b
    return(c(a, b))
}
int.div(21, 4)

# No need to worry about the details here.
# Just want to show that functions do not always have to return() something.
AIcanadian <- function(who, reply_to) {
    system(paste("say -v", who, "Sorry!"))
}
# AIcanadian("Alex", "Sorry I stepped on your foot.")

# Train my chatbot - AlphaGo style.
# I'll let Alex and Victoria talk to each other.
# MacOS has their voices recorded.
chat_log <- rep(NA, 8)
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
    a*b
}

a_times_2_unless_you_want.something.else.but.I.refuse.3(a = 5)

a_times_2_unless_you_want.something.else.but.I.refuse.3(a = 5, b = 4)

# a_times_2_unless_you_want.something.else.but.I.refuse.3(a = 5, b = 3)







set.seed(20200306)
N <- 200
height <- round(rnorm(n = N, mean = 180, sd = 10)) # in centimeter
weight <- round(rnorm(n = N, mean = 80, sd = 10)) # in kilograms
age <- round(rnorm(n = N, mean = 50, sd = 10))
treatment <- sample(c(TRUE, FALSE), size = N, replace = T, prob = c(0.3,0.7))
HF <- sample(c(TRUE, FALSE), size = N, replace = T, prob = c(0.1,0.9))

fake <- data.frame(height, weight, age, treatment, HF)
head(fake)





# Trick:
FALSE+TRUE+TRUE








