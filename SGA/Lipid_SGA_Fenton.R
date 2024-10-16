
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(dplyr)
library(dplyr)
library(pdftools)
library(stringr)


data <- read.csv2("LipidomeMoms_New.csv", fileEncoding = "latin1", sep = ",")
data <- data[!is.na(data$BirthWeight),] # remove rows with NA BW



##### tables for SGA 

# Function to convert week_day to numeric week
convert_week_day <- function(week_day) {
  # Split by the '+' symbol
  parts <- strsplit(week_day, "\\+")[[1]]
  week <- as.numeric(parts[1])
  day <- as.numeric(parts[2])
  
  # Convert day to fraction of week and add to week
  week + day / 7
}


#####SGA weight

#intergrowth data boys weight
pdf_text_data <- pdf_text("TableForTheAssignmentOfSizeForGestationalAgeAtBirth_v3a.pdf")  # Load the PDF
lines <- str_split(pdf_text_data, "\n") # Split text into lines
table_lines <- c(lines[[1]][3:23])  # Assume the table is between line 14 and 52 and 14 to 44 in the 1st and 2nd page
table_data <- str_split_fixed(table_lines, "\\s+", n = 8) # Adjust `n` to the number of columns
table_df<- as.data.frame(table_data, stringsAsFactors = FALSE)
colnames(table_df) <- c("empty", "week", "third_girls", "tenth_girls", "ninetieth_girls","third_boys", "tenth_boys", "ninetieth_boys" )
 
data$SGA_weight = 0
for (i in 1:nrow(data)) {
  GA  <- as.numeric(data$GAdelivery[i]) 
  ### to late ? like GA =43 ?
  if (GA>=max(table_df$week)){GA=max(table_df$week)}
  
    sex <- data$Sex[i]
    # Find the smallest week that is greater than or equal to GA
    next_week <- min(table_df$week[table_df$week >= GA])
    
    if (sex == 2) { #Female
      # Extract the corresponding row and get the value from the 'tenth' column
      tenth_value <- as.numeric(table_df$third_girls[table_df$week == next_week])
    }
    if (sex == 1) { #Male
      # Extract the corresponding row and get the value from the 'tenth' column
      tenth_value <- as.numeric(table_df$third_boys[table_df$week == next_week])
    }
    if( data$BirthWeight[i]<=tenth_value ) {data$SGA_weight[i] = 1}
    
}


###### 
write.csv(data, file = "Lipid_mom_SGA.csv", row.names = FALSE)





