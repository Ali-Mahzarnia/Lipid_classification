
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(dplyr)
library(dplyr)
library(pdftools)
library(stringr)


data <- read.csv2("LipidomeMoms_New.csv", fileEncoding = "latin1", sep = ",")
data <- data[!is.na(data$BirthWeight),] # remove rows with NA BW

# 
# 
# ##### tables for SGA 
# 
# # Function to convert week_day to numeric week
# convert_week_day <- function(week_day) {
#   # Split by the '+' symbol
#   parts <- strsplit(week_day, "\\+")[[1]]
#   week <- as.numeric(parts[1])
#   day <- as.numeric(parts[2])
#   
#   # Convert day to fraction of week and add to week
#   week + day / 7
# }
# 
# 
# #####SGA weight
# 
# #intergrowth data boys weight
# pdf_text_data <- pdf_text("INTERGROWTH-21st_Weight_Standards_Boys.pdf")  # Load the PDF
# lines <- str_split(pdf_text_data, "\n") # Split text into lines
# table_lines <- c(lines[[1]][14:52],lines[[2]][14:44])  # Assume the table is between line 14 and 52 and 14 to 44 in the 1st and 2nd page
# table_data <- str_split_fixed(table_lines, "\\s+", n = 9) # Adjust `n` to the number of columns
# table_df_boys <- as.data.frame(table_data, stringsAsFactors = FALSE)
# #early
# pdf_text_data <- pdf_text("INTERGROWTH-21st_Weight_Standards_Boys_preterm.pdf")  # Load the PDF
# lines <- str_split(pdf_text_data, "\n") # Split text into lines
# table_lines <- c(lines[[1]][14:52],lines[[2]][14:37])  # Assume the table is between line 14 and 52 and 14 to 44 in the 1st and 2nd page
# table_data <- str_split_fixed(table_lines, "\\s+", n = 9) # Adjust `n` to the number of columns
# table_df_boys_early <- as.data.frame(table_data, stringsAsFactors = FALSE)
# table_df_boys  <- rbind(table_df_boys_early, table_df_boys)
# 
# colnames(table_df_boys) <- c("empty", "week_day", "third", "fifth", "tenth", "fiftieth", "ninetieth", "ninetyfifth", "ninetyseventh")
# #convert week_day to numeric week
# table_df_boys$week <- sapply(table_df_boys$week_day, convert_week_day)
# 
# 
# 
# 
# 
# #intergrowth data Girls wieght
# pdf_text_data <- pdf_text("INTERGROWTH-21st_Weight_Standards_Girls.pdf")  # Load the PDF
# lines <- str_split(pdf_text_data, "\n") # Split text into lines
# table_lines <- c(lines[[1]][14:52],lines[[2]][14:44])  # Assume the table is between line 14 and 52 and 14 to 44 in the 1st and 2nd page
# table_data <- str_split_fixed(table_lines, "\\s+", n = 9) # Adjust `n` to the number of columns
# table_df_girls <- as.data.frame(table_data, stringsAsFactors = FALSE)
# 
# #early
# pdf_text_data <- pdf_text("INTERGROWTH-21st_Weight_Standards_Girls_preterm.pdf")  # Load the PDF
# lines <- str_split(pdf_text_data, "\n") # Split text into lines
# table_lines <- c(lines[[1]][14:52],lines[[2]][14:37])  # Assume the table is between line 14 and 52 and 14 to 44 in the 1st and 2nd page
# table_data <- str_split_fixed(table_lines, "\\s+", n = 9) # Adjust `n` to the number of columns
# table_df_girls_early <- as.data.frame(table_data, stringsAsFactors = FALSE)
# 
# table_df_girls  <- rbind(table_df_girls_early, table_df_girls)
# 
# colnames(table_df_girls) <- c("empty", "week_day", "third", "fifth", "tenth", "fiftieth", "ninetieth", "ninetyfifth", "ninetyseventh")
# #convert week_day to numeric week
# table_df_girls$week <- sapply(table_df_girls$week_day, convert_week_day)
# 




data$PTB = 0
for (i in 1:nrow(data)) {
  GA  <- as.numeric(data$GAdelivery[i]) 
    thresh= 
    if( GA < 37 ) {data$PTB[i] = 1}
    
}


###### 
write.csv(data, file = "Lipid_mom_PTB.csv", row.names = FALSE)





