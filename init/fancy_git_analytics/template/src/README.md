# Source Code

This directory contains the primary source code for building, analyzing, and connecting to 
datasources in this project. In here, there are four sub-folders:

* `data_loading`: This folder should contain scripts for reading in and connecting to data 
sources like CSV files or SQL databases
* `data_processing`: This folder should contain scripts for defining how to process or transform the raw 
data into cleaner, more useful data for analysis
* `analysis`: This folder should contain scripts for defining how to conduct the primary analysis
 of any given project, specifically building core machine learning models or other regression 
 models. 
* `reporting`: Scripts in this folder take the results of analysis and build summary tables or 
summary plots that describe what was discovered in the data.
* `utilities`: Modules in this folder are core utility and infrastructure routines to help this project 
define logging and capture errors.

