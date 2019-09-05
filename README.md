# EphysAnalysis

This is an interactive code for analyzing, plotting, performing statsitical tests on electrophysiological recordings acquired with Igor. From single-cell to averages between cell classes, there are designated Jupiter notebooks (JN) and modules for each stimulus condition. To start, there's a simple pipeline: 

1. Check basic cell properties: run the JN "CellProp" to analyze access resistance, membrane baseline and IF curve. 
   (associate module: "util0.py" )
2. Analyze each stimulus condition for each cell recording: run the JN "AnalysisObject". 
   (written in OOP. Associated module: "utilObj0.py")
   a. Plot the raw trace, then align stimulus epochs with recording photodiode signal. 
   b. Save the epoch-aligned data & the epoch-averaged data in Pandas DataFrames (".pkl" files). 
   c. Plot and save the plots of each epoch and of the average of each epoch type. 
   d. In the end, there's the option to plot Flowfield maps. 
3. Reiterate Step2 until all cells recorded are fully analyzed. 
4. Average across all recordings for each stimulus. I split this part into several notebooks, organized based on stimulus type. 
   a. Edges & Flahes stimulus: run the JN "AllCells_EF". (associate module: "utils_EF.py")
   b. 12 direction gratings: "AllCells_LR". (associate module: utils_LR_PvN_PDND.py)
   c. Temporal frequency: "AllCellsPvnPDND". (associate module: utils_LR_PvN_PDND.py)
   d. Flowfield maps: "AllCellsMaps". (associate module: utils_map.py)
   
