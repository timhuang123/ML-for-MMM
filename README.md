# ML-for-MMM

All the gas data in ‘Table S5-1 Mar 2022.xlsx’, ‘CO2-N2 - 1.xlsx’and ‘Mix gas data-GJ 27 July.xlsx’ are extracted from published work.

The excel sheet named ‘Table S5-1 Mar 2022.xlsx’ contains the membranes’ physical properties component information, experimental conditions, and relative permeability and selectivity of MOFs based MMMs. The excel sheet named ‘Book3-3.xlsx’ contains data points in each row. Based on that, two individual random forest models are built to learn the relationship between input variables and relative permeability and relative selectivity, respectively. Experimental and predicted CO2-CH4 separation of Cu-CAT-1 based MMMs also have been listed in this table. 

The excel sheet named ‘Mix gas data-GJ 27 July.xlsx’ included gas permeation data tested using CO2/CH4 or CO2/N2 mixed gas as feeding gas. 

Some data exploration work: in the script of ‘correlation_matrix.py’, the code of producing Pearson correlation matrix among input and output variables has been prepared. While the script of “permutation feature importance.py” is used to compute feature importance. The script of “partial dependence plot.py” can provide numerical value which show marginal effect of each feature in its whole range on the output variable.

The excel sheet named ‘CO2-N2 - 1.xlsx’ contains literature data of MOF-based mixed matrix membranes for CO2/N2 separation. The script of ‘RF_transfer learning_implementation.py’ shows the implementation of random forest model as well as transfer learning using random forest.

The excel sheet named ‘MOFs matches with Polymers.xlsx’ shows each MOF cooperated with polymers in the MMMs for CO2/CH4 separation, which is summarized based on Table S5.
