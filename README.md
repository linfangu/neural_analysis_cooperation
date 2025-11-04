# Statistical machine learning models of large scale neuron population activity \

### 1. Particial-least-squares regression unique variance explained 
   Build a particial-least-squares regression model to predict population neural activity with large-scale behavioral data. Calculate the unique variance in the neural space that is explained by selected behavioral variables. 
   Follow unique_variance.m, briefly: calulate the total variance that can be explained by all behaviors. Then shuffle the target variable, leaving other behaviors intact, and calculate variance explained by this new matrix. The reduction of variance explained after shuffling the target behavior variable is the unique variance associated with this behavior. 
### 2. Receiver Operating Characteristic (ROC)
   Identify neurons that are significantly tuned to behaviors/stimuli.
   Use neural activity to classify binary behaviors, compare classifier's performance across different thresholds by plotting the true positive rate against the false positive rate. Use area under the ROC curve as index for classification accuracy. Compare observed to 1000 shuffles to compute the p-value. 
### 4. Support vector machine decoder
   Decode behavioral and cognitive events using population activity by maximizing the margin (difference between classes) of the decision hyperplane. 
