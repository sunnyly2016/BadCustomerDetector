This is an updated version of BadCustomerDetector.

The repo has two key folders: The code, the test and an example.

In the folder BadCustomerDetector, there is the main class.

And in the folder test, there is an unit test file, written with pytest

And there is an quick example as well.

There is only one class, but with 4 functions.

1. bad_customer_detector(df, log_transformation = True, elbow_finder = False, n_cluster = 3, outliers_fraction = 0.05, method = 'Voting', plot_wss = False, S = 0, curve='convex', direction='decreasing')

This is the main one a user should use. Provide the right format of dataframe, user should be able to get the bad customer labels with 5 different models applying to the data and by default, a 'Voting' is conducted for determining the bad customers.

Minimal requirement for the function is providing a dataframe. 
Then you can choose if the do a log_transformation for the variables as well as providing the best K. This usually happens after EDA. Otherwise you can use elbow_finder to find the best number of the clusters, it needs a good understanding of the WSS, the shape(curve), the trend(decreasing) and sensitivity(S)

Finally, user can choose what models to use to find the outliers(bad custmomers), either choose from methods among 'CBLOF','FB','HBOS','IF','KNN' or 'Voting' which will determine bad customers based on the voting result of all the model outputs


2. outlier_detector

This is the actual outlier detection function , it takes the results of the clustering

3. calculate_WSS

This is to calculate the within-cluster sum of square (wss) which will later be used for finding the best K if a user want
 
4. k_finder

Utilized a package called KneeLocation to find the elbow of the WSS plot


To better understand the parameter:

df, log_transformation = True, elbow_finder = False, n_cluster = 3, outliers_fraction = 0.05, method = 'Voting', plot_wss = False, S = 0, curve='convex', direction='decreasing'

