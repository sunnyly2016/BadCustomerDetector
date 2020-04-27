## Bad Customer Detector

The repo has two key folders and serveral examples: **BadCustomerDetector**, **test** and several examples.

In the folder **BadCustomerDetector**, there is main class. And in the folder **test**, there is an unit test file, written with *pytest*

And there are 4 examples as well.

- DetectionExample.ipynb, this is the basic detection given the dataset. 
- InferenceExample.ipynb, this is use the given dataset as the "training" dataset, and make predictions for new data samples
- RestfulServiceExample.ipynb, this is a restful service, can be used for **realtime streaming data** and **ready for deployment** . When running this one, first of all need to run the flask "RestfulService.py" to start the service, then the ipynb can access the service by http post
- PypiExample.ipynb, I published the library to Pypi.org which enables the package can be installed directly by using `pip install BadCustomerDetector=0.4`

There is only one class, but with 4 functions.

1. bad_customer_detector(df, log_transformation = True, elbow_finder = False, n_cluster = 3, outliers_fraction = 0.05, method = 'Voting', plot_wss = False, S = 0, curve='convex', direction='decreasing').

This is the entrance function users should use. Provide the right format of dataframe, user shouldget the bad customer labels with 5 different models applied to the data and by default, a 'Voting' is conducted for determining the bad customers. Minimal requirement for the function is providing a dataframe. Then users can choose if want to do a *log_transformation* for the input variables as well as providing the best K. This usually happens after EDA. Otherwise you can use *elbow_finder* to find the best number of the clusters, it needs a good understanding of the WSS, the shape(*curve*), the trend(*decreasing*) and sensitivity(*S*).

Finally, users can choose what models to use to find the outliers(bad custmomers), either choose from methods among 'CBLOF','FB','HBOS','IF','KNN' or 'Voting' which will determine bad customers based on the voting result of all the model outputs.


2. outlier_detector()

This is the actual outlier detection function , it takes the results of the clustering as the input.

3. calculate_WSS()

This is to calculate the within-cluster sum of square (wss) which will later be used for finding the best K if a user wants.
 
4. k_finder()

Utilized a package called KneeLocation to find the elbow from the WSS plot and the elbow is the best K.


