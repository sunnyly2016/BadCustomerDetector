from flask import Flask, request
import json
from BadCustomerDetector.BadCustomerDetection import BadCustomerDetection as bcd
import pandas as pd
import numpy as np
app = Flask(__name__)


@app.route('/detector',methods=['POST'])
def detect():
    dict_input = request.form['features']
    dict_input = json.loads(dict_input)
    
    data = {'unique_offer_clicked':[dict_input['unique_offer_clicked']], 
        'total_offer_clicks':[dict_input['total_offer_clicks']],
        'unique_offer_rides':[dict_input['unique_offer_rides']], 
        'total_offer_rides':[dict_input['total_offer_rides']],
        'total_offers_claimed':[dict_input['total_offers_claimed']]
       } 
 
    new_sample = pd.DataFrame(data)
    print(new_sample)
        
    new_sample_log = np.log(new_sample)
    new_sample_scaled = scaler.transform(new_sample_log)
   
    cluster_prediction = kmeans_sel.predict(new_sample_scaled)[0]
    
    outlier_detection = detectors_list[cluster_prediction]
    outlier_labels = outlier_detection.predict(new_sample)[0]
    
    cluster_outlier = {}
    cluster_outlier['cluster'] = str(cluster_prediction)
    cluster_outlier['outlier_labels'] = str(outlier_labels)
    
    return json.dumps(cluster_outlier)
    
if __name__ == '__main__':
    df = pd.read_csv('./test/unique_consumers.csv')
    detector = bcd()
    print('Model is training')
    kmeans_sel, scaler, detectors_list, outliers = detector.bad_customer_detector(df, method = 'KNN')
    print('Model is ready')
    app.run(debug = True, host = '0.0.0.0', port = 3838)