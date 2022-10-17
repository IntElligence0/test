# Energy Models Package

                             THIS IS A PACKAGE OF MODELS OF PREDICT IN TIMESERIES FORECASTING                
             this package helps any developer in univariate and multivariate-multi-step time series forcasting in house-power-consumption dataset lets take a look about each type 
             Real-world time series forecasting is challenging for a whole host of reasons not limited to problem features such as having multiple input variables,the requirement 
             to predict multiple time steps,nd the need to a perform the same type of prediction for multiple physical sites.

# Installation

````
pip install Power_models
````

# Models list
  
  * LSTM 
  * BILSTM
  * GRU
  * BIGRU
  * TimeDistributer
  * CNN
  * TCN
  All models take 3 parameters except TCN :
    
    * must take value 
      -1 : n_steps
      -2 : n_features 
    * default value = 1 
      -3 : n_outputs  
      
  TCN Model you can build it by just givy it data because all parameters have default vaulues
  you can else change any value you need .
  
  TCN build parameters and its default values :
  
  * batch_size = 100 
  * epochs = 200
  * verbose = 1
  * tcn1_units = 128
  * tcn2_units = 64
  * tcn1_kernel_size = 5
  * tcn2_kernel_size = 1 
  * activation = "relu"
  * return_sequences = True
  * dropout = 0.2
  
# Package Folders 
 
 * Data
 * models
 
# how to use the package

 first you must read the data set you want to use the models on it 
 and then import preprocess_data from Data folder :
 
 ````
 from Data import preprocess_data 
 ````
 
 now you can build model by import it from models folder :
 
  ````
  from models import Models as m
  ````
 Now you can use m to import any model on the models list 
 After that you will able to predict and evaluate your models used.
 
 Models have evaluation function
 
  ````
  model.predict(X)
  ````
 
 Models have evaluation function you can give it  the model and (actual , predicted) values 
  ````
  m.evaluate(model,actual,pred)
  ````
 
 Else you can calculate loss using metrics function for train and test both :
 
  ````
  m.print_metrics(model,Y_train,Y_pred_train,Y_test,Y_pred_test)
  ```` 