                       THIS IS A PACKAGE For creating built_in dataset to import it directly by importing it                 
         this package helps any developer to import built_in, preprocessing dataset and pass it to models to use it whether 
         multi_variate or uni_variate timeseries forcating, this dataset about Measurements of electric power consumption in
         one household with a one-minute sampling rate over a period of almost 4 years.Different electrical quantities and 
         some sub-metering values are available. 

# Installation

pip install electricpower

# electricpower_package
   * electricpower
      * __init__.py
      *   data
         householdpower.csv
   * test
      * __init__.py
      *  test.py
   * setup.py
   * tox.ini
   * DESCRIPTION.rst
   * MANIFEST.md
   * README.md

# how to use the package

```
import electricpower as pw
```
Then we

``` 
# import load_data to get the built-in, preprocessing data by this code:
pw.load_data()
```
# write functions that we will import in load_data() like:

 1- train_test_split() : take data and return train_data, test_data

```
def train_test_split(data_frame, test_size=0.3):
        """
        :param data_frame: The whole dataframe needed to split the data
        :param test_size:  setting the size of test set , initially equals 30%
        :return: two sets after splitting the data , one for training and the other for testing
        """

        train_size = 1 - test_size
        end_idx = int(data_frame.shape[0] * train_size * 100 // 100)

        train = data_frame.iloc[:end_idx, :]
        test = data_frame.iloc[end_idx:, :]

        return train, test
```


 2- scale_data() : take train_data, test_data and perform scaling on them
```
def scale_data(train, test):
    scaler = MinMaxScaler().fit(train)
    return scaler.transform(train), scaler.transform(test), scaler
```
 3- univariate_splitter() : take data and return arrays of input_feature and output_feature

```
def univariate_splitter(data_frame):
        """
        :param df:
        :return: two arrays one for features and the other for output
        """

        input_features = []
        ouput_feature = []

        len_df = data_frame.shape[0]

        for i in range(len_df):

            end_idx = i + 1

            if end_idx > len_df - 1:
                break

            input_x, output_y = data_frame[i:end_idx, 1:], data_frame[end_idx: end_idx + 1, 0]

            input_features.append(input_x)
            ouput_feature.append(output_y)

        return np.array(input_features), np.mean(np.array(ouput_feature), axis=1)
```

 4- multivariate_splitter() : take data and return arrays of input_feature and output_feature:
```
def multivariate_splitter(df, input_size=21, output_size=7):
        """
        :param df:
        :param input_size: how many samples added to each input
        :param output_size: how many values will be predicted from each output
        :return: two arrays one for features and the other for output
        """

        input_features = []
        ouput_feature = []

        len_df = df.shape[0]

        for i in range(len_df):

            end_idx = i + input_size

            if end_idx > len_df - output_size:
                break

            input_x, output_y = df[i:end_idx, 1:], df[end_idx: end_idx + output_size, 0]

            input_features.append(input_x)
            ouput_feature.append(output_y)

        return np.array(input_features), np.array(ouput_feature)
```   

**first we read data by pkg_resources**
**then we import all of these functions to load_data(), so once we import it we get data splitted,scalled and converted:**

=============

# NOTE:the __name__ variable stores the module name

```
def load_data():
    stream= pkg_resources.resource_stream(__name__, r'data\householdpower.csv')
    data_fram=pd.read_csv(stream,encoding='latin-1',parse_dates=['date_time'], index_col= 'date_time')
    data_fram['sub_metering_remaining'] = (data_fram.Global_active_power * 1000  / 60 ) - (data_fram.Sub_metering_1 + data_fram.Sub_metering_2 + data_fram.Sub_metering_3)
    data_fram = data_fram.resample('D').sum()
    data_fram = data_fram.resample('D').mean()
    X_train, X_test = train_test_split(data_frame=data_fram)
    X_train, X_test, scaler = scale_data(train=X_train, test=X_test)
    choosing=input('UNivariate or Multivariate (U or M)?')
    if choosing=='U':
        X_train, Y_train =univariate_splitter(X_train)
        X_test, Y_test = univariate_splitter(X_test)
    if choosing=="M":
        X_train, Y_train =multivariate_splitter(X_train)
        X_test, Y_test = multivariate_splitter(X_test)
    return X_train,X_test,Y_train,Y_test,f'shape of X_train,{X_train.shape}',f'shape of X_test, {X_test.shape}',f'shape of Y_train, {Y_train.shape}',f'shape of Y_test,{Y_test.shape}'
```
