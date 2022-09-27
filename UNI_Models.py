from numpy import array
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Bidirectional,TimeDistributed,ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


class UNI_Models_Singleton: 
    
    __instance = None
   
    @staticmethod 
    def get_instance():
        if UNI_Models_Singleton.__instance==None:
            UNI_Models_Singleton([],0)
        return UNI_Models_Singleton.__instance
    def __init__(self,pred,epoch):
        
        if UNI_Models_Singleton.__instance != None:
            raise Exception("Model cannot be instantiated more than once!")
        else:
            self.pred=pred
            self.epochs=epoch
            UNI_Models_Singleton.__instance = self
            
     # using for univariate
    
    def MLP_Model(self ,n_steps,X,y):
            model = Sequential()
            model.add(Dense(100, activation='relu', input_dim=n_steps))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            # fit model
            model.fit(X, y, epochs=self.epochs, verbose=0)
            # demonstrate prediction
            x_input = array(self.pred)
            x_input = x_input.reshape((1,n_steps))
            yhat = model.predict(x_input, verbose=0)
            print(yhat)
            
    def CNN_Model(self,n_steps,n_features,X,y):  
          
         model = Sequential()
         model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
         model.add(MaxPooling1D(pool_size=2))
         model.add(Flatten())
         model.add(Dense(50, activation='relu'))
         model.add(Dense(1))
         model.compile(optimizer='adam', loss='mse')        
         # fit model
         model.fit(X, y, epochs=self.epochs, verbose=0)
         # demonstrate prediction
         x_input = array(self.pred)
         x_input = x_input.reshape((1, n_steps, n_features))
         yhat = model.predict(x_input, verbose=0)
         print(yhat)
         
    def Vanilla_LSTM(self,n_steps,n_features,X,y):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=0)
        # demonstrate prediction
        x_input = array(self.pred)
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat)
        
    def Stacked_LSTM(self,n_steps,n_features,X,y):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=0)
        # demonstrate prediction
        x_input = array(self.pred)
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat)
        
    def Bidirectional_LSTM(self,n_steps,n_features,X,y):
           
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=0)
        # demonstrate prediction
        x_input = array(self.pred)
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat)
      
    def CNN_LSTM(self,n_seq,n_steps,n_features,X,y):

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=0)
        # demonstrate prediction
        x_input = array(self.pred)
        x_input = x_input.reshape((1, n_seq, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat)
        
    def ConvLSTM(self,n_seq,n_steps,n_features,X,y):
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=self.epochs, verbose=0)
        # demonstrate prediction
        x_input = array(self.pred)
        x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat)
    