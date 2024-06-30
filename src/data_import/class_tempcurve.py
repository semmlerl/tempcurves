import numpy as np
import pandas as pd 
from plotnine import ggplot, aes, geom_line, ggsave, geom_point
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt


class class_tempcurve: 
        
    # initiates a new instance of a tempcurve with the resprective raw.data 
    def __init__(self, temp_curve_df, patient_id, patient_vein_count): 
        
        self.raw_data = temp_curve_df    
        self.patient_id = patient_id
        self.trace_id = patient_id + str(patient_vein_count)
        
        self.cutted_trace = []
        self.features = {}
    
    
    # find the dipping point of a temp curve and saves a cutted_trace, that starts with the dipping point 
    def find_dipping_point(self):
        
        point_20 =self.raw_data[self.raw_data.Temperature <= 20].index[0]  
        
        dipping_point = 0
        
        for i in range(point_20,0, -1): 
            if self.raw_data['Temperature'].iloc[i]>= self.raw_data['Temperature'].iloc[i - 1] and self.raw_data['Temperature'].iloc[i]>31:
                dipping_point = i
                break
            
        self.dipping_point = dipping_point
        self.cutted_trace = self.raw_data.iloc[dipping_point:].copy()  
        self.cutted_trace.iloc[:,0]= range(1,len(self.cutted_trace['Time'])+1)
        
        
    def extract_features(self):
        
        self.features['mean_temp'] = self.cutted_trace['Temperature'].mean()
        self.features['min_temp'] = self.cutted_trace['Temperature'].min()
        self.features['min_temp_t'] = np.argmin(self.cutted_trace['Temperature'])
        self.features['max_temp'] = self.cutted_trace['Temperature'].max()
        self.features['std_temp'] = self.cutted_trace['Temperature'].std()
        self.features['length'] = self.cutted_trace['Temperature'].shape[0]
        
        # kinetics of decline
        try: 
            self.features['t30']=self.cutted_trace[self.cutted_trace.Temperature <= 30].iloc[0,0]
            self.features['t20']=self.cutted_trace[self.cutted_trace.Temperature <= 20].iloc[0,0]
            self.features['t10']=self.cutted_trace[self.cutted_trace.Temperature <= 10].iloc[0,0]
            self.features['t0']=self.cutted_trace[self.cutted_trace.Temperature <= 0].iloc[0,0]
            self.features['t_10']=self.cutted_trace[self.cutted_trace.Temperature <= -10].iloc[0,0]
            self.features['t_20']=self.cutted_trace[self.cutted_trace.Temperature <= -20].iloc[0,0]
            self.features['t_30']=self.cutted_trace[self.cutted_trace.Temperature <= -30].iloc[0,0]
            self.features['t_40']=self.cutted_trace[self.cutted_trace.Temperature <= -40].iloc[0,0]
            
        except: 
            pass
        
        # Slope of temperature decrease starting with timepoint 0
        try:
                initial_slope = np.polyfit(self.cutted_trace['Time'], self.cutted_trace['Temperature'], 1)[0]
        except np.linalg.LinAlgError:
                initial_slope = np.nan  # Handle cases with fitting issues
        
        self.features['initial_slope'] = initial_slope
        
        # Cooling energy (integral below the maximum temperature)
        self.features['cooling_energy'] = sum(x - self.features['max_temp'] for x in self.cutted_trace['Temperature'])
        self.features['average_cooling_energy'] = self.features['cooling_energy']/ self.features['length']
        
        ## calculate the exponential decay fit  
        ## using bounds to set the starting point of the fit to the temperature at time 0 and the limit to the min temperature, so only the tau is actually fitted 
        params, cv = scipy.optimize.curve_fit(exp_decay, 
                                              self.cutted_trace['Time'].iloc[0:self.features['min_temp_t']], 
                                              self.cutted_trace['Temperature'].iloc[0:self.features['min_temp_t']],
                                              bounds = (
                                                  [((self.cutted_trace['Temperature'].iloc[0]-1) + abs((self.features['min_temp']-1))), -100, (self.features['min_temp']-1)], 
                                                  [(self.cutted_trace['Temperature'].iloc[0] + abs((self.features['min_temp']-1))), 100, self.features['min_temp']]
                                                    )
                                              )
        self.features['time_constant'] = params[1]
        self.features['tau'] = 1/params[1]
        
        """ plotting the decay fit 
        a, tau, min_temp = params 
        plt.plot( self.cutted_trace['Time'].iloc[0:self.features['min_temp_t']], self.cutted_trace['Temperature'].iloc[0:self.features['min_temp_t']], '.', label="data")
        plt.plot(self.cutted_trace['Time'].iloc[0:self.features['min_temp_t']], exp_decay(self.cutted_trace['Time'].iloc[0:self.features['min_temp_t']], a, tau, min_temp), '--', label="fitted")
        plt.title("Fitted Exponential Curve" + str(params))
        plt.show()
        """  
        ## calculating a moving average of the trace
        self.cutted_trace['Smooth'] = self.cutted_trace['Temperature'].rolling(5, center = True, min_periods = 1).mean()       
        
    def plot_dipping_point(self, outpath): 
        
        g = (
            ggplot(self.raw_data, aes( x = 'Time', y = 'Temperature'))
            + geom_line()
            + geom_point(aes (x = self.dipping_point, y = self.raw_data['Temperature'].iloc[self.dipping_point]), color = "Red")
        )
        
        ggsave(g, filename = (outpath + self.trace_id + ".png"))
        print(outpath + self.trace_id + ".png saved")  
        
    def plot_cutted_trace(self, ax): 
        
        """
        plots the cutted_trace on the subplot axes of the calling plot
        
        Arguments: 
            - ax : an axes class object from mathplotlib
            
        """       
        # transposes the data into long format with a coloumn Time, a coloumn value with the temps and variable with the respective group(Temperature, Smooth)
        data_long_format = pd.melt(self.cutted_trace, ['Time'])      
        
        ## generate the lineplot 
        p = sns.lineplot(
            data=data_long_format, 
            x="Time", y="value", 
            hue = "variable", 
            legend = False, 
            ax = ax            
        )        
        
        p.set(title = self.trace_id, xlabel = None, ylabel = None)
                                  
        return p
    
def exp_decay(x, a, tau, min_temp):
    ## returns an exponential decay function to model the temp decay 
    
    return a*np.exp(-tau * x) + min_temp
        