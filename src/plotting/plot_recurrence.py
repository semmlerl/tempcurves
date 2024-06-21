from plotnine import ggplot, aes, geom_point, geom_jitter, geom_line, geom_smooth, geom_bar, ggtitle, scale_x_discrete, geom_violin
import pandas as pd
import pickle
import patchworklib as pw


## loading data

with open("../../../../data/extracted/extracted_features_df.p", 'rb') as f: extracted_features_df = pickle.load(f)
    
with open("../../../data/extracted/extracted_features_grouped.p", 'rb') as f: extracted_features_grouped = pickle.load(f)

# plotting

ggplot(extracted_features_df, aes(x  = "Recurrence", y = "average_cooling_energy"))    + geom_jitter(alpha=0.3)+ geom_bar(stat= "summary", alpha = 0.3)


## raw data graphs for all patients

## loop through all patients
for patient in data_re_fe['ID'].unique(): 
    
    
    ## filter all tempcurves for a patient
    patient_curves = list(filter(lambda data_raw: data_raw['id'] == patient, data_raw))
    
    ## select recurrence data for that patient
    recurrence = data_re_fe[data_re_fe['ID'] == patient]
    
    plots = []  ## initiales list that is going to store the plots 
    
    ## plot all tempcurves
    for curve in patient_curves: 
        
        p = ggplot(curve['data'], aes( x = "Time", y = "Temperature"))+ geom_line()+ggtitle(curve['trace_id'])
        plots.append(p)        
    
    ## adding annotation
    string = ("recurrence: " + str(recurrence['Recurrence'].values[0]) + ", Reconduction sites: " + str(recurrence['Reconduction_Site'].values[0]))
    plots[0] += ggtitle(string)    
        
    # convert all plots to pw objects
    formatted = []
    for plot in plots:        
        format = pw.load_ggplot(plot, figsize = (2,2))
        formatted.append(format)
  
     ## generate a single file with all plots 
    total = formatted[0]
    for format in formatted: 
       total += format
       
    total.savefig("../../../data/output/" + patient + ".png")    
    print(patient)
