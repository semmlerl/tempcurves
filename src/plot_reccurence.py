from plotnine import ggplot, aes, geom_point, geom_line, geom_smooth, geom_bar
import pandas as pd


## loading data

re_table = pd.read_excel('../../data/recurrence.xlsx')

with open("../../data/extracted/recurrence_data_with_features.p", 'rb') as f: data_re_fe = pickle.load(f)
    
with open("../../data/extracted/extracted_features_list.p", 'rb') as f: data_fe = pickle.load(f)

with open("../../data/extracted/raw_data_list.p", 'rb') as f: data_raw = pickle.load(f)


# plotting

ggplot(data) + aes( x = "Reconduction_Site", y = "LA_FLAECHE") + geom_point() + geom_smooth(method= "lm")

 
ggplot(data_re_fe, aes(x  = "Reconduction_Site", y = "min_temp" ))    + geom_point()+ geom_bar(stat= "summary", alpha = 0.3)
