from plotnine import ggplot, aes, geom_point, geom_jitter, geom_line, geom_smooth, geom_bar, ggtitle, scale_x_discrete, geom_violin, geom_histogram
import pickle

## loading data

with open("../../../../data/extracted/extracted_features_df.p", 'rb') as f: extracted_features_df = pickle.load(f)
    
with open("../../../data/extracted/extracted_features_grouped.p", 'rb') as f: extracted_features_grouped = pickle.load(f)

# plotting

ggplot(extracted_features_df, aes(x  = "Recurrence", y = "min_temp_t"))    + geom_jitter(alpha=0.3)+ geom_bar(stat= "summary", alpha = 0.3)

ggplot(extracted_features_df) + geom_histogram(aes( x = "length"))
