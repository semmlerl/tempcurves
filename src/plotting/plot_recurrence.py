from plotnine import ggplot, aes, geom_point, geom_jitter, geom_line, geom_smooth, geom_bar, ggtitle, scale_x_discrete, geom_violin
import pickle

## loading data

with open("../../../../data/extracted/extracted_features_df.p", 'rb') as f: extracted_features_df = pickle.load(f)
    
with open("../../../data/extracted/extracted_features_grouped.p", 'rb') as f: extracted_features_grouped = pickle.load(f)

# plotting

ggplot(extracted_features_df, aes(x  = "Recurrence", y = "t20"))    + geom_jitter(alpha=0.3)+ geom_bar(stat= "summary", alpha = 0.3)
