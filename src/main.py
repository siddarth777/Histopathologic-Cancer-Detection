import warnings
warnings.filterwarnings("ignore")

import data_loading
import eda_visualization
import pixel_stats
import color_analysis
import texture_analysis
import morphological_analysis
import dimensionality_reduction
import feature_engineering
import classification_baseline
import reporting

def main():
    print("Setup & Data Loading")
    data_loading.set_seed()
    df, df_sample = data_loading.load_data_and_sample()
    
    print("EDA Visualization")
    eda_visualization.plot_class_distribution(df)
    eda_visualization.plot_sample_images(df_sample, data_loading.load_full_image, data_loading.load_center_crop)
    
    print("Pixel Stats")
    pixel_stats.plot_pixel_stats(df_sample, data_loading.load_full_image, data_loading.load_center_crop)
    
    print("Color Analysis")
    color_analysis.plot_color_analysis(df_sample, data_loading.load_full_image, data_loading.load_center_crop)
    
    print("Texture Analysis")
    texture_analysis.plot_texture_analysis(df_sample, data_loading.load_full_image, data_loading.load_center_crop)
    
    print("Morphological Analysis")
    morphological_analysis.plot_morphological_analysis(df_sample, data_loading.load_full_image, data_loading.load_center_crop)
    
    print("Dimensionality Reduction (PCA)")
    pca_full, sc_full, Xp_f, Xs_f, y_f = dimensionality_reduction.run_pca_analysis(df_sample, data_loading.load_full_image, "Full96")
    pca_crop, sc_crop, Xp_c, Xs_c, y_c = dimensionality_reduction.run_pca_analysis(df_sample, data_loading.load_center_crop, "Crop32")
    
    print("Dimensionality Reduction (LDA)")
    lda_f, Xl_f, sep_full = dimensionality_reduction.run_lda_analysis(Xp_f, y_f, "Full96")
    lda_c, Xl_c, sep_crop = dimensionality_reduction.run_lda_analysis(Xp_c, y_c, "Crop32")
    
    print("Feature Engineering")
    X_tr_f, X_val_f, y_tr_f, y_val_f = feature_engineering.engineer_features(
        df_sample, data_loading.load_full_image, pca_full, sc_full, "Full96")
    
    X_tr_c, X_val_c, y_tr_c, y_val_c = feature_engineering.engineer_features(
        df_sample, data_loading.load_center_crop, pca_crop, sc_crop, "Crop32")
    
    print("Classification Baseline")
    res_full = classification_baseline.evaluate_classifiers(X_tr_f, X_val_f, y_tr_f, y_val_f, "Full96")
    res_crop = classification_baseline.evaluate_classifiers(X_tr_c, X_val_c, y_tr_c, y_val_c, "Crop32")
    
    print("Final Reporting")
    reporting.generate_comparative_report(
        res_full, res_crop, sep_full, sep_crop,
        total_len=len(df), pos_cnt=df["label"].sum(), neg_cnt=(df["label"]==0).sum()
    )
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
