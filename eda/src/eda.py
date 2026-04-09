import data_loading
import eda_visualization
import pixel_stats
import color_analysis
import texture_analysis
import morphological_analysis
import dimensionality_reduction
import kl_divergence_analysis


def run_eda(batch_size=1024):
    print("Setup & Data Loading")
    data_loading.set_seed()
    df = data_loading.load_data()

    print("EDA Visualization")
    eda_visualization.plot_class_distribution(df)

    print("Pixel Stats")
    pixel_stats.plot_pixel_stats(
        df, data_loading.load_full_image, data_loading.load_center_crop, batch_size=batch_size
    )

    print("Color Analysis")
    color_analysis.plot_color_analysis(
        df, data_loading.load_full_image, data_loading.load_center_crop, batch_size=batch_size
    )

    print("Texture Analysis")
    texture_analysis.plot_texture_analysis(
        df, data_loading.load_full_image, data_loading.load_center_crop, batch_size=batch_size
    )

    print("Morphological Analysis")
    morphological_analysis.plot_morphological_analysis(
        df, data_loading.load_full_image, data_loading.load_center_crop, batch_size=batch_size
    )

    print("KL Divergence Analysis")
    kl_divergence_analysis.generate_kl_reports(
        df, data_loading.load_full_image, "Full96", batch_size=batch_size
    )
    kl_divergence_analysis.generate_kl_reports(
        df, data_loading.load_center_crop, "Crop32", batch_size=batch_size
    )

    print("Dimensionality Reduction (PCA)")
    _, _, pca_full_proj, _, y_full = dimensionality_reduction.run_pca_analysis(
        df, data_loading.load_full_image, "Full96", batch_size=batch_size
    )
    _, _, pca_crop_proj, _, y_crop = dimensionality_reduction.run_pca_analysis(
        df, data_loading.load_center_crop, "Crop32", batch_size=batch_size
    )

    print("Dimensionality Reduction (LDA)")
    dimensionality_reduction.run_lda_analysis(pca_full_proj, y_full, "Full96")
    dimensionality_reduction.run_lda_analysis(pca_crop_proj, y_crop, "Crop32")

    print("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()