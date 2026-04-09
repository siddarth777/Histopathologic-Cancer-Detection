import argparse
import EDA.eda.src.data_loading as data_loading
import EDA.eda.src.eda_visualization as eda_visualization
import EDA.eda.src.pixel_stats as pixel_stats
import EDA.eda.src.color_analysis as color_analysis
import EDA.eda.src.texture_analysis as texture_analysis
import EDA.eda.src.morphological_analysis as morphological_analysis
import EDA.eda.src.dimensionality_reduction as dimensionality_reduction
import EDA.eda.src.kl_divergence_analysis as kl_divergence_analysis
import EDA.eda.src.edge_density_analysis as edge_density_analysis


def run_eda(batch_size=1024, sample_n=None):
    """Run the full EDA pipeline.

    Args:
        batch_size: Number of images to process per batch.
        sample_n: If set, use a stratified sample of this many images
                  instead of the full dataset.  Pass ``None`` for all data.
    """
    print("Setup & Data Loading")
    data_loading.set_seed()

    if sample_n is not None:
        _, df = data_loading.load_data_and_sample(sample_n)
        print(f"  Using stratified sample of {len(df)} images")
    else:
        df = data_loading.load_data()
        print(f"  Using full dataset ({len(df)} images)")

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

    print("Edge Density Analysis")
    edge_density_analysis.generate_edge_density_report(
        df, data_loading.load_full_image, "Full96", batch_size=batch_size
    )
    edge_density_analysis.generate_edge_density_report(
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
    parser = argparse.ArgumentParser(description="Run EDA pipeline")
    parser.add_argument(
        "--sample_n", type=int, default=None,
        help="Number of images to sample (stratified). Omit to use all data.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024,
        help="Batch size for processing (default: 1024).",
    )
    args = parser.parse_args()
    run_eda(batch_size=args.batch_size, sample_n=args.sample_n)