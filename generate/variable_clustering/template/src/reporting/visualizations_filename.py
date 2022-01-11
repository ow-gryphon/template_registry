from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def plot_pca_top_two(frame, target_prediction):
    """

    :param frame: A pandas DataFrame.
    :param target_prediction: The name of the predicted class membership column.
    :return: The instance of the plot created.
    """

    fig, ax = plt.subplots()
    unique_clusters = frame[target_prediction].unique()

    pca_to_fit = PCA(n_components=2)

    frame_no_prediction = frame[[col for col in frame.columns
                                 if col not in ['index', target_prediction]]]
    pca_frame = pd.DataFrame(pca_to_fit.fit_transform(frame_no_prediction))
    frame_with_pc = pd.concat([pca_frame, frame], axis=1)
    frame_pc_grouped_by_cluster = frame_with_pc.groupby([target_prediction])

    for c_idx, cluster in enumerate(unique_clusters):
        c_cluster = frame_pc_grouped_by_cluster.get_group(cluster)
        ax.scatter(c_cluster.loc[:, 0], c_cluster.loc[:, 1], label='Cluster ' + str(c_idx))

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Top two principal components')

    return fig
