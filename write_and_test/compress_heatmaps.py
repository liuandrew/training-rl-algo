import pickle
import numpy as np

widths = [2, 3, 4, 8, 16, 32, 64]

for width in widths:
    data_folder = 'data/pdistal_rim_heatmap/'
    heatmap_path = data_folder + f'width{width}_checkpoint_hms'
    heatmap_half_path = data_folder + f'width{width}_checkpoint_hms_half'
    heatmaps = pickle.load(open(heatmap_path, 'rb'))

    heatmap_half = {}

    for t in heatmaps:
        heatmap_half[t] = {}
        for chk in heatmaps[t]:
            heatmap_half[t][chk] = []
            for hm in heatmaps[t][chk]:
                heatmap_half[t][chk].append(hm.astype(np.float16))
                
    pickle.dump(heatmap_half, open(heatmap_half_path, 'wb'))

