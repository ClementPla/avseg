from fundus_vessels_toolkit import FundusData, SegToGraph
from fundus_vessels_toolkit.pipelines.avseg_to_tree import GNNAVSegToTree
from skimage.segmentation import expand_labels
import numpy as np
from fundus_vessels_toolkit.pipelines.seg_to_graph import GraphSimplifyArg
from fundus_vessels_toolkit.segment_to_graph.av_map_fixing import fix_av_map

params = SegToGraph(
    skeletonize_method="lee",
    fix_hollow=True,
    clean_branches_tips=15,
    min_terminal_branch_length=15,
    min_terminal_branch_calibre_ratio=1,
    simplify_graph_arg=GraphSimplifyArg(
        reconnect_endpoints=True,
        junctions_merge_distance=1,
        min_orphan_branches_length=3,
        max_cycles_length=3,
        simplify_topology=False,
    ),
    parse_geometry=True,
    adaptative_tangents=True,
)


def post_process(image, vessels, od, mac):
    fundusData = FundusData(fundus=image, vessels=vessels, od=od, macula=mac)
    av2tree = GNNAVSegToTree()

    a_tree, v_tree = av2tree(fundusData)

    a_skel = a_tree.geometric_data().branch_label_map(
        connect_nodes=True, interpolate=True
    )
    v_skel = v_tree.geometric_data().branch_label_map(
        connect_nodes=True, interpolate=True
    )

    reconstructed_maps = np.zeros_like(a_skel)
    reconstructed_maps[a_skel > 0] = 1
    reconstructed_maps[v_skel > 0] = 2

    reconstructed_maps = expand_labels(reconstructed_maps, distance=10)

    reconstructed_maps = reconstructed_maps * (vessels > 0)
    reconstructed_maps[od == 1] = vessels[od == 1]
    return reconstructed_maps


def post_process_multilabel(image, artery, vein, od, mac):
    vessels = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    vessels[vein > 0] = 2  # Vein
    vessels[artery > 0] = 1  # Artery

    fundusData = FundusData(
        fundus=image,
        vessels=vessels,
        od=od,
        macula=mac,
    )
    av2tree = GNNAVSegToTree()

    tree = av2tree(fundusData)
    av = fix_av_map(fundusData.av, tree)

    reconstructed_maps = np.zeros((av.shape[0], av.shape[1], 3), dtype=np.uint8)
    reconstructed_maps[av == 2, 2] = 255  # Vein
    reconstructed_maps[av == 1, 0] = 255  # Artery channel
    reconstructed_maps[av > 0, 1] = 255  # Both

    return reconstructed_maps
