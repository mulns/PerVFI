def build_generator_arch(version):
    if version.lower() == "v00":
        from .PFlowVFI_V0 import Network

        model = Network(dilate_size=9)

    ################## ABLATION ##################
    elif version.lower() == "ab-b-n":
        from .PFlowVFI_ablation import Network

        model = Network(dilate_size=7, mask_type="binary", noise=True)
    elif version.lower() == "ab-b-nf":
        from .PFlowVFI_ablation import Network

        model = Network(dilate_size=7, mask_type="binary", noise=False)
    elif version.lower() == "ab-qb-nf":
        from .PFlowVFI_ablation import Network

        model = Network(dilate_size=7, mask_type="quasi-binary", noise=False)
    elif version.lower() == "ab-a":
        from .PFlowVFI_adaptive import Network

        model = Network(dilate_size=7)
    ################## ABLATION ##################

    elif version.lower() == "vb":
        from .PFlowVFI_Vb import Network

        model = Network(9)

    elif version.lower() == "v10":
        from .PFlowVFI_V1 import Network_flow

        model = Network_flow(5)
    elif version.lower() == "v1b":
        from .PFlowVFI_V1 import Network_base

        model = Network_base(5)
    elif version.lower() == "v20":
        from .PFlowVFI_V2 import Network_flow

        model = Network_flow(5)
    elif version.lower() == "v2b":
        from .PFlowVFI_V2 import Network_base

        model = Network_base(5)

    return model
