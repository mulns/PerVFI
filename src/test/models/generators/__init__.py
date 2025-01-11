def build_generator_arch(version):
    if version.lower() == "v00":
        from .PFlowVFI_V0 import Network

        model = Network(dilate_size=9)

    elif version.lower() == "vb":
        from .PFlowVFI_Vb import Network

        model = Network(9)


    return model
