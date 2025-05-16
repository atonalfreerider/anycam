from anycam.models.anycam import AnyCam


def make_depth_predictor(conf, **kwargs):
    from anycam.models.depth_predictor_wrapper import NPZDepthWrapper
    predictor = NPZDepthWrapper.from_conf(conf, **kwargs)
    return predictor

def make_pose_predictor(conf, **kwargs):
    enc_type = conf["type"]
    if enc_type == "anycam":
        predictor = AnyCam(conf)
    else:
        raise NotImplementedError(f"Unsupported pose predictor type: {enc_type}")
    return predictor


def make_depth_aligner(conf, **kwargs):
    da_type = conf["type"]
    if da_type == "identity":
        aligner = None
    else:
        raise NotImplementedError(f"Unsupported depth aligner type: {da_type}")
    return aligner
