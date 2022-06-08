
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned' or opt.dataset_mode == 'unaligned_scale')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'recycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet' or opt.dataset_mode == 'unaligned_triplet_scale' or opt.dataset_mode == 'unaligned_scale')
        from .recycle_gan_model import RecycleGANModel
        model = RecycleGANModel()
    elif opt.model == 'reCycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet' or opt.dataset_mode == 'unaligned_triplet_scale' or opt.dataset_mode == 'unaligned_scale')
        from .reCycle_gan_model import ReCycleGANModel
        model = ReCycleGANModel()
    elif opt.model == 'unsup_single':
        #assert(opt.dataset_mode == 'unaligned_scale')
        from .unsup_model_single import UnsupModel
        model = UnsupModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

def create_model_RED(opt):
    model = None
    from .RED_model import REDModel
    model = REDModel(opt)
    print("model [%s] was created" % (model.name()))
    return model

def create_model_Shift(opt):
    model = None
    from .Shift_model import ShiftModel
    model = ShiftModel(opt)
    print("model [%s] was created" % (model.name()))
    return model
