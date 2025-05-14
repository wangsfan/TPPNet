from models.ttpnet import TTPNet

def get_model(model_name, class_name, args):
    name = model_name.lower()
    if name=="TTPNet":
        return TTPNet(class_name, args)
    else:
        assert 0
