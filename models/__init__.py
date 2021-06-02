from ResNet import resnet50, resnet152
from SED import SED

def get_model(model_name):
    try:
        return eval(model_name)()
    except:
        raise NameError(f'{model_name} is not defined')