from omar_pfem.model import Transolver_Irregular_Mesh


def get_model(args):
    model_dict = {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh,
    }
    return model_dict[args.model]
