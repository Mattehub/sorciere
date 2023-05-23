

import pickle

def saving(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)
        
def loading(name):
    with open(name,"rb") as f:
        loaded_obj=pickle.load(f)
        
    return loaded_obj

