from Training import Training_PINN
import pickle


with open("waveforms.pickle", "rb+") as f:
    inputs, states = pickle.load(f)

Training_PINN(inputs,states)