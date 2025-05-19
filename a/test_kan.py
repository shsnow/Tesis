
from kan import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,5,1], grid=3, k=3, seed=42, device=device)


#Aqui va el dataset


# plot KAN at initialization
model(dataset['train_input']);
model.plot()


# train the model
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001);

model.plot()



model = model.prune()
model.plot()



#Continue training and replot


model.fit(dataset, opt="LBFGS", steps=50);


model = model.refine(10)

model.fit(dataset, opt="LBFGS", steps=50);


#Automatically or manually set activation functions to be symbolic


mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)


#Continue training till machine precision

model.fit(dataset, opt="LBFGS", steps=50);


#Obtain the symbolic formula

from kan.utils import ex_round

ex_round(model.symbolic_formula()[0][0],4)
