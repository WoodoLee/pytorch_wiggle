import gdal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib._color_data as mcd
import ROOT
import numpy as np
import pandas as pd
from mayavi import mlab
import time
from mpl_toolkits.mplot3d import Axes3D
from fittingFunction import *
from scipy.optimize import curve_fit
import torch
from torch import nn
import matplotlib.pyplot as plt
"""
x = torch.arange(1, 11, dtype=torch.float).unsqueeze(dim=1)
y = x / 2 + 1 + torch.randn(10).unsqueeze(dim=1) / 5
print(x, y)

data = torch.cat((x, y), dim=1)
data = pd.DataFrame(data.numpy())

data.to_csv('data/02_Linear_Regression_Model_Data.csv', header=['x', 'y'])
"""

# ---------------------------------------------------------------- #
# Load preprocessed Data                                           #
# ---------------------------------------------------------------- #
ROOT.ROOT.EnableImplicitMT()
f = ROOT.TFile.Open("g2wd.root", "read")
mc = f.Get("MC")
hit = f.Get("Hit")
dataMc, columnsMc = mc.AsMatrix(return_labels=True)
dataHit, columnsHit = hit.AsMatrix(return_labels=True)
dfMc = pd.DataFrame(data=dataMc, columns=columnsMc)
dfHit = pd.DataFrame(data=dataHit, columns=columnsHit)
binN = 1000
#bins = np.linspace(0, 30, 6000)
bins = np.linspace(0, 30, binN)
data_entries, bins_1 = np.histogram(dfMc.loc[(dfMc['verPMag'] > 200) &  (dfMc['verPMag'] < 2750), ["mcTime"] ], bins=bins)

binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

N0 = data_entries[0]
#for i in data_entries:
#    print(i)
#Constant parameters
mumass    = 105.6583715 #muon mass [MeV]
emass     = 0.510998928 #electron mass [MeV]
E_max_mrf = mumass/2.0  #maximum positron energy from muon decay in muon-rest-frame (MRF)
#Experimental parameters at J-PARC
mumom     = 300.0 #muon beam momentum [MeV]
beta      = mumom / np.sqrt( mumom**2 + mumass**2) #muon beam velocity (~ 0.9432)
gamma     = np.sqrt( mumom**2 + mumass**2 ) /mumass; #~ 3.0103
tau       = 2.1969811
gammatau  = gamma* tau#[us]
E_max_lab = gamma*0.5*mumass*(1+beta) #maximum positron energy from muon decay in lab-frame [MeV] (~ 309.031 MeV)
E_max_lab = gamma*mumass #(obsolete), used in TDR
omega_a   = 2.0*np.pi/2.111

pol = 0.5
A = 0.42
Asymmetry = pol * A
phi = 0.

#x_train = binscenters.astype(np.float32)
#y_train = data_entries.astype(np.float32)

x = binscenters
y = data_entries

# Avoid copy data, just refer

plt.xlim(0, 11);    plt.ylim(0, 8)
plt.scatter(x, y)
plt.title('wiggle hist')
# plt.show()
plt.savefig('wiggle_hist.png')


# ---------------------------------------------------------------- #
# Load Model                                                       #
# ---------------------------------------------------------------- #

model = nn.Linear(in_features=1, out_features=1, bias=True)
"""
print(model)
print(model.weight)
print(model.bias)
"""

# ---------------------------------------------------------------- #
# Set loss function and optimizer                                  #
# ---------------------------------------------------------------- #

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

print(model(x))


# ---------------------------------------------------------------- #
# Train Model                                                      #
# ---------------------------------------------------------------- #

for step in range(501):
    prediction = model(x)
    loss = criterion(input=prediction, target=y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------- #
# Display output                                                   #
# ---------------------------------------------------------------- #

def display_results(model, x, y):
    prediction = model(x)
    loss = criterion(input=prediction, target=y)
    
    plt.clf()
    plt.xlim(0, 11);    plt.ylim(0, 8)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
    plt.title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.data.item(), model.bias.data.item()))
    plt.show()
    # plt.savefig('02_Linear_Regression_Model_trained.png')

display_results(model, x, y)


# ---------------------------------------------------------------- #
# Save Model                                                       #
# ---------------------------------------------------------------- #

torch.save(obj=model, f='wiggle.pt')

# ---------------------------------------------------------------- #
# Load and Use Model                                               #
# ---------------------------------------------------------------- #

loaded_model = torch.load(f='wiggle.pt')

display_results(loaded_model, x, y)
