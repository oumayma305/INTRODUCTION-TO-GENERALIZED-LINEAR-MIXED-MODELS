import numpy as np # Importe les fonctions de numpy
from numpy.random import normal as  norm # Import juste une fonction de numpy pour simplifier son nom par norm
from numpy import sqrt as  sqrt # Import juste une fonction de numpy pour simplifier son nom par sqrt
from random import Random
nb = 8525 # Nombre d'individus
Age = np.round(norm(size = nb,loc = 54,scale = 11),0) 
WBC = norm(size = nb,loc = 6000,scale = 9)# Simuations de WBC autour de 6000
Intercept=[1]*8525
RBC = norm(size = nb,loc = 4.6,scale = 2)
sex = np.random.randint(2, size=nb)
Married=np.random.randint(2, size=nb)
y=np.random.randint(2,4,8525)


import pandas as df # Importation de pandas pour la gestion des tableaux de données (dataframes)
from pandas import DataFrame as DataFrame # Importation uniquement de la fonction DataFrame pour réduire son écriture
compil = {'Intercept':Intercept,'Age':Age,'Married':Married,'sex':sex,'WBC':WBC,'RBC':RBC,'y':y}
compil = DataFrame(compil)

data=compil
import statsmodels.formula.api as smf
import statsmodels.api as sm
glm = smf.glm('y ~ sex + Age + Married+WBC+RBC',
              data=data, family=sm.families.Poisson())

res_o = glm.fit()
print(res_o.summary())


import statsmodels.formula.api as smf
from patsy import dmatrices
formula = "y ~ sex*Married*Age"
md  = smf.mixedlm(formula, compil, groups=compil["Intercept"])
mdf = md.fit()
print(mdf.summary())


import statsmodels.formula.api as smf
from patsy import dmatrices
formula = "y ~ WBC*RBC"
md  = smf.mixedlm(formula, compil, groups=compil["Intercept"])
mdf = md.fit()
print(mdf.summary())


import pandas as pd
fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
random_effects = pd.DataFrame(mdf.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'groups': 'LMM'})

Y, X   = dmatrices(formula, data=compil, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('y ~ -1+RBC+WBC+Age+sex+Married', data=compil, return_type='matrix')

X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)

%% ploting function 
import matplotlib.pyplot as plt
def plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=Z,Y=Y):
    plt.figure(figsize=(18,9))
    ax1 = plt.subplot2grid((2,2), (0, 0))
    ax2 = plt.subplot2grid((2,2), (0, 1))
    ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)
    
    fe_params.plot(ax=ax1)
    random_effects.plot(ax=ax2)
    
    ax3.plot(Y.flatten(),'o',color='k',label = 'Observed', alpha=.25)
    for iname in fe_params.columns.get_values():
        fitted = np.dot(X,fe_params[iname])+np.dot(Z,random_effects[iname]).flatten()
        print("The MSE of "+iname+ " is " + str(np.mean(np.square(Y.flatten()-fitted))))
        ax3.plot(fitted,lw=1,label = iname, alpha=.5)
    ax3.legend(loc=0)
    #plt.ylim([0,5])
    plt.show()

plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=Z,Y=Y)




