import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# import random # This is a test
from sklearn.tree import DecisionTreeRegressor

###############################################################################
#############          USER INPUT         #####################################

ind_cyt=[0,1,2, 3, 4, 5] # index to select cytokines
# 0 = TNF, 1=IL1, 2=GCSF, 3=KC, 4=MIP, 5=TIM
# if ind_cyt=[0, 1, 2, 3, 4, 5] then all cytokines will be selected
# ind_cyt=[0,1,2,5] best for fibrosis

ind_t=[2] # index to select time point(s)  
# [0,1,2,3,4] corresponds to day - 7, 5, 35, 80, 105
# ind_t=[2] best for fibrosis

scale_flag='mean' # Scale data according to mean=0 and std=1
# or with 'median', giving median=0 and interquartile range=1

scale_flag='mean' # Scale data according to mean=0 and std=1
# or with 'median', giving median=0 and interquartile range=1

m_d=3 # Max depth of regression tree

###############################################################################


# Read data
# os.chdir('C:\\Users\\123\\Desktop\\PROCCA\\Cytokines\\Decision tree')

# Make indices to access control and irradiated mice
ind_c   = np.linspace(0, 8, 9, dtype = int)
ind_i   = np.linspace(9, 18, 10, dtype = int)
nf      = len(ind_cyt)

# read data
fil     = 'Book1.csv'
df      = pd.read_csv(fil)
dat     = df.to_numpy()

# Both total

endp1       = dat[:, 2]
dummy       = endp1[ind_i]
ind         = np.where(dummy != '-')
mea         = np.mean(dummy[ind].astype(float))
ind         = np.argwhere(endp1 == '-')
endp1[ind]  = mea
endp1       = endp1.astype(float)
endp1       = (endp1-np.mean(endp1))/np.std(endp1)
endp2       = 0.5*(dat[:, 6]+dat[:, 7])
endp2       = -(endp2-np.mean(endp2))/np.std(endp2)
endp        = np.concatenate((endp1, endp2))

n_mice      = 19 # NB overides =len(endp)
mouse_ID    = dat[:,0]
var         = df.columns.values

# Group indices for identifying coloumn data
# Saliva TNF
sal_TNF=np.linspace(8,12,5, dtype=int) #0
# Saliva IL1
sal_IL1=np.linspace(13,17,5, dtype=int) #1
# Saliva GCSF 
sal_GCSF=np.linspace(18,22,5, dtype=int) #2
# Saliva KC 
sal_KC=np.linspace(23,27,5, dtype=int) #3
# Saliva MIP 
sal_MIP=np.linspace(28,32,5, dtype=int) #4
# Saliva TIM
sal_TIM=np.linspace(33,37,5, dtype=int) #5

# Create list of indices to select set of cytokines based on ind_cyt
index_sal_all=np.concatenate((sal_TNF, sal_IL1, sal_GCSF, sal_KC, sal_MIP, sal_TIM))
lcyt=len(ind_cyt)
index_sal=np.empty(lcyt*5, dtype=int) # Always 5 time points
j=0
for i in range(lcyt):
    index_sal[j:j+5]=index_sal_all[ind_cyt[i]*5:(ind_cyt[i]+1)*5]
    j=j+5

#Function to scale data uniformly
def scale(dat, index, ind_t, flag):
    nint=round(len(index)/5)
    nt=len(ind_t) 
    rows=dat.shape[0]  
    cols=nint*nt 
    dat_new=np.empty([rows, cols])
    j=0
    k=0
    for i in range(nint):
        ind=index[j:j+5]
        ind=ind[ind_t]
        x=dat[:,ind]
        if flag=='mean':
            xm=np.mean(x)
            sig=np.std(x)
        if flag=='median':
            xm=np.median(x)
            sig=np.percentile(x, 75)-np.percentile(x, 25)
        dat_new[:, k:k+nt]=(x-xm)/sig
        j=j+5
        k=k+nt
    return dat_new

xn=scale(dat, index_sal, ind_t, scale_flag)

regr=DecisionTreeRegressor(max_depth=m_d, random_state = 0)

ind_all=np.concatenate((ind_c, ind_i))
ind_all=np.concatenate((ind_all, ind_all+n_mice))
X=np.concatenate((xn, xn))
endp_pred=np.empty(2*n_mice)
endp_m=np.empty(2*n_mice)
impo=np.empty(nf)
for i in range(n_mice):
    ind=np.where((ind_all != i) & (ind_all != i+n_mice)) # select all mice but #i, make model
    ydum=endp[ind]
    xdum=np.squeeze(X[ind,:], 0)
    xp1=X[i,:].reshape(1,-1) # Get feature vector for mouse #i
    xp2=X[i+n_mice,:].reshape(1,-1)
    xp=np.concatenate((xp1, xp2))
    endp_m[i], endp_m[i+n_mice] = np.mean(ydum), np.mean(ydum)
    pcr=regr.fit(xdum, ydum)
    endp_pred[i],endp_pred[i+n_mice]=regr.predict(xp)
    impo = impo + regr.feature_importances_

print(impo/np.sum(impo))

n=len(endp)
RMSE=np.sqrt(np.sum((endp-endp_pred)**2)/n)
RMSE_baseline=np.sqrt(np.sum((endp-endp_m)**2)/n)

print(RMSE, RMSE_baseline)

mini=np.min(np.concatenate((endp, endp_pred)))
maxi=np.max(np.concatenate((endp, endp_pred)))

ind_c=np.concatenate((ind_c, ind_c+n_mice))
ind_i=np.concatenate((ind_i, ind_i+n_mice))

mouse_ID=np.concatenate((mouse_ID, mouse_ID))
ind_c_f=ind_c[0:8]
ind_c_s=ind_c[9:19]
ind_i_f=ind_i[0:8]
ind_i_s=ind_i[9:19]

ind_f=np.concatenate((ind_c_f, ind_i_f))
ind_s=np.concatenate((ind_c_s, ind_i_s))
nf=len(ind_f)
ns=len(ind_s)
RMSE_f=np.sqrt(np.sum((endp[ind_f]-endp_pred[ind_f])**2)/nf)
RMSE_s=np.sqrt(np.sum((endp[ind_s]-endp_pred[ind_s])**2)/ns)

print(RMSE_f, RMSE_s)

# Plot regression results

fig, ax = plt.subplots()
ax.scatter(endp[ind_c_f], endp_pred[ind_c_f], s=35, alpha=0.7, marker='o', color='tab:blue', edgecolors=None, label='Controls, fibrosis')
ax.scatter(endp[ind_c_s], endp_pred[ind_c_s], s=35, alpha=0.7, marker='v', color='tab:blue',edgecolors=None, label='Controls, saliva')
ax.scatter(endp[ind_i_f], endp_pred[ind_i_f], s=35, alpha=0.7, marker='o', color='tab:orange', edgecolors=None, label='Irradiated , fibrosis')
ax.scatter(endp[ind_i_s], endp_pred[ind_i_s], s=35, alpha=0.7, marker='v', color='tab:orange',edgecolors=None, label='Irradiated, saliva')
# for i in range(n):
#     ax.annotate(mouse_ID[i], (endp[i], endp_pred[i]), fontsize=10)
#ax.plot([mini-0.2, maxi+0.2], [mini-0.2, maxi+0.2], color='k', linestyle='dashed')

fs_su=14
ax.set_xlim(mini-0.2, maxi+0.2)
ax.set_ylim(mini-0.2, maxi+0.2)
ax.set_ylabel('Predicted score', fontsize=fs_su)
ax.set_xlabel('Measured score', fontsize=fs_su)
ax.legend(fontsize=fs_su-3)
plt.xticks(fontsize=fs_su-2)
plt.yticks(fontsize=fs_su-2)

print("Print Statement 1")
print("Print Statement 2")
print("Print Statement 3")
print("Print Statement 4")
print("Print Statement 5")
print("Print Statement 6")

plt.savefig('Predict_tree.png', dpi=600)