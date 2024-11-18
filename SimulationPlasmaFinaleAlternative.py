import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit
import matplotlib.animation as animation
import imageio.v2 as imageio
import os
import math
from tqdm import tqdm

# Initialisation des matrices
n = 200  # Taille matrice spatiale
nt = 3_000  # Taille matrice temporelle

# Constantes (en float32)
V0 = np.float32(5000)
f = np.float32(15000)
omega = np.float32(2 * np.pi * f)
facteur_etalement = np.float32(0.1)
kb = np.float32(1.38e-23)
T = np.float32(300)
m_e = np.float32(9.1e-31)
m_i = np.float32(5e-27)
e = np.float32(1.6e-19)
mu_e = np.float32(0.1)
mu_i = np.float32(1e-3)
Debye_length = np.float32(10e-6)
PermittivityParois = np.float32(4)
Permittivitecuivre = np.float32(10e6)
Permittivitetungsten = np.float32(10e6)

ConductiviteCuivre = np.float32(5.96e7)
ConductiviteTungsten = np.float32(1.79e7)
ConductiviteHelium = np.float32(1e-18)
ConductivitePyrex = np.float32(1e-12)

D_e = np.float32(0.1)     # coef diffusion electron
D_i = np.float32(0.01)    # coef diffusion ion

# Définition dt et dx
Longueur = np.float32(13e-2)  # Longueur du domaine en m (x)
C = np.float32(0.2)
delta_x = delta_y = Longueur / (n - 1)
delta_t = C * delta_x**2 / (2 * D_e)
max_u_e = np.float32(np.sqrt(kb * T / m_e))  # pour les électrons
max_u_i = np.float32(np.sqrt(kb * T / m_i))  # pour les ions

# Calcul des conditions CFL
delta_t_advection_e = C * delta_x / max_u_e
delta_t_advection_i = C * delta_x / max_u_i

# Calcul de delta_t pour les termes advectifs
delta_t_advection = min(delta_t_advection_e, delta_t_advection_i)
delta_t = min(delta_t_advection, delta_t)
CFL_diffusion = D_e * delta_t / (delta_x ** 2)
print("CFL CONDITIONS",CFL_diffusion)
tempstotal=nt*delta_t
print("Voici dt",delta_t)
print("Voici dx",delta_x)
print("TEMPSTOTAL",tempstotal)



#Longueur reac
Longueur = 13e-2  # Longueur du domaine en m (x)
Longueur_Reac = 10e-2  # Longueur du reacteur en m (x)
Largeur_Reac = 4e-3  # Largeur reac en m
Largeur = 3e-2  # Largeur du domaine en m (y)
centre = n // 2

#Positions x et y
X_electrode_HV_start = 3e-2  # Start electrode HV
X_electrode_HV_stop = 3.6e-2  # Stop electrode HV

Y_HVelectrode1_start = 0.35e-2 + Largeur / 2
Y_HVelectrode1_stop = 0.85e-2 + Largeur / 2
Y_HVelectrode2_start = -0.85e-2 + Largeur / 2
Y_HVelectrode2_stop = -0.35e-2 + Largeur / 2

Y_Parois1_start = 0.35e-2 + Largeur / 2
Y_Parois1_stop = Y_Parois1_start + 3e-3

Y_Parois2_start = -0.35e-2 + Largeur / 2
Y_Parois2_stop = Y_Parois2_start - 3e-3

X_electrode_Tungsten_start = X_electrode_HV_start - 0.2e-2
X_electrode_Tungsten_stop = X_electrode_HV_stop + 0.2e-2

Y_electrode_Tungsten_start = Largeur / 2 - 0.05e-2
Y_electrode_Tungsten_stop = Largeur / 2 + 0.05e-2

#Initialisation Variables V/Efield/sigma/rho
Voltage = np.ones((n, n, nt))
dxVoltage=np.ones((n, n, nt))
dyVoltage=np.ones((n, n, nt))

Electric_Field_x = np.ones((n, n, nt))
Electric_Field_y = np.ones((n, n, nt))
Electric_Field_total=np.ones((n,n,nt))

Permittivity = np.ones((n, n, nt))
sigma=np.ones((n,n,nt))
rho=np.ones((n, n, nt))

#Ini densité de courant
J=np.ones((n,n,nt))

#ini densités
n_e = np.ones((n, n, nt))
n_i = np.ones((n, n, nt))

#Initialisation sources
S_e = np.ones((n, n,nt))
S_i = np.ones((n, n,nt))

#Initialisation des vitesses des é- et des ions suivant x et y
u_e=np.ones((n,n,nt))
v_e=np.ones((n,n,nt))

u_i=np.ones((n,n,nt))
v_i=np.ones((n,n,nt))

Vitessetotal_electron=np.ones((n,n,nt))
Vitessetotal_ion=np.ones((n,n,nt))

#Init températures
T_matrice=np.ones((n,n,nt))

########Zones
Zone_Heliumy1=centre + int(Y_Parois2_start / delta_y)
Zone_Heliumy2=centre+int(Y_Parois1_start/delta_y)
Zone_Heliumx1=0
Zone_Heliumx2=int(Longueur_Reac / delta_x)

Zone_Parois_Haute_x1=Zone_Parois_Basse_x1=0
Zone_Parois_Haute_y1=centre + int(Y_Parois1_start / delta_y)
Zone_Parois_Haute_y2=centre + int(Y_Parois1_stop / delta_y)

Zone_Parois_Haute_x2=Zone_Parois_Basse_x2=int(Longueur_Reac / delta_x)
Zone_Parois_Basse_y1=centre + int(Y_Parois2_stop / delta_y)
Zone_Parois_Basse_y2=centre + int(Y_Parois2_start / delta_y)

Zone_Electrode_Haute_x1=Electrode_Basse_x1=int(X_electrode_HV_start / delta_x)
Zone_Electrode_Haute_x2=Electrode_Basse_x2=int(X_electrode_HV_stop / delta_x)
Zone_Electrode_Haute_y1=centre + int(Y_HVelectrode1_start / delta_y)
Zone_Electrode_Haute_y2=centre + int(Y_HVelectrode1_stop / delta_y)
Zone_Electrode_Basse_y1=centre + int(Y_HVelectrode2_start / delta_y)
Zone_Electrode_Basse_y2=centre + int(Y_HVelectrode2_stop / delta_y)

####Initialisation sigma
sigma[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2,Zone_Electrode_Haute_y1: Zone_Electrode_Haute_y2, 0] = ConductiviteCuivre
sigma[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2,Zone_Electrode_Basse_y1: Zone_Electrode_Basse_y2, 0] = ConductiviteCuivre

sigma[0: Zone_Heliumx2,Zone_Parois_Haute_y1: centre + int(Y_Parois1_stop / delta_y), 0] = ConductivitePyrex
sigma[0: Zone_Heliumx2,Zone_Parois_Basse_y1: Zone_Heliumy1, 0] = ConductivitePyrex

sigma[int(X_electrode_Tungsten_start / delta_x): int(X_electrode_Tungsten_stop / delta_x),centre + int(Y_electrode_Tungsten_start / delta_y): centre + int(Y_electrode_Tungsten_stop / delta_y),0] = ConductiviteTungsten

# Initialisation permittivite
Permittivity[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2,Zone_Electrode_Haute_y1: Zone_Electrode_Haute_y2, :] = Permittivitecuivre
Permittivity[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2,Zone_Electrode_Basse_y1: Zone_Electrode_Basse_y2, :] = Permittivitecuivre

Permittivity[0: Zone_Heliumx2,Zone_Parois_Haute_y1: centre + int(Y_Parois1_stop / delta_y), :] = PermittivityParois
Permittivity[0: Zone_Heliumx2,Zone_Parois_Basse_y1: Zone_Heliumy1, :] = PermittivityParois

Permittivity[int(X_electrode_Tungsten_start / delta_x): int(X_electrode_Tungsten_stop / delta_x),centre + int(Y_electrode_Tungsten_start / delta_y): centre + int(Y_electrode_Tungsten_stop / delta_y),:] = Permittivitetungsten

####Source intialisation
S_e[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,:]=0
S_i[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,:]=0

#Vitesse
u_e[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,0]=np.sqrt(kb*T/m_e)
v_e[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,0]=np.sqrt(kb*T/m_e)

u_i[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,0]=np.sqrt(kb*T/m_i)
v_i[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,0]=np.sqrt(kb*T/m_i)

Vitessetotal_electron[:,:,0]=np.sqrt(u_e[:,:,0]**2 +v_e[:,:,0]**2)
Vitessetotal_ion[:,:,0]=np.sqrt(u_i[:,:,0],v_i[:,:,0]**2)



#Initialisation densités
n_e[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,0]=1e15
n_i[Zone_Electrode_Haute_x1:Zone_Electrode_Haute_x2,Zone_Heliumy1:Zone_Heliumy2,0]=1e9

plt.imshow(Voltage[:, :, 0].T, cmap='plasma', origin='lower')
plt.colorbar(label='Voltage')
plt.show()

plt.imshow(n_e[:, :, 0].T, cmap='plasma', origin='lower')
plt.colorbar(label='n_e')
plt.show()

plt.imshow(n_i[:, :, 0].T, cmap='plasma', origin='lower')
plt.colorbar(label='n_i')
plt.show()


@njit()
def conditions_aux_limites(x):
    ###4 domaines
    x[:, 0, :] = x[:, 0, :]
    x[0, :, :] = 0
    x[-1, :, :] = 0
    x[:, -1, :] = 0
    #Electrode du haut = 0
    x[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Haute_y2, :] = 0
    #Parois haute avec elec gauche x
    x[0: Zone_Electrode_Haute_x1, Zone_Parois_Haute_y2, :] = 0
    #Parois haute avec elec droite x
    x[ Zone_Electrode_Haute_x2:Zone_Heliumx2, Zone_Parois_Haute_y2, :] = 0
    #Parois haute avec elec gauche y
    x[ Zone_Electrode_Haute_x1, Zone_Electrode_Haute_y1:Zone_Electrode_Haute_y2, :] = 0
    #Parois haute avec elec droite y
    x[Zone_Electrode_Haute_x2, Zone_Electrode_Haute_y1:Zone_Electrode_Haute_y2, :] = 0
    #Parois bord reac haute
    x[Zone_Heliumx2, Zone_Parois_Haute_y1:Zone_Parois_Haute_y2, :] = 0

    # Electrode du bas = 0
    x[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Basse_y2, :] = 0
    # Parois basse avec elec gauche x
    x[0: Zone_Electrode_Haute_x1, Zone_Parois_Basse_y2, :] = 0
    # Parois basse avec elec droite x
    x[Zone_Electrode_Haute_x2:Zone_Heliumx2, Zone_Parois_Basse_y2, :] = 0
    # Parois basse avec elec gauche y
    x[Zone_Electrode_Haute_x1, Zone_Electrode_Basse_y1:Zone_Electrode_Basse_y2, :] = 0
    # Parois haute avec elec droite y
    x[Zone_Electrode_Haute_x2, Zone_Electrode_Basse_y1:Zone_Electrode_Basse_y2, :] = 0
    # Parois bord reac basse
    x[Zone_Heliumx2, Zone_Parois_Basse_y1:Zone_Parois_Basse_y2, :] = 0

    #Parois internes du reac
    x[0: Zone_Heliumx2, Zone_Heliumy1, :] = 0
    x[0: Zone_Heliumx2, Zone_Heliumy2, :] = 0
    return x

@njit()
def conditions_aux_limites_nonzero(x):
    ###4 domaines
    x[:, 0, :] = x[:, -1, :]
    x[0, :, :] = x[-1, :, :]
    x[-1, :, :] =x[-2, :, :]
    x[:, -1, :] = x[:, -2, :]
    #Electrode du haut = 0
    x[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Haute_y2, :] =x[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Haute_y2+1, :]
    #Parois haute avec elec gauche x
    x[0: Zone_Electrode_Haute_x1, Zone_Parois_Haute_y2, :] = x[0: Zone_Electrode_Haute_x1, Zone_Parois_Haute_y2+1, :]
    #Parois haute avec elec droite x
    x[ Zone_Electrode_Haute_x2:Zone_Heliumx2, Zone_Parois_Haute_y2, :] = x[ Zone_Electrode_Haute_x2:Zone_Heliumx2, Zone_Parois_Haute_y2+1, :]
    #Parois haute avec elec gauche y
    x[ Zone_Electrode_Haute_x1, Zone_Electrode_Haute_y1:Zone_Electrode_Haute_y2, :] = x[ Zone_Electrode_Haute_x1-1, Zone_Electrode_Haute_y1:Zone_Electrode_Haute_y2, :]
    #Parois haute avec elec droite y
    x[Zone_Electrode_Haute_x2, Zone_Electrode_Haute_y1:Zone_Electrode_Haute_y2, :] = x[Zone_Electrode_Haute_x2+1, Zone_Electrode_Haute_y1:Zone_Electrode_Haute_y2, :]
    #Parois bord reac haute
    x[Zone_Heliumx2, Zone_Parois_Haute_y1:Zone_Parois_Haute_y2, :] = x[Zone_Heliumx2+1, Zone_Parois_Haute_y1:Zone_Parois_Haute_y2, :]

    # Electrode du bas = 0
    x[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Basse_y2, :] =x[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Basse_y2-1, :]
    # Parois basse avec elec gauche x
    x[0: Zone_Electrode_Haute_x1, Zone_Parois_Basse_y2, :] = x[0: Zone_Electrode_Haute_x1, Zone_Parois_Basse_y2-1, :]
    # Parois basse avec elec droite x
    x[Zone_Electrode_Haute_x2:Zone_Heliumx2, Zone_Parois_Basse_y2, :] = x[Zone_Electrode_Haute_x2:Zone_Heliumx2, Zone_Parois_Basse_y2-1, :]
    # Parois basse avec elec gauche y
    x[Zone_Electrode_Haute_x1, Zone_Electrode_Basse_y1:Zone_Electrode_Basse_y2, :] = x[Zone_Electrode_Haute_x1-1, Zone_Electrode_Basse_y1:Zone_Electrode_Basse_y2, :]
    # Parois haute avec elec droite y
    x[Zone_Electrode_Haute_x2, Zone_Electrode_Basse_y1:Zone_Electrode_Basse_y2, :] = x[Zone_Electrode_Haute_x2+1, Zone_Electrode_Basse_y1:Zone_Electrode_Basse_y2, :]
    # Parois bord reac basse
    x[Zone_Heliumx2, Zone_Parois_Basse_y1:Zone_Parois_Basse_y2, :] = x[Zone_Heliumx2+1, Zone_Parois_Basse_y1:Zone_Parois_Basse_y2, :]

    #Parois internes du reac
    x[0: Zone_Heliumx2, Zone_Heliumy1, :] = x[0: Zone_Heliumx2, Zone_Heliumy1+1, :]
    x[0: Zone_Heliumx2, Zone_Heliumy2, :] =x[0: Zone_Heliumx2, Zone_Heliumy2-1, :]
    return x



def rk4_step_continuity_equations(t, dt, n_e, n_i,S_e,S_i,u_e,v_e,u_i,v_i):
    """
    Fonction pour effectuer une étape RK4 (Runge-Kutta d'ordre 4) pour les équations de continuité.

    Paramètres :
    - t : Temps actuel.
    - dt : Pas de temps.
    - n_e : Densité électronique actuelle (tableau NumPy en 3D).
    - n_i : Densité ionique actuelle (tableau NumPy en 3D).
    -S_e :  Flux electron
    -S_i : Flux ion
    -v_e:  Total speed of electrons
    -v_i:  Total speed of ion

    Retourne :
    - n_e : Prochaine estimation de la densité électronique après un pas de temps dt.
    - n_i : Prochaine estimation de la densité ionique après un pas de temps dt.
    """

    n_e=conditions_aux_limites_nonzero(n_e)
    n_i=conditions_aux_limites_nonzero(n_i)
    u_e = conditions_aux_limites_nonzero(u_e)
    v_e=conditions_aux_limites_nonzero(v_e)
    u_i = conditions_aux_limites_nonzero(u_i)
    v_i=conditions_aux_limites_nonzero(v_i)
    S_e=conditions_aux_limites_nonzero(S_e)
    S_i=conditions_aux_limites_nonzero(S_i)

    # Calcul des pentes k1 pour n_e et n_i
    k1_ne = dt *((-1/(delta_x*delta_y))*((n_e[2:,1:-1,t]*u_e[2:,1:-1,t] - n_e[:-2,1:-1,t]*u_e[:-2,1:-1,t]) / (2 * delta_x) + (n_e[1:-1,2:,t]*v_e[1:-1,2:,t] - n_e[1:-1,:-2,t]*v_e[1:-1,:-2,t]) / (2 * delta_y))+S_e[1:-1,1:-1,t])
    k1_ni = dt *((-1/(delta_x*delta_y))*((n_i[2:,1:-1,t]*u_i[2:,1:-1,t] - n_i[:-2,1:-1,t]*u_i[:-2,1:-1,t]) / (2 * delta_x) + (n_i[1:-1,2:,t]*v_i[1:-1,2:,t] - n_i[1:-1,:-2,t]*v_i[1:-1,:-2,t]) / (2 * delta_y))+S_i[1:-1,1:-1,t])
    # Calcul des pentes k2 pour n_e et n_i
    k2_ne =dt *((-1/(delta_x*delta_y))*(((n_e[2:,1:-1,t]+0.5*k1_ne)*u_e[2:,1:-1,t] - (n_e[:-2,1:-1,t]+0.5*k1_ne)*u_e[:-2,1:-1,t]) / (2 * delta_x) + ((n_e[1:-1,2:,t]+0.5*k1_ne)*v_e[1:-1,2:,t] - (n_e[1:-1,:-2,t]+0.5*k1_ne)*v_e[1:-1,:-2,t]) / (2 * delta_y))+S_e[1:-1,1:-1,t])
    k2_ni =dt *((-1/(delta_x*delta_y))*(((n_i[2:,1:-1,t]+0.5*k1_ni)*u_i[2:,1:-1,t] - (n_i[:-2,1:-1,t]+0.5*k1_ni)*u_i[:-2,1:-1,t]) / (2 * delta_x) + ((n_i[1:-1,2:,t]+0.5*k1_ni)*v_i[1:-1,2:,t] - (n_i[1:-1,:-2,t]+0.5*k1_ni)*v_i[1:-1,:-2,t]) / (2 * delta_y))+S_i[1:-1,1:-1,t])

    # Calcul des pentes k3 pour n_e et n_i
    k3_ne =dt *((-1/(delta_x*delta_y))*(((n_e[2:,1:-1,t]+0.5*k2_ne)*u_e[2:,1:-1,t] - (n_e[:-2,1:-1,t]+0.5*k2_ne)*u_e[:-2,1:-1,t]) / (2 * delta_x) + ((n_e[1:-1,2:,t]+0.5*k2_ne)*v_e[1:-1,2:,t] - (n_e[1:-1,:-2,t]+0.5*k2_ne)*v_e[1:-1,:-2,t]) / (2 * delta_y))+S_e[1:-1,1:-1,t])
    k3_ni =dt *((-1/(delta_x*delta_y))*(((n_i[2:,1:-1,t]+0.5*k2_ni)*u_i[2:,1:-1,t] - (n_i[:-2,1:-1,t]+0.5*k2_ni)*u_i[:-2,1:-1,t]) / (2 * delta_x) + ((n_i[1:-1,2:,t]+0.5*k2_ni)*v_i[1:-1,2:,t] - (n_i[1:-1,:-2,t]+0.5*k2_ni)*v_i[1:-1,:-2,t]) / (2 * delta_y))+S_i[1:-1,1:-1,t])

    # Calcul
    k4_ne = dt *((-1/(delta_x*delta_y))*(((n_e[2:,1:-1,t]+0.5*k3_ne)*u_e[2:,1:-1,t] - (n_e[:-2,1:-1,t]+0.5*k3_ne)*u_e[:-2,1:-1,t]) / (2 * delta_x) + ((n_e[1:-1,2:,t]+0.5*k3_ne)*v_e[1:-1,2:,t] - (n_e[1:-1,:-2,t]+0.5*k3_ne)*v_e[1:-1,:-2,t]) / (2 * delta_y))+S_e[1:-1,1:-1,t])
    k4_ni = dt *((-1/(delta_x*delta_y))*(((n_i[2:,1:-1,t]+0.5*k3_ni)*u_i[2:,1:-1,t] - (n_i[:-2,1:-1,t]+0.5*k3_ni)*u_i[:-2,1:-1,t]) / (2 * delta_x) + ((n_i[1:-1,2:,t]+0.5*k3_ni)*v_i[1:-1,2:,t] - (n_i[1:-1,:-2,t]+0.5*k3_ni)*v_i[1:-1,:-2,t]) / (2 * delta_y))+S_i[1:-1,1:-1,t])

    # Mise à jour des densités électronique et ionique
    n_e[1:-1,1:-1,t+1] = n_e[1:-1,1:-1,t] + (1.0 / 6.0) * (k1_ne + 2 * k2_ne + 2 * k3_ne + k4_ne)
    n_i[1:-1,1:-1,t+1] = n_i[1:-1,1:-1,t] + (1.0 / 6.0) * (k1_ni + 2 * k2_ni + 2 * k3_ni + k4_ni)

    return n_e, n_i

Voltagefinal=np.ones((n,n,nt))
Electric_Field_x_final=np.ones((n,n,nt))
Electric_Field_y_final=np.ones((n,n,nt))
Electric_Field_total_final=np.ones((n,n,nt))
n_e_final=np.ones((n,n,nt))
n_i_final=np.ones((n,n,nt))
u_e_final=np.ones((n,n,nt))
v_e_final=np.ones((n,n,nt))
u_i_final=np.ones((n,n,nt))
v_i_final=np.ones((n,n,nt))
Vitessetotal_electron_final=np.ones((n,n,nt))
Vitessetotal_ion_final=np.ones((n,n,nt))
J_final=np.ones((n,n,nt))
sigma_final=np.ones((n,n,nt))

#@jit(nopython=True)
def Electric_Field(Voltage, Permittivity,sigma, Electric_Field_x, Electric_Field_y,Electric_Field_total,rho,J,n_i,n_e,u_e,u_i,v_e,v_i,Vitessetotal_electron, Vitessetotal_ion,S_e,S_i):
    for time in range(0, nt - 1):
        print("Progress", time * 100 / (nt - 1), '%')
        #####Conditions aux limites
        n_e = conditions_aux_limites(n_e)
        n_i = conditions_aux_limites(n_i)
        u_e = conditions_aux_limites(u_e)
        v_e = conditions_aux_limites(v_e)
        u_i = conditions_aux_limites(u_i)
        v_i = conditions_aux_limites(v_i)
        Vitessetotal_electron = conditions_aux_limites(Vitessetotal_electron)
        Vitessetotal_ion = conditions_aux_limites(Vitessetotal_ion)
        S_e = conditions_aux_limites(S_e)
        S_i = conditions_aux_limites(S_i)
        J = conditions_aux_limites(J)
        sigma = conditions_aux_limites(sigma)
        rho = conditions_aux_limites(rho)
        Voltage = conditions_aux_limites(Voltage)
        Electric_Field_x = conditions_aux_limites(Electric_Field_x)
        Electric_Field_y = conditions_aux_limites(Electric_Field_y)
        Electric_Field_total = conditions_aux_limites(Electric_Field_total)

        # Initialisation Voltage
        Voltage[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Haute_y1: Zone_Electrode_Haute_y2,:] = V0 * np.sin(omega *delta_t* time)
        Voltage[Zone_Electrode_Haute_x1: Zone_Electrode_Haute_x2, Zone_Electrode_Basse_y1: Zone_Electrode_Basse_y2,:] = V0 * np.sin(omega * delta_t*time)

        ##CALCUL
        rho[1:-1,1:-1,time+1]=e*(n_i[1:-1,1:-1,time]-n_e[1:-1,1:-1,time])

        epsilon_x1 = (Permittivity[1:-1, 1:-1,time] + Permittivity[2:, 1:-1,time] ) / (2)
        epsilon_x2 = (Permittivity[:-2, 1:-1,time]  +Permittivity[1:-1, 1:-1,time] ) / (2)
        epsilon_y1 = (Permittivity[1:-1, 1:-1,time]  + Permittivity[1:-1, 2:,time] ) / (2)
        epsilon_y2 = (Permittivity[1:-1, 1:-1,time]  + Permittivity[1:-1, :-2,time] ) / (2)

        a = (epsilon_x1 + epsilon_x2) / delta_x
        b = (epsilon_y1 + epsilon_y2) / delta_y
        c = rho[1:-1,1:-1,time]

        Voltage[1:-1, 1:-1,time+1] = (epsilon_x1 * Voltage[2:, 1:-1,time] + epsilon_x2 * Voltage[:-2, 1:-1,time] +epsilon_y1 * Voltage[1:-1, 2:,time] + epsilon_y2 * Voltage[1:-1, :-2,time] - c) / (a + b)

        Electric_Field_x[1:-1, 1:-1, time + 1],Electric_Field_y[1:-1, 1:-1, time + 1]  =np.gradient(-Voltage[1:-1, 1:-1, time],delta_x,delta_y)
        Electric_Field_total[1:-1, 1:-1, time + 1] = np.sqrt(Electric_Field_x[1:-1, 1:-1, time] ** 2 + Electric_Field_y[1:-1, 1:-1, time] ** 2)


        ####Transport des particules:

        u_e[1:-1,1:-1,time+1]= u_e[1:-1,1:-1,time] +delta_t*((-mu_e*Electric_Field_x[1:-1,1:-1,time])-D_e*((n_e[2:,1:-1,time]-n_e[:-2,1:-1,time])/2*delta_x))
        v_e[1:-1, 1:-1, time + 1] = v_e[1:-1,1:-1,time]+delta_t*((-mu_e*Electric_Field_y[1:-1,1:-1,time])-D_e*((n_e[1:-1,2:,time]-n_e[1:-1,:-2,time])/2*delta_y))
        Vitessetotal_electron[1:-1,1:-1,time+1]=np.sqrt(u_e[1:-1,1:-1,time]**2 + v_e[1:-1,1:-1,time]**2)

        u_i[1:-1,1:-1,time+1]= u_i[1:-1,1:-1,time ]    +delta_t*((-mu_i*Electric_Field_x[1:-1,1:-1,time])-D_i*((n_i[2:,1:-1,time]-n_i[:-2,1:-1,time])/2*delta_x))
        v_i[1:-1, 1:-1, time + 1] =v_i[1:-1,1:-1,time] +delta_t*((-mu_i*Electric_Field_y[1:-1,1:-1,time])-D_i*((n_i[1:-1,2:,time]-n_i[1:-1,:-2,time])/2*delta_y))
        Vitessetotal_ion[1:-1, 1:-1, time + 1] = np.sqrt(u_i[1:-1, 1:-1, time] ** 2 + v_i[1:-1, 1:-1, time] ** 2)

        #Sigma
        sigma[1:-1,1:-1,time+1]=sigma[1:-1,1:-1,0]*(n_e[1:-1,1:-1,time]+n_i[1:-1,1:-1,time])

        #Loi d'Ohm
        J[1:-1,1:-1,time+1]=sigma[1:-1,1:-1,time]*0.25*(Electric_Field_total[2:,1:-1,time]+Electric_Field_total[1:-1,2:,time]+Electric_Field_total[:-2,1:-1,time]+Electric_Field_total[1:-1,:-2,time])/delta_x**2

        ####Continuité ions et electrons
        #n_e,n_i=rk4_step_continuity_equations(time, delta_t, n_e, n_i,S_e,S_i,u_e,v_e,u_i,v_i)
        n_e[1:-1, 1:-1, time + 1] = n_e[1:-1, 1:-1, time] - delta_t * (
                ((u_e[2:, 1:-1, time] * n_e[2:, 1:-1, time] - u_e[:-2, 1:-1, time] * n_e[:-2, 1:-1, time]) / (
                            2 * delta_x)) +
                ((v_e[1:-1, 2:, time] * n_e[1:-1, 2:, time] - v_e[1:-1, :-2, time] * n_e[1:-1, :-2, time]) / (
                            2 * delta_y))
        ) + delta_t * S_e[1:-1, 1:-1, time]

        # Update ion denS_ity
        n_i[1:-1, 1:-1, time + 1] = n_i[1:-1, 1:-1, time] - delta_t * (
                ((u_i[2:, 1:-1, time] * n_i[2:, 1:-1, time] - u_i[:-2, 1:-1, time] * n_i[:-2, 1:-1, time]) / (
                            2 * delta_x)) +
                ((v_i[1:-1, 2:, time] * n_i[1:-1, 2:, time] - v_i[1:-1, :-2, time] * n_i[1:-1, :-2, time]) / (
                            2 * delta_y))
        ) + delta_t * S_i[1:-1, 1:-1, time]
        print(f"#######################Boucle{time}#################")

        #Sauvegarder les variables
        Voltagefinal[:,:,time]=Voltage[:,:,time]
        Electric_Field_x_final[:, :, time] =Electric_Field_x[:, :, time]
        Electric_Field_y_final[:, :, time] = Electric_Field_y[:, :, time]
        Electric_Field_x_final[:, :, time] = Electric_Field_x[:, :, time]
        Electric_Field_total_final[:, :, time] = Electric_Field_total[:, :, time]
        n_e_final[:,:,time]=n_e[:,:,time]
        n_i_final[:,:,time]=n_i[:,:,time]
        J_final[:,:,time]=J[:,:,time]
        u_e_final[:,:,time]=u_e[:,:,time]
        v_e_final[:,:,time]=v_e[:,:,time]
        u_i_final[:,:,time]=u_i[:,:,time]
        v_i_final[:,:,time]=v_i[:,:,time]
        Vitessetotal_electron_final[:,:,time]=Vitessetotal_electron[:,:,time]
        Vitessetotal_ion_final[:,:,time]=Vitessetotal_ion[:,:,time]
        sigma_final[:,:,time]=sigma[:,:,time]

        # Seuils
        threshold_n_e = 1e30
        threshold_u_e = 1e20

        # Créer des masques booléens pour les conditions
        mask_n_e = n_e > threshold_n_e
        mask_u_e = u_e > threshold_u_e

        # Afficher les valeurs supérieures aux seuils
        if np.any(mask_n_e):  # Vérifie si au moins une valeur satisfait la condition
            print("Values in n_e above threshold:")
            print(n_e[mask_n_e])

        if np.any(mask_u_e):  # Vérifie si au moins une valeur satisfait la condition
            print("Values in u_e above threshold:")
            print(u_e[mask_u_e])

    return Voltagefinal,Electric_Field_x_final,Electric_Field_y_final,Electric_Field_total_final,n_e_final,n_i_final,J_final,u_e_final,v_e_final,u_i_final,v_i_final,Vitessetotal_electron_final,Vitessetotal_ion_final,sigma_final


V,Ex,Ey,Etot,ne,ni,Jf,ue,ve,ui,vi,Vetot,Vitot,sigmaf= Electric_Field(Voltage, Permittivity,sigma, Electric_Field_x, Electric_Field_y,Electric_Field_total,rho,J,n_i,n_e,u_e,u_i,v_e,v_i,Vitessetotal_electron, Vitessetotal_ion,S_e,S_i)

@jit(nopython=True)
def generate_video(V, name,fps, duration):
    num_frames = fps * duration
    vmin=np.min(V[:,:,:])
    vmax = np.max(V[:, :, :])
    # Ensure that we do not exceed the available time steps
    num_time_steps = V.shape[2]
    if num_time_steps < num_frames:
        raise ValueError(f"Number of time steps ({num_time_steps}) is less than required frames ({num_frames}).")

    file_directory = r"E:\2emeAnnee\Physique\Simulation\Python\Figures"
    # Create a writer object using imageio
    writer = imageio.get_writer(file_directory + "\\" + name+'.mp4', fps=fps)

    # Calculate the step size for selecting frames from V
    step_size = max(num_time_steps // num_frames, 1)

    # Loop over the frames to create the video
    for frame_idx in tqdm(range(num_frames)):
        t = frame_idx * step_size
        frame = V[:, :, t]
        # Create a matplotlib figure
        plt.figure(figsize=(8, 5))
        # Display the frame using imshow
        if name=="Voltage":
            plt.imshow(frame.T, cmap='plasma',origin='lower',vmin=-5000,vmax=5000)
        plt.imshow(frame.T, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.axis('off')  # Hide the axes
        plt.title(f"Evolution of "+name+f"\n time={t*delta_t:.2e}s")
        plt.colorbar()
        # Save the figure to a temporary file
        plt.savefig(r'E:\2emeAnnee\Physique\Simulation\Python\Figures\temp\temp_frame.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Read the saved frame and append it to the video
        writer.append_data(imageio.imread('temp_frame.png'))

    # Close the writer to finalize the video file
    writer.close()


# Example usage:
if __name__ == "__main__":
    # Generate the video with 30 fps and 30 seconds duration
    generate_video(V,"Voltage",fps=30, duration=30)
    generate_video(Ex, "Ex", fps=30, duration=30)
    generate_video(Ey, "Ey", fps=30, duration=30)
    generate_video(Etot, "Etot", fps=30, duration=30)
    generate_video(ne, "ne", fps=30, duration=30)
    generate_video(ni, "ni", fps=30, duration=30)
    generate_video(Jf, "J", fps=30, duration=30)
    generate_video(ue, "ue", fps=30, duration=30)
    generate_video(ve, "ve", fps=30, duration=30)
    generate_video(ui, "ui", fps=30, duration=30)
    generate_video(vi, "vi", fps=30, duration=30)
    generate_video(Vetot, "Vetot", fps=30, duration=30)
    generate_video(Vitot, "Vitot", fps=30, duration=30)
    generate_video(sigmaf, "sigma", fps=30, duration=30)

    print("Video has been created successfully.")


