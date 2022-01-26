from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
##########################Constantes
cp = 1200
D = 15 * 10 **(-6)
rho = rho0 = 1.1614
L = 2 * 10 ** -3
Lslot = 0.5 * 10 ** -3
Lcoflow = 0.5 * 10 ** -3

Fo = 0.2
CFL = 0.4

n = 100
nt =2000
iterations = 15_000

# Pas d'espace
delta_x = L / (n - 1)
delta_y = delta_x

# Pas de temps
F = (Fo * delta_x ** 2) / D
C = CFL * delta_x
delta_t = min(F, C) #5.4*10**-6
delta_t= 5.441621603237766e-06

x=np.zeros((n,n))
y=np.zeros((n,n))

for i in range(0, n):
    for j in range(0, n):
        x[i, j] = i * delta_x
        y[i, j] = j * delta_y

# Initialisation des variables
Phi = np.zeros((n, n, nt))  # Phi=u
Phiv = np.zeros((n, n, nt))  # Phiv=v
Phifinal=np.zeros((n,n,nt))
Phivfinal = np.zeros((n, n,nt))
Temp=np.ones((n, n, nt))
Y = np.zeros((n, n, nt,5))
div_Phi = np.zeros((n, n))
P = np.zeros((n, n))
Pkplus1 = np.zeros((n, n))
Pold = np.zeros((n, n))
b = np.zeros((n, n))


##################################################Vitesse

@jit(nopython=True)
def vitesse(Phi,Phiv,Phifinal,Phivfinal,div_Phi,P,delta_x,delta_y,rho0,delta_t,Lslot,Lcoflow,Pkplus1):
    for k in range(0, nt - 1):  # boucle temporelle
        if k%10==0:
            print("Vitesse",(k / nt) * 100,"%")

        # Top et bottom wall
        Phiv[  int((Lslot + Lcoflow) / delta_x):,0,k] = 0
        Phiv[ int((Lslot + Lcoflow) / delta_x):,-1,k] = 0
        Phi[  int((Lslot + Lcoflow) / delta_x):,0,k] = 0
        Phi[  int((Lslot + Lcoflow) / delta_x):,-1,k] = 0

        # Lslot
        Phiv[ :int(Lslot / delta_x),0,k] = 1
        Phiv[ :int(Lslot / delta_x),-1,k] = -1
        Phi[ :int(Lslot / delta_x),0,k] = 0
        Phi[  :int(Lslot / delta_x),-1,k] = 0

        # Lcoflow
        Phiv[  int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x),0,k] = 0.2
        Phiv[ int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x),-1,k] = -0.2
        Phi[ int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x),0,k] = 0
        Phi[  int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x),-1,k] = 0

        # Mur gauche
        Phi[ 0,:,k] = 0
        Phiv[0,:,k]=Phiv[1,:,k]
        # Free outlet
        Phi[ -1, :,k] = Phi[-2, :,k]
        Phiv[ -1, :,k] = Phi[-2, :,k]



        #Lax Wendroff
        # Calcul vitesse sans pression
        Phi[1:-1, 1:-1, k + 1] = Phi[1:-1, 1:-1, k ] + D*delta_t*( ((Phi[2:,1:-1,k]-2*Phi[1:-1,1:-1,k]+Phi[:-2,1:-1,k])/(delta_x**2))+((Phi[1:-1,2:,k]-2*Phi[1:-1,1:-1,k]+Phi[1:-1,:-2,k])/(delta_y**2)))\
                    - Phi[1:-1,1:-1,k]*delta_t*((Phi[2:,1:-1,k]-Phi[:-2,1:-1,k])/(2*delta_x))+((((Phi[1:-1,1:-1,k]**2)*(delta_t**2))/(2*delta_x**2))*(Phi[2:,1:-1,k]-2*Phi[1:-1,1:-1,k]+Phi[:-2,1:-1,k]))\
                    - Phiv[1:-1,1:-1,k]*delta_t*((Phi[1:-1,2:,k]-Phi[1:-1,:-2,k])/(2*delta_y))+((((Phiv[1:-1,1:-1,k]**2)*(delta_t**2))/(2*delta_y**2))*(Phi[1:-1,2:,k]-2*Phi[1:-1,1:-1,k]+Phi[1:-1,:-2,k]))
        Phiv[1:-1, 1:-1, k + 1] = Phiv[1:-1, 1:-1, k] + + D*delta_t*( ((Phiv[2:,1:-1,k]-2*Phiv[1:-1,1:-1,k]+Phiv[:-2,1:-1,k])/(delta_x**2))+((Phiv[1:-1,2:,k]-2*Phiv[1:-1,1:-1,k]+Phiv[1:-1,:-2,k])/(delta_y**2)))\
                    - Phi[1:-1,1:-1,k]*delta_t*((Phiv[2:,1:-1,k]-Phiv[:-2,1:-1,k])/(2*delta_x))+((((Phi[1:-1,1:-1,k]**2)*(delta_t**2))/(2*delta_x**2))*(Phiv[2:,1:-1,k]-2*Phiv[1:-1,1:-1,k]+Phiv[:-2,1:-1,k]))\
                    - Phiv[1:-1,1:-1,k]*delta_t*((Phiv[1:-1,2:,k]-Phiv[1:-1,:-2,k])/(2*delta_y))+((((Phiv[1:-1,1:-1,k]**2)*(delta_t**2))/(2*delta_y**2))*(Phiv[1:-1,2:,k]-2*Phiv[1:-1,1:-1,k]+Phiv[1:-1,:-2,k]))


        # Calcul divergence et terme source
        div_Phi[1:-1, 1:-1] = ((Phi[2:, 1:-1, k] - Phi[:-2, 1:-1, k]) /(2 * delta_x)) + ((Phiv[1:-1, 2:, k] - Phiv[1:-1, :-2, k]) / (2 * delta_y))
        b = (rho0*(delta_x**2)  * div_Phi) / delta_t

        ######Pression
        # @jit(nopython=True)
        # P[ int((Lslot + Lcoflow) / delta_x):,0]=P[ int((Lslot + Lcoflow) / delta_x):,1]
        # P[int((Lslot + Lcoflow) / delta_x):,-1]=P[int((Lslot + Lcoflow) / delta_x):,-2]
        # P[0,:]=P[1,:]
        # P[-1,:]=0
        # P[:int((Lslot + Lcoflow) / delta_x), 0] = 0
        # P[:int((Lslot + Lcoflow) / delta_x), -1] = 0


        P[0, :] = P[1, :]
        P[:, 0] = P[:, 1]
        P[:, -1] = P[:, -2]
        P[-1, :] = 0
        for it in range(0, iterations):
            # Pkplus1[int((Lslot + Lcoflow) / delta_x):, 0] = Pkplus1[int((Lslot + Lcoflow) / delta_x):, 1]
            # Pkplus1[int((Lslot + Lcoflow) / delta_x):, -1] = Pkplus1[int((Lslot + Lcoflow) / delta_x):, -2]
            # Pkplus1[0, :] = P[1, :]
            # Pkplus1[-1, :] = 0
            # Pkplus1[:int((Lslot + Lcoflow) / delta_x), 0] = 0
            # Pkplus1[:int((Lslot + Lcoflow) / delta_x), -1] = 0

            Pkplus1[0, :] = Pkplus1[1, :]
            Pkplus1[:, 0] = Pkplus1[:, 1]
            Pkplus1[:, -1] = Pkplus1[:, -2]
            Pkplus1[-1, :] = 0

            Pkplus1[1:-1, 1:-1] = (0.25) * (Pkplus1[:-2, 1:-1] + Pkplus1[2:, 1:-1] + Pkplus1[1:-1, 2:] + Pkplus1[1:-1, :-2] - b[1:-1, 1:-1])

        P[:, :] = Pkplus1[:, :]

        # Top et bottom wall
        Phiv[int((Lslot + Lcoflow) / delta_x):, 0, k] = 0
        Phiv[int((Lslot + Lcoflow) / delta_x):, -1, k] = 0
        Phi[int((Lslot + Lcoflow) / delta_x):, 0, k] = 0
        Phi[int((Lslot + Lcoflow) / delta_x):, -1, k] = 0

        # Lslot
        Phiv[:int(Lslot / delta_x), 0, k] = 1
        Phiv[:int(Lslot / delta_x), -1, k] = -1
        Phi[:int(Lslot / delta_x), 0, k] = 0
        Phi[:int(Lslot / delta_x), -1, k] = 0

        # Lcoflow
        Phiv[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), 0, k] = 0.2
        Phiv[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), -1, k] = -0.2
        Phi[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), 0, k] = 0
        Phi[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), -1, k] = 0

        # Mur gauche
        Phi[0, :, k] = 0
        Phiv[0, :, k] = Phiv[1, :, k]
        # Free outlet
        Phi[-1, :, k] = Phi[-2, :, k]
        Phiv[-1, :, k] = Phi[-2, :, k]
        Phifinal[1:-1, 1:-1, k+1 ] = Phi[1:-1, 1:-1, k] - ((delta_t / rho) * (P[2:, 1:-1] - P[:-2, 1:-1])) / (2 * delta_x)
        Phivfinal[1:-1, 1:-1, k+1 ] = Phiv[1:-1, 1:-1, k] - ((delta_t / rho) * (P[1:-1, 2:] - P[1:-1, :-2])) / (2 * delta_y)

        strain = np.zeros(n)
        for j in range(1, n - 1):
            strain[j] = ((Phivfinal[1, j + 1, -1] - Phivfinal[1, j - 1, -1]) / (delta_x * 2))
        print(strain)
        strain = np.absolute((strain))
        strain=np.sum(strain) / len(strain)
        print(strain)


    return Phifinal, Phivfinal,strain



##############################################################Chimie

################Constantes chimiques Masse molaire
WCH4 = 16*10**-3
WO2 = 32*10**-3
WN2 = 28*10**-3
WH2O = 18*10**-3
WCO2 = 44*10**-3
liste_W = [WCH4, WO2, WN2, WH2O, WCO2]

################Constantes chimiques Nombre stochio
nuCH4 = -1
nuO2 = -2
nuN2 = 0
nuH2O = 2
nuCO2 = 1
liste_nu = [nuCH4, nuO2, nuN2, nuH2O, nuCO2]

##############Constantes chimiques Enthalpies
DeltahCH4 = -74.9 * 10 ** 3
DeltahN2 =0
DeltahO2 = 0
DeltahH2O = -241.818 * 10 ** 3
DeltahCO2 = -393.52 * 10 ** 3
liste_Deltah = [DeltahCH4, DeltahO2, DeltahN2, DeltahH2O, DeltahCO2]



Phi,Phiv,strain=vitesse(Phi,Phiv,Phifinal,Phivfinal,div_Phi,P,delta_x,delta_y,rho0,delta_t,Lslot,Lcoflow,Pkplus1)

#Y,t,Temperature=reacteur_0D(delta_t,temps,temps_final,Qreac,Yreac,wkreac,wtreac,liste_Deltah,liste_nu,Treac,rho)
@jit(nopython=True)
def transport_espece(Phi,Phiv,Y,delta_t,delta_x):
    #Initialisation des Y
    for k in range(0, nt - 1):
       # Top et bottom wall

       # Lslot
       Y[:int(Lslot / delta_x), -1, k,0] = 1
       Y[:int(Lslot / delta_x), 0, k,1] = 0.2
       Y[:int(Lslot / delta_x), 0, k, 2] = 0.8

       # Lcoflow
       Y[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), 0, k,2] = 1
       Y[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), -1, k,2] = 1
       Y[int((Lslot + Lcoflow) / delta_x):, 0, k,:] = 0
       Y[int((Lslot + Lcoflow) / delta_x):, -1, k,:] = 0

       # Mur gauche
       Y[0, :, k,:] = Y[1, :, k,:]


       # Free outlet
       Y[-1, :, k,:] = Y[-2, :, k,:]

       #Boucle sur les espèces
       for e in range(0,4):
           derivee_seconde_Y = delta_t * (((Y[2:, 1:-1, k,e] - 2 * Y[1:-1, 1:-1, k,e] + Y[:-2, 1:-1, k,e]) / (delta_x ** 2)) + ((Y[1:-1, 2:, k,e] - 2 * Y[1:-1, 1:-1, k,e] + Y[1:-1, :-2, k,e]) / (delta_y ** 2)))
           derivee_premiere_Yx=- Phi[1:-1, 1:-1, k] * delta_t * ((Y[2:, 1:-1, k,e] - Y[:-2, 1:-1, k,e]) / (2 * delta_x)) + ((((Phi[1:-1, 1:-1, k] ** 2) * (delta_t ** 2)) / (2 * delta_x ** 2)) * (Y[2:, 1:-1, k,e] - 2 * Y[1:-1, 1:-1, k,e] + Y[:-2, 1:-1, k,e]))
           derivee_premiere_Yy =- Phiv[1:-1,1:-1,k]*delta_t*((Y[1:-1,2:,k,e]-Y[1:-1,:-2,k,e])/(2*delta_y))+((((Phiv[1:-1,1:-1,k]**2)*(delta_t**2))/(2*delta_y**2))*(Y[1:-1,2:,k,e]-2*Y[1:-1,1:-1,k,e]+Y[1:-1,:-2,k,e]))

           #Evolution de chacun des Y via transport UNIQUEMENT
           Y[1:-1, 1:-1, k + 1,e] = Y[1:-1, 1:-1, k,e] + D * derivee_seconde_Y + derivee_premiere_Yx + derivee_premiere_Yy
           print("Espece",(k / nt) * 100,"%")
           # Lslot
       Y[:int(Lslot / delta_x), -1, k, 0] = 1
       Y[:int(Lslot / delta_x), 0, k, 1] = 0.2
       Y[:int(Lslot / delta_x), 0, k, 2] = 0.8

       # Lcoflow
       Y[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), 0, k, 2] = 1
       Y[int(Lslot / delta_x):int((Lslot + Lcoflow) / delta_x), -1, k, 2] = 1
       Y[int((Lslot + Lcoflow) / delta_x):, 0, k, :] = 0
       Y[int((Lslot + Lcoflow) / delta_x):, -1, k, :] = 0

       # Mur gauche
       Y[0, :, k, :] = Y[1, :, k, :]

       # Free outlet
       Y[-1, :, k, :] = Y[-2, :, k, :]
    return Y
Y_transport=transport_espece(Phi,Phiv,Y,delta_t,delta_x)



@jit(nopython=True)
def Temperature(Phi,Phiv,Temp):
    #Initialisation des Températures

    for k in range(0, nt - 1):

        Temp[:int((Lslot + Lcoflow)/delta_x), 0, k] = 300
        Temp[:int((Lslot + Lcoflow)/delta_x), -1, k] = 300
        Temp[:, int(n / 2) - int(0.25e-3 / delta_x):int(n / 2) + int(0.25e-3 / delta_x), k ] = 1000
        # Mur gauche
        Temp[0, :,k ] = Temp[1, :,k ]

        # Free outlet
        Temp[-1, :, k] = Temp[-2, :, k]

        #Propagation de la Température
        Temp[1:-1, 1:-1, k + 1] = Temp[1:-1, 1:-1, k] + D * delta_t * (((Temp[2:, 1:-1, k] - 2 * Temp[1:-1, 1:-1, k] + Temp[:-2, 1:-1, k]) / (delta_x ** 2)) + ((Temp[1:-1, 2:, k] - 2 * Temp[1:-1, 1:-1, k] + Temp[1:-1, :-2, k]) / (delta_y ** 2))) \
                                  - Phi[1:-1, 1:-1, k] * delta_t * ((Temp[2:, 1:-1, k] - Temp[:-2, 1:-1, k]) / (2 * delta_x)) + ((((Phi[1:-1, 1:-1, k] ** 2) * (delta_t ** 2)) / (2 * delta_x ** 2)) * (Temp[2:, 1:-1, k] - 2 * Temp[1:-1, 1:-1, k] + Temp[:-2, 1:-1, k])) \
                                  - Phiv[1:-1, 1:-1, k] * delta_t * ((Temp[ 1:-1, 2:, k] - Temp[ 1:-1, :-2, k]) / (2 * delta_y)) + ((((Phiv[1:-1, 1:-1, k] ** 2) * (delta_t ** 2)) / (2 * delta_y ** 2)) * (Temp[ 1:-1, 2:, k] - 2 * Temp[1:-1, 1:-1, k] + Temp[ 1:-1, :-2, k]))

        Temp[:int((Lslot + Lcoflow) / delta_x), 0, k] = 300
        Temp[:int((Lslot + Lcoflow) / delta_x), -1, k] = 300
        Temp[:, int(n / 2) - int(0.25e-3 / delta_x):int(n / 2) + int(0.25e-3 / delta_x), k] = 1000
        # Mur gauche
        Temp[0, :, k] = Temp[1, :, k]

        # Free outlet
        Temp[-1, :, k] = Temp[-2, :, k]

        print("Temp", (k / nt) * 100, "%")
    return Temp
Temp_transport=Temperature(Phi,Phiv,Temp)
plt.contourf(x,y,Temp[:,:,-1])
plt.colorbar()
plt.title("Temp")
plt.show()

#Constantes pour la chimie
temps = 0
temps_final = 1e-5
Treac =700
Qreac = np.zeros((5))
Yreac = np.zeros((5))
wkreac = np.zeros((5))
wtreac=0

plt.contourf(x,y,Y[:,:,-1,0])
plt.colorbar()
plt.title("Methane")
plt.show()
plt.contourf(x,y,Y[:,:,-1,1])
plt.colorbar()
plt.title("O2")
plt.show()
plt.contourf(x,y,Y[:,:,-1,2])
plt.colorbar()
plt.title("N2")
plt.show()
plt.contourf(x,y,Y[:,:,-1,3])
plt.colorbar()
plt.title("H2O")
plt.show()
plt.contourf(x,y,Y[:,:,-1,4])
plt.colorbar()
plt.title("CO2")
plt.show()


def reacteur_0D(delta_t,temps,temps_final,Qreac,Yreac,wkreac,wtreac,liste_Deltah,liste_nu,Treac,rho):
    #Pas plus petit car chimie très rapide 10^-5
    delta_t=delta_t*10**-2
    #Initialisation des Y
    Yreac[0] = 0.055
    Yreac[1] = 0.233*(1-Yreac[0])
    Yreac[2] = 1-Yreac[0]-Yreac[1]
    Yreac[3]=0
    Yreac[4]=0
    tableau_temps = []
    tableau_Treac=np.zeros(((int(temps_final / delta_t)+1)))
    tableau_Y = np.zeros(((int(temps_final / delta_t)+1), 5))

    #Indice pour parcourir les tableaux
    indice = 0

    #On définit un temps final au bout duquel on ârrete de faire évoluer la chimie
    while temps<temps_final:
        wtreac = 0
        for e in range(0, 5):
            Qreac[e] = ((1.1 * 10 ** 8) * ((Yreac[0]*rho)/(16*10**-3)) * (((Yreac[1]*rho)/(32*10**-3)) ** 2)) * np.exp(-10000 / Treac )

            wkreac[ e] = liste_W[e] * liste_nu[e] * Qreac[ e]
            wtreac+=-liste_Deltah[e]*Qreac[e]*liste_nu[e]
            Yreac[e]=Yreac[e]+delta_t*wkreac[e]/rho
            tableau_Y[indice,e]=Yreac[e]
        tableau_Treac[indice]=Treac
        tableau_temps.append(temps)
        Treac=Treac+(delta_t*wtreac)/(rho*cp)

        temps = temps+delta_t
        indice=indice+1


    return tableau_Y,tableau_temps,tableau_Treac


Q=np.zeros((n,n,nt,5))
wk=np.zeros((n,n,nt,5))
wt=np.zeros((n,n,nt))
Temp_transport_chelou=np.zeros((n,n,nt))
Y_transport_chelou=np.zeros((n,n,nt,5))

def  reacteur(Temp_transport,delta_t,temps,temps_final,Q,Y_transport,wk,wt,liste_Deltah,liste_nu,rho):

    delta_t=delta_t

    for k in range(0, nt - 1):
        # Mur gauche
        Y_transport_chelou[0, :, k, :] = Y_transport_chelou[1, :, k, :]
        Temp_transport_chelou[0, :,k ] = Temp_transport_chelou[1, :,k ]

        # Free outlet
        Temp_transport_chelou[-1, :, k] = Temp_transport_chelou[-2, :, k]
        Y_transport_chelou[-1, :, k, :] = Y_transport_chelou[-2, :, k, :]


        for e in range(0, 5):
            Q[1:-1,1:-1,k,e] = ((1.1 * 10 ** 8) * ((Y_transport[1:-1,1:-1,k,0] * rho) / (16 * 10 ** -3)) * (((Y_transport[1:-1,1:-1,k,1] * rho) / (32 * 10 ** -3)) ** 2)) * np.exp(-10000 / Temp_transport[1:-1,1:-1,k])
            wk[1:-1,1:-1,k,e] = liste_W[e] * liste_nu[e] * Q[1:-1,1:-1,k,e]
            wt[1:-1,1:-1,k+1] += -liste_Deltah[e] * Q[1:-1,1:-1,k,e] * liste_nu[e]
            Y_transport_chelou[1:-1,1:-1,k+1,e] = Y_transport[1:-1,1:-1,k,e] + delta_t * wk[1:-1,1:-1,k,e] / rho

        Temp_transport_chelou[1:-1,1:-1,k+1] = Temp_transport[1:-1,1:-1,k] + (delta_t * wt[1:-1,1:-1,k]) / (rho * cp)
        # Mur gauche
        Y_transport_chelou[0, :, k, :] = Y_transport_chelou[1, :, k, :]
        Temp_transport_chelou[0, :,k ] = Temp_transport_chelou[1, :,k ]

        # Free outlet
        Temp_transport_chelou[-1, :, k] = Temp_transport_chelou[-2, :, k]
        Y_transport_chelou[-1, :, k, :] = Y_transport_chelou[-2, :, k, :]


        print("Reacteur", (k / nt) * 100, "%")
    return Y_transport_chelou,Temp_transport_chelou


norme_vitesse = np.sqrt(Phi ** 2 + Phiv ** 2)

################vitesse
plt.contourf(x,y,Phi[:,:,-1])
plt.colorbar()
plt.title("Phi")
plt.show()
plt.contourf(x,y,Phiv[:,:,-1])
plt.colorbar()
plt.title("Phiv")
plt.show()
plt.contourf(x,y,norme_vitesse[:,:,-1])
plt.colorbar()
plt.title("norme_vitesse")
plt.show()
###################Pression
plt.contourf(x,y,P[:,:])
plt.title("Pression")
plt.colorbar()
plt.show()

##############Chimie reacteur 0D
#Y,t,Temperature=reacteur_0D(delta_t,temps,temps_final,Qreac,Yreac,wkreac,wtreac,liste_Deltah,liste_nu,Treac,rho)
"""
plt.plot(t,Y[:,0],label="Methane")

plt.plot(t,Y[:,1],label="O2")


plt.plot(t,Y[:,2],label="N2")

plt.plot(t,Y[:,3],label="H2O")


plt.plot(t,Y[:,4],label="CO2")
plt.legend()
plt.show()
plt.plot(t,Temperature)
plt.title("Température")
plt.show()
plt.contourf(x,y,Temp[:,:,-1])
plt.colorbar()
plt.show()
plt.contourf(x,y,Temp[:,:,50])
plt.colorbar()
plt.show()
"""
######Reacteur
Yf,Tempf=reacteur(Temp_transport,delta_t,temps,temps_final,Q,Y_transport,wk,wt,liste_Deltah,liste_nu,rho)

#Plot reacteurs
plt.contourf(x,y,Yf[:,:,-1,0])
plt.colorbar()
plt.title("Methane")
plt.show()
plt.contourf(x,y,Yf[:,:,-1,1])
plt.colorbar()
plt.title("O2")
plt.show()
plt.contourf(x,y,Yf[:,:,-1,2])
plt.colorbar()
plt.title("N2")
plt.show()
plt.contourf(x,y,Yf[:,:,-1,3])
plt.colorbar()
plt.title("H2O")
plt.show()
plt.contourf(x,y,Yf[:,:,-1,4])
plt.colorbar()
plt.title("CO2")
plt.show()
plt.contourf(x,y,Tempf[:,:,-1])
plt.title("Temp")
plt.colorbar()
plt.show()

#Print strain
print("Strain",strain)
#Tache de N2
count = 0
for j in range(0,n):
    if Y_transport[2, j , -1,2] > 0.10 and Y_transport[2, j ,-1,2] < 0.90:
        count += 1
print(count*delta_x)

print("Tache de N2 1",count*delta_x)

#Print Temp max
print("Température max",np.amax(Tempf[:,:,-1]))
#Quivers
x = np.linspace(0,n,len(Phifinal[1]))
y = np.linspace(0,n,len(Phivfinal[2]))
plt.figure()
plt.plot(y,Phivfinal[0,:,-2])
plt.show()
plt.figure()
plt.quiver(np.transpose(Phifinal[:,:,-1]),np.transpose(Phivfinal[:,:,-1]))
plt.show()

#Convergences
liste_n=[40,50,55,70,75,80,90,100]
liste_strain=[2796,2170,1873,1541,1506,1474.9,1455.9,1459]
listetache=[0.00015,0.0008,0.00107,0.0011,0.0012,0.00149,0.0015,0.00151]
liste_temp=[0,1884.03,1885.35,1885.12,1885.38,1885.45,1884.8,1884.29]
plt.plot(liste_n,liste_strain)
plt.show()
plt.plot(liste_n,listetache)
plt.show()
plt.plot(liste_n,liste_temp)
plt.show()
fig = plt.figure()
i=0
im = plt.imshow(norme_vitesse[:,:,0],animated=True,cmap='plasma')
n=1000
plt.colorbar()
#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(norme_vitesse[:,:,i-1])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("vitesse.gif")

fig = plt.figure()
i=0
im = plt.imshow(Y_transport_chelou[:,:,0,0],animated=True,cmap='plasma')
n=1000
plt.colorbar()

#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(Y_transport_chelou[:,:,i-1,0])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("Methane.gif")

fig = plt.figure()
i=0
im = plt.imshow(Y_transport_chelou[:,:,0,1],animated=True,cmap='plasma')
n=1000
plt.colorbar()

#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(Y_transport_chelou[:,:,i-1,1])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("O2.gif")


fig = plt.figure()
i=0
im = plt.imshow(Y_transport_chelou[:,:,0,2],animated=True,cmap='plasma')
n=1000
plt.colorbar()

#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(Y_transport_chelou[:,:,i-1,2])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("N2.gif")


fig = plt.figure()
i=0
im = plt.imshow(Y_transport_chelou[:,:,0,3],animated=True,cmap='plasma')
n=1000
plt.colorbar()

#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(Y_transport_chelou[:,:,i-1,3])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("H2O.gif")


fig = plt.figure()
i=0
im = plt.imshow(Y_transport_chelou[:,:,0,4],animated=True,cmap='plasma')
n=1000
plt.colorbar()

#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(Y_transport_chelou[:,:,i-1,3])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("H2O.gif")

fig = plt.figure()
i=0
im = plt.imshow(Temp_transport_chelou[:,:,0],animated=True,cmap='plasma')
n=1000
plt.colorbar()

#@jit(nopython=True)
def updatefig(*args) :
    global i
    if (i<n):
        i += 20
    else :
        i=0
    im.set_array(Temp_transport_chelou[:,:,i-1])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=5000, blit=True)
ani.save("Temperature.gif")
