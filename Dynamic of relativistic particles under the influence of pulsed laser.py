import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal

Position_Aff=1

N=1 #nombre de particules
M=1000
Mx=M #☺taille de la boite
My=M
#Mz=M
taille = 1e-5
it=10000#nombre d'iterations pour que la laser atteigne le bout de la boite

m=9.1e-31 #masse
e=-1.602e-19 #charge
u0=1.2566e-6 #mu0
eps0=8.854e-12 #eps0
C=3e8 #vitesse lumière

T0=0
T1=taille/C #en gros 1e-13
Energie = 1e-3 #energie en Joule
Dia=10*1e-6 #diameètre de focalisation du laser
THO = 30*1e-15 #temps de l'impulsion laser
dt=T1/it

Puissance=Energie/THO
Surface=np.pi*np.square(Dia/2)
Intensite=Puissance/Surface
print('Intensité du Laser :',Intensite,'W.m-2')

E0=np.sqrt(Intensite/(C*eps0/2))
print('E0 =',E0)
B0=E0/C

x0=np.zeros((N,3))
v0=np.zeros((N,3))

CE0=np.zeros((3))
CM0=np.zeros((3))



dx=taille/Mx
dy=taille/My
dz=taille/My

x = np.arange(0, taille, dx)
y = np.arange(0, taille, dy)

_lambda = 1e-6
k = (2 * np.pi) / _lambda
omega = k * C

def gaussian2d(xymesh, mx, my, sig_x, sig_y):
    (x, y) = xymesh
    gauss2D = (np.exp(-((x-mx)**2/(2*sig_x**2) + (y-my)**2/(2*sig_y**2)) ))
    return np.ravel(gauss2D)

def Laser_(T,posi):
    CE_=np.zeros((3))
    CB_=np.zeros((3))
    xi=posi[0]
    yi=posi[1]
    gauss=gaussian2d((xi, yi), C*T-taille,taille/2 , C * THO /4, Dia/4 )
    CE_[1]=E0* np.cos(omega * T - k * xi)*gauss
    CB_[2]=B0*np.cos(omega * T - k * xi)*gauss
    return CE_,CB_


def Vit(vitesse):
    V=(np.sqrt(np.square(vitesse[0,0])+np.square(vitesse[0,1])))
    return V

def gamma(vitesse):
    gam=np.sqrt(1+np.square((vitesse/C)))
    return gam



CM0[2]=B0
CE0[0]=E0

dt2=dt/2
nt=dt*it
x0[:,0]=0.5*taille
x0[:,1]=0.5*taille
v0[:,0]=0


print('dt=',dt)

#------------------------------
class Particule:
    "def"
    
p= Particule()
p.position=x0
p.vitesse=v0
p.m=m
p.q=e
#------------------------------

qm=p.q/p.m

pmoy=np.zeros((N,3))

vmoins=p.vitesse
umoins=vmoins

umoy=umoins
pmoins=p.position
v_moy=Vit(umoy)
gamma_=gamma(v_moy)
gam_=gamma_
gam_m=gam_
print(gam_m)
print('gamma 0 =',gamma_)

T=0.0*taille/(C) #set up du temps ou on commence pour asvoir a quel endroit de la boite est le laser
Beta=v_moy/C

tps_it=4*it*2#le facteur multiplicateur devant indique la portion de la boite que le laser parcours pendant la simu

Posi_X=np.zeros(((int(tps_it/2))))
Posi_Y=np.zeros(((int(tps_it/2))))
Posi_X[0]=p.position[0,0]
Posi_Y[0]=p.position[0,1]
Vitesse_moyenne=np.zeros((int(tps_it/2)))
U_moins=np.zeros((int(tps_it/2)))
E_part=np.zeros((int(tps_it/2)))
B_part=np.zeros((int(tps_it/2)))
gamma_part=np.ones((int(tps_it/2)))
Tps_part=np.zeros((int(tps_it/2)))
#Position_moyenne=np.zeros((tps_it,2))

if(Position_Aff==1):
    for j in range(1,int(tps_it)):
        i=0
        if j<0.1*tps_it and j%(tps_it/100)==0 :
            print(int((j/tps_it)*100),'%')
        if j%(tps_it/10)==0 :
            print(int((j/tps_it)*100),'%')
        if j>0.9*tps_it and j%(tps_it/100)==0 :
            print(int((j/tps_it)*100),'%')
        if j %2!=0:
            gam_moins=gamma(Vit(umoins))
            p.position=((umoins*dt)/gam_moins)+pmoins
            pmoins=p.position
            CE0,CB0=Laser_(T,p.position[i])
            E_part[int(j/2)]=np.linalg.norm(CE0)
            B_part[int(j/2)]=np.linalg.norm(CB0)

        else:

            um=(umoins+qm*(CE0)*dt2)
            gam_m=gamma(Vit(um))
            t=qm*dt2*CB0/gam_m
            uprime=um+np.cross(um,t)
            s=2*t/(1+t**2)
            uplus=um+np.cross(uprime,s)
            up=uplus+qm*CE0*dt2
            umoy=(up+umoins)/2
            umoins=up
            gam_moy=gamma(Vit(umoy))
            gamma_part[int(j/2)]=gam_moy
            Posi_X[int(j/2)]=p.position[0,0]
            Posi_Y[int(j/2)]=p.position[0,1]
            
            U_moins[int(j/2)]=Vit(umoins)
            Vitesse_moyenne[int(j/2)]=Vit(umoy/gam_moy)
            Tps_part[int(j/2)]=T

        T=T+dt2
    
    


#plt.figure(figsize=(5,3))
plt.title("Position de la particule en fonction du temps pour une \n énergie E = {} J et un temps d'impulsion THO = {} fs\n".format(Energie,int(THO*1e15)))
plt.ylabel("Position y [m]")
plt.xlabel("Position x [m]")
plt.plot(Posi_X,Posi_Y)
plt.show()


plt.title("Norme de la vitesse de la particule en fonction du temps pour une \n énergie E = {} J et un temps d'impulsion THO = {} fs\n".format(Energie,int(THO*1e15)))
plt.ylabel("Vitesse [m/s]")
plt.xlabel("Temps [s]")
plt.plot(Tps_part,Vitesse_moyenne)
plt.show()


plt.title("Norme de la vitesse de la particule en fonction de sa position en X pour une \n énergie E = {} J et un temps d'impulsion THO = {} fs\n".format(Energie,int(THO*1e15)))
plt.ylabel("Vitesse [m/s]")
plt.xlabel("Position x [m]")
plt.plot(Posi_X,Vitesse_moyenne)
plt.show()

plt.title("Position en Y de la particule en fonction du temps pour une \n énergie E = {} J et un temps d'impulsion THO = {} fs\n".format(Energie,int(THO*1e15)))
plt.ylabel("Position y [m]")
plt.xlabel("Temps [s]")
plt.plot(Tps_part,Posi_Y)
plt.show()


plt.plot(gamma_part)
plt.show()




#---------------------------------------------------------------
#partie affichage vitesse en fonction du temps
"""
for j in range(0,tps_it/2):
    plt.scatter(j,EM[j])
    plt.plot(p.position[0,0],p.position[0,1])
    plt.axis('square')
    plt.set_aspect('equal', adjustable='box')
    plt.xlabel('Tps')
    plt.ylabel('EM')
"""
#plt.plot(Vitesse_moyenne)

#plt.plot(Posi_X,Vitesse_moyenne)

#---------------------------------------------------------------
"""
#plt.figure(figsize=(5,3))
plt.title("Champ électrique dans la boite à l'instant T = {} \n".format(T))
#plt.title('f model: T= {}'.format(t))
plt.ylabel("Position y [m]")
plt.xlabel("Position x [m]")
plt.tight_layout()

print('Afficahge Laser')

mapg = "seismic"
#mapg='gnuplot'
plt.pcolormesh(x, y, np.transpose(E), cmap = mapg)
#plt.pcolormesh(x, y, E, cmap = mapg)
plt.axis('scaled')
plt.colorbar()
#plt.axis([0.41e-5, 0.42e-5, 0, 1e-5])
plt.show
#"""
