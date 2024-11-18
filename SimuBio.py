import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres
D = 1e-6  # Coefficient de diffusion (général)
k_decay = 1e-3  # Taux de dégradation
r_fibro = 0.01  # Taux de prolifération des fibroblastes
r_kera = 0.01  # Taux de prolifération des kératinocytes
r_cyto = 0.01  # Taux de production des cytokines
r_angiogen = 0.01  # Taux de production des facteurs angiogéniques
r_ECM = 0.01  # Taux de production de la matrice extracellulaire
alpha = 0.1  # Effet stimulant des RONS sur les fibroblastes
beta = 0.05  # Effet inhibiteur des cytokines sur les fibroblastes
gamma = 0.1  # Effet stimulant des RONS sur les kératinocytes
M_fibro = 0.01  # Coefficient de migration des fibroblastes

# Dimensions de la plaie
L = 1.0  # Longueur de la plaie
Nx = 100  # Nombre de points de discrétisation
dx = L / Nx  # Taille de la cellule

# Conditions initiales
C_RONS = np.zeros(Nx)
C_RONS[int(Nx / 2)] = 1.0
C_fibro = np.ones(Nx) * 0.1
C_kera = np.ones(Nx) * 0.1
C_cyto = np.zeros(Nx)
C_macroph = np.zeros(Nx)
C_angiogen = np.zeros(Nx)
C_ECM = np.zeros(Nx)


# Équations de cicatrisation des plaies
def wound_healing(t, y):
    C_RONS = y[:Nx]
    C_fibro = y[Nx:2 * Nx]
    C_kera = y[2 * Nx:3 * Nx]
    C_cyto = y[3 * Nx:4 * Nx]
    C_macroph = y[4 * Nx:5 * Nx]
    C_angiogen = y[5 * Nx:6 * Nx]
    C_ECM = y[6 * Nx:]

    dC_RONS_dt = np.zeros_like(C_RONS)
    dC_fibro_dt = np.zeros_like(C_fibro)
    dC_kera_dt = np.zeros_like(C_kera)
    dC_cyto_dt = np.zeros_like(C_cyto)
    dC_macroph_dt = np.zeros_like(C_macroph)
    dC_angiogen_dt = np.zeros_like(C_angiogen)
    dC_ECM_dt = np.zeros_like(C_ECM)

    for i in range(1, Nx - 1):
        dC_RONS_dt[i] = D * (C_RONS[i + 1] - 2 * C_RONS[i] + C_RONS[i - 1]) / dx ** 2 - k_decay * C_RONS[i]
        dC_fibro_dt[i] = r_fibro * C_fibro[i] * (1 + alpha * C_RONS[i] - beta * C_cyto[i]) + M_fibro * (
                    C_fibro[i + 1] - C_fibro[i - 1]) / (2 * dx)
        dC_kera_dt[i] = r_kera * C_kera[i] * (1 + gamma * C_RONS[i]) + M_fibro * (C_kera[i + 1] - C_kera[i - 1]) / (
                    2 * dx)
        dC_cyto_dt[i] = r_cyto * C_macroph[i] - k_decay * C_cyto[i] + D * (
                    C_cyto[i + 1] - 2 * C_cyto[i] + C_cyto[i - 1]) / dx ** 2
        dC_macroph_dt[i] = r_cyto * (C_RONS[i] + C_cyto[i]) - k_decay * C_macroph[i] + D * (
                    C_macroph[i + 1] - 2 * C_macroph[i] + C_macroph[i - 1]) / dx ** 2
        dC_angiogen_dt[i] = r_angiogen * (C_fibro[i] + C_macroph[i]) - k_decay * C_angiogen[i] + D * (
                    C_angiogen[i + 1] - 2 * C_angiogen[i] + C_angiogen[i - 1]) / dx ** 2
        dC_ECM_dt[i] = r_ECM * C_fibro[i] - k_decay * C_ECM[i] + D * (
                    C_ECM[i + 1] - 2 * C_ECM[i] + C_ECM[i - 1]) / dx ** 2

    return np.concatenate([dC_RONS_dt, dC_fibro_dt, dC_kera_dt, dC_cyto_dt, dC_macroph_dt, dC_angiogen_dt, dC_ECM_dt])


# Conditions initiales combinées
y0 = np.concatenate([C_RONS, C_fibro, C_kera, C_cyto, C_macroph, C_angiogen, C_ECM])
time = np.linspace(0, 10, 100)  # Temps de simulation

# Simulation
result = solve_ivp(wound_healing, [0, 10], y0, t_eval=time)

# Extraction des résultats
RONS_result = result.y[:Nx, :]
fibro_result = result.y[Nx:2 * Nx, :]
kera_result = result.y[2 * Nx:3 * Nx, :]
cyto_result = result.y[3 * Nx:4 * Nx, :]
macroph_result = result.y[4 * Nx:5 * Nx, :]
angiogen_result = result.y[5 * Nx:6 * Nx, :]
ECM_result = result.y[6 * Nx:, :]

# Affichage des résultats
plt.figure(figsize=(14, 10))

plt.subplot(231)
for i in range(0, 100, 20):
    plt.plot(np.linspace(0, L, Nx), RONS_result[:, i], label=f't={time[i]:.1f}')
plt.xlabel('Position')
plt.ylabel('Concentration de RONS')
plt.title('Diffusion et dégradation des RONS')
plt.legend()

plt.subplot(232)
for i in range(0, 100, 20):
    plt.plot(np.linspace(0, L, Nx), fibro_result[:, i], label=f't={time[i]:.1f}')
plt.xlabel('Position')
plt.ylabel('Concentration de fibroblastes')
plt.title('Prolifération des fibroblastes')
plt.legend()

plt.subplot(233)
for i in range(0, 100, 20):
    plt.plot(np.linspace(0, L, Nx), kera_result[:, i], label=f't={time[i]:.1f}')
plt.xlabel('Position')
plt.ylabel('Concentration de kératinocytes')
plt.title('Prolifération des kératinocytes')
plt.legend()

plt.subplot(234)
for i in range(0, 100, 20):
    plt.plot(np.linspace(0, L, Nx), cyto_result[:, i], label=f't={time[i]:.1f}')
plt.xlabel('Position')
plt.ylabel('Concentration de cytokines')
plt.title('Production de cytokines')
plt.legend()

plt.subplot(235)
for i in range(0, 100, 20):
    plt.plot(np.linspace(0, L, Nx), macroph_result[:, i], label=f't={time[i]:.1f}')
plt.xlabel('Position')
plt.ylabel('Concentration de macrophages')
plt.title('Recrutement des macrophages')
plt.legend()

plt.subplot(236)
for i in range(0, 100, 20):
    plt.plot(np.linspace(0, L, Nx), angiogen_result[:, i], label=f't={time[i]:.1f}')
plt.xlabel('Position')
plt.ylabel('Concentration des facteurs angiogéniques')
plt.title('Angiogenèse')
plt.legend()

plt.tight_layout()
plt.show()
