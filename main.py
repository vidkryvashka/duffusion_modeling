# main.py
# import numpyTimestep: 1157/1157 [00:00<00:00, 12327.79it/s] numpy
import numpy as np
import matplotlib.pyplot as plt
import config
import os
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from numba import jit
import time  # Додано для вимірювання часу (опціонально)

# Створюємо папку для збереження результатів
if not os.path.exists("results"):
    os.makedirs("results")

# Увімкнення інтерактивного режиму для matplotlib
plt.ion()

# Функція 1: Ініціалізація сітки
def initialize_grid():
    x = np.linspace(-4, 4, config.nx)
    z = np.linspace(0, 6, config.nz)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z

# Функція 2: Ініціалізація концентрації та швидкості
def initialize_conditions(x, z):
    channel_left = np.where(z >= 3, -0.1, -0.1 - (2 - 0.1) / 3 * (3 - z))
    channel_right = np.where(z >= 3, 0.1, 0.1 + (2 - 0.1) / 3 * (3 - z))

    q = np.zeros((config.nx, config.nz))
    v_z = np.zeros((config.nx, config.nz))

    q[:, -1] = config.q_top  

    for j in range(config.nz):
        z_val = z[j]
        x_left = -0.1 - (2 - 0.1) / 6 * (6 - z_val)
        x_right = 0.1 + (2 - 0.1) / 6 * (6 - z_val)
        channel_indices = np.where((x >= x_left) & (x <= x_right))[0]

        for i in channel_indices:
            v_z[i, j] = config.v_z_channel + (0.02 - 0.001) / 6 * (6 - z_val)

    return q, v_z, channel_left, channel_right

# Функція 3: Побудова матриць для неявної схеми
def build_matrices(q, v_z):
    N = config.nx * config.nz
    A = lil_matrix((N, N))
    B = lil_matrix((N, N))
    
    alpha_x = config.D * config.dt / (2 * config.dx**2)
    alpha_z = config.D * config.dt / (2 * config.dz**2)
    beta_z = config.dt / (4 * config.dz)
    
    for j in range(config.nz):
        for i in range(config.nx):
            idx = i + j * config.nx
            A[idx, idx] = 1 + 2 * alpha_x + 2 * alpha_z
            B[idx, idx] = 1 - 2 * alpha_x - 2 * alpha_z
            
            if i > 0:
                A[idx, idx - 1] = -alpha_x
                B[idx, idx - 1] = alpha_x
            if i < config.nx - 1:
                A[idx, idx + 1] = -alpha_x
                B[idx, idx + 1] = alpha_x
            
            if j > 0:
                A[idx, idx - config.nx] = -alpha_z + beta_z * v_z[i, j]
                B[idx, idx - config.nx] = alpha_z - beta_z * v_z[i, j]
            if j < config.nz - 1:
                A[idx, idx + config.nx] = -alpha_z - beta_z * v_z[i, j]
                B[idx, idx + config.nx] = alpha_z + beta_z * v_z[i, j]
    
    A = A.tocsr()
    B = B.tocsr()
    
    return A, B

# Функція 4: Оновлення концентрації (неявна схема)
def update_concentration(q, v_z, x, z, channel_left, channel_right):
    A, B = build_matrices(q, v_z)
    q_flat = q.flatten()
    rhs = B @ q_flat
    q_new_flat = spsolve(A, rhs)
    q_new = q_new_flat.reshape((config.nx, config.nz))
    channel_mask = get_channel_mask(x, z, channel_left, channel_right)
    q_new = np.where(~channel_mask, np.clip(q_new, 0, 1e3), 0.0)
    return q_new

# Функція для маски каналу
def get_channel_mask(x, z, channel_left, channel_right):
    mask = np.zeros((len(x), len(z)), dtype=bool)
    for j in range(len(z)):
        x_left = channel_left[j]
        x_right = channel_right[j]
        mask[:, j] = (x >= x_left) & (x <= x_right)
    return mask

# Функція 5: Застосування граничних умов
def apply_boundary_conditions(q, x, z, channel_left, channel_right):
    q_new = q.copy()
    q_new[:, -1] = config.q_top
    q_new[:, 0] = q_new[:, 1]
    q_new[0, :] = q_new[1, :]
    q_new[-1, :] = q_new[-2, :]
    return q_new

# Функція 6: Перевірка умов зупинки
def check_stopping_condition(q, time):
    min_q = np.min(q)
    if np.isnan(min_q):
        print(f"Знайдено nan на часі {time / (3600 * 24):.2f} днів. Зупиняємо моделювання.")
        return True
    if np.all(q >= config.q_threshold):
        print(f"Увесь зразок забруднений (> {config.q_threshold} кг/м³) через {time / (3600 * 24):.2f} днів")
        return True
    if time % 50000 == 0:
        print(f"Час: {time / (3600 * 24):.2f} днів, Мінімальна концентрація: {min_q:.4f} кг/м³")
    return False

# Функція 7: Візуалізація з межами областей і збереженням
def visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника", save=True):
    plt.clf()  # Очищаємо попередній графік
    plt.contourf(X.T, Z.T, q, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='Концентрація (кг/м³)')
    plt.plot(channel_left, z, 'r--', linewidth=1.5, label='Межі каналу')
    plt.plot(channel_right, z, 'r--', linewidth=1.5)
    plt.title(f'{title} через {time / (3600 * 24):.2f} днів')
    plt.xlabel('x (м)')
    plt.ylabel('z (м)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    if save:
        plt.savefig(f"results/concentration_{time / (3600 * 24):.2f}_days.png", dpi=300, bbox_inches='tight')
    
    plt.draw()  # Оновлюємо графік
    plt.pause(0.001)  # Коротка пауза для відображення

# Основна функція моделювання
def run_simulation():
    x, z, X, Z = initialize_grid()
    q, v_z, channel_left, channel_right = initialize_conditions(x, z)
    
    snapshot_times = [1 * 24 * 3600, 10 * 24 * 3600]  # Час для збереження знімків
    snapshots_taken = {t: False for t in snapshot_times}
    
    time = 0
    visualization_interval = 5000  # Оновлювати візуалізацію кожні 50 секунд
    
    while time < config.max_time:
        q_new = update_concentration(q, v_z, x, z, channel_left, channel_right)
        q_new = apply_boundary_conditions(q_new, x, z, channel_left, channel_right)
        q = q_new
        time += config.dt
        
        # Періодичне оновлення візуалізації
        if time % visualization_interval < config.dt:
            visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, save=False)
        
        # Збереження знімків у потрібні моменти
        for snapshot_time in snapshot_times:
            if not snapshots_taken[snapshot_time] and abs(time - snapshot_time) < config.dt:
                visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, save=True)
                snapshots_taken[snapshot_time] = True
        
        if check_stopping_condition(q, time):
            break
    
    # Фінальна візуалізація
    visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника (кінцевий результат)", save=True)
    plt.ioff()  # Вимикаємо інтерактивний режим
    plt.show()  # Показуємо фінальний графік

if __name__ == "__main__":
    run_simulation()