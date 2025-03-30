# main.py
import numpy as np
import matplotlib.pyplot as plt
import config
import os
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
import time

# Створюємо папку для збереження результатів
if not os.path.exists("results"):
    os.makedirs("results")

plt.ion()

# Функція 1: Ініціалізація сітки
def initialize_grid():
    x = np.linspace(-4, 4, config.nx)
    z = np.linspace(0, 6, config.nz)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z

# Функція 2: Визначення геометрії каналу та ініціалізація умов
def initialize_conditions(x, z):
    # Геометрія каналу (один раз задаємо межі)
    channel_left = np.where(z >= 3, -0.1, -0.1 - (2 - 0.1) / 3 * (3 - z))
    channel_right = np.where(z >= 3, 0.1, 0.1 + (2 - 0.1) / 3 * (3 - z))

    # Маска каналу (True там, де канал)
    channel_mask = np.zeros((config.nx, config.nz), dtype=bool)
    for j in range(config.nz):
        x_left = channel_left[j]
        x_right = channel_right[j]
        channel_mask[:, j] = (x >= x_left) & (x <= x_right)

    # Маска землі (True там, де земля)
    soil_mask = ~channel_mask

    # Ініціалізація концентрації
    q = np.zeros((config.nx, config.nz))
    q[:, -1] = config.q_top  # Початкова концентрація тільки на поверхні
    
    return q, channel_left, channel_right, channel_mask, soil_mask

# Функція 3: Побудова матриць для неявної схеми (лише дифузія)
def build_matrices(soil_mask):
    N = config.nx * config.nz
    A = lil_matrix((N, N))  # Матриця для неявної частини
    B = lil_matrix((N, N))  # Матриця для явної частини
    
    alpha_x = config.D * config.dt / (2 * config.dx**2)
    alpha_z = config.D * config.dt / (2 * config.dz**2)
    
    for j in range(config.nz):
        for i in range(config.nx):
            idx = i + j * config.nx
            if soil_mask[i, j]:  # Дифузія тільки в землі
                A[idx, idx] = 1 + 2 * alpha_x + 2 * alpha_z
                B[idx, idx] = 1 - 2 * alpha_x - 2 * alpha_z
                
                if i > 0 and soil_mask[i-1, j]:
                    A[idx, idx - 1] = -alpha_x
                    B[idx, idx - 1] = alpha_x
                if i < config.nx - 1 and soil_mask[i+1, j]:
                    A[idx, idx + 1] = -alpha_x
                    B[idx, idx + 1] = alpha_x
                
                if j > 0 and soil_mask[i, j-1]:
                    A[idx, idx - config.nx] = -alpha_z
                    B[idx, idx - config.nx] = alpha_z
                if j < config.nz - 1 and soil_mask[i, j+1]:
                    A[idx, idx + config.nx] = -alpha_z
                    B[idx, idx + config.nx] = alpha_z
            else:  # У каналі немає дифузії
                A[idx, idx] = 1
                B[idx, idx] = 1
    
    A = A.tocsr()
    B = B.tocsr()
    
    return A, B

# Функція 4: Оновлення концентрації (неявна схема, лише дифузія в землі)
def update_concentration(q, soil_mask):
    A, B = build_matrices(soil_mask)
    q_flat = q.flatten()
    rhs = B @ q_flat
    q_new_flat = spsolve(A, rhs)
    q_new = q_new_flat.reshape((config.nx, config.nz))
    q_new[~soil_mask] = 0.0  # У каналі концентрація завжди 0
    q_new = np.clip(q_new, 0, 1e3)
    return q_new

# Функція 5: Застосування граничних умов
def apply_boundary_conditions(q, soil_mask):
    q_new = q.copy()
    q_new[:, -1] = config.q_top  # Верхня межа (поверхня)
    q_new[:, 0] = q_new[:, 1]    # Нижня межа: ∂q/∂z = 0
    q_new[0, :] = q_new[1, :]    # Ліва межа: ∂q/∂x = 0
    q_new[-1, :] = q_new[-2, :]  # Права межа: ∂q/∂x = 0
    q_new[~soil_mask] = 0.0      # У каналі концентрація 0
    return q_new

# Функція 6: Перевірка умов зупинки
def check_stopping_condition(q, time):
    min_q = np.min(q)
    max_q = np.max(q)
    if np.isnan(min_q):
        print(f"Знайдено nan на часі {time / (3600 * 24):.2f} днів.")
        return True
    if np.all(q >= config.q_threshold):
        print(f"Увесь зразок забруднений (> {config.q_threshold}) через {time / (3600 * 24):.2f} днів")
        return True
    if time % 50000 == 0:
        print(f"Час: {time / (3600 * 24):.2f} днів, Min: {min_q:.4f}, Max: {max_q:.4f}")
    return False

# Функція 7: Візуалізація з межами областей і збереженням
def visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника", save=True):
    plt.clf()
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
    
    plt.draw()
    plt.pause(0.001)

# Основна функція моделювання
def run_simulation():
    x, z, X, Z = initialize_grid()
    q, channel_left, channel_right, channel_mask, soil_mask = initialize_conditions(x, z)
    
    snapshot_times = [1 * 24 * 3600, 10 * 24 * 3600]
    snapshots_taken = {t: False for t in snapshot_times}
    
    time = 0
    visualization_interval = 5000
    
    while time < config.max_time:
        q_new = update_concentration(q, soil_mask)
        q_new = apply_boundary_conditions(q_new, soil_mask)
        q = q_new
        time += config.dt
        
        if time % visualization_interval < config.dt:
            visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, save=False)
        
        for snapshot_time in snapshot_times:
            if not snapshots_taken[snapshot_time] and abs(time - snapshot_time) < config.dt:
                visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, save=True)
                snapshots_taken[snapshot_time] = True
        
        if check_stopping_condition(q, time):
            break
    
    visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника (кінцевий результат)", save=True)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()