# main.py
import numpy as np
import matplotlib.pyplot as plt
import config
import os
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve

# Створюємо папку для збереження результатів
if not os.path.exists("results"):
    os.makedirs("results")

# Функція 1: Ініціалізація сітки
def initialize_grid():
    x = np.linspace(-4, 4, config.nx)
    z = np.linspace(0, 6, config.nz)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z

# Функція 2: Ініціалізація концентрації та швидкості
def initialize_conditions(x, z):
    channel_left = -0.1 - (2 - 0.1) / 6 * (6 - z)
    channel_right = 0.1 + (2 - 0.1) / 6 * (6 - z)
    
    q = np.zeros((config.nx, config.nz))
    v_z = np.zeros((config.nx, config.nz))
    
    for j in range(config.nz):
        z_val = z[j]
        v_z_at_z = -0.02 + (0.02 - 0.001) / 6 * (6 - z_val)
        x_left = -0.1 - (2 - 0.1) / 6 * (6 - z_val)
        x_right = 0.1 + (2 - 0.1) / 6 * (6 - z_val)
        channel_indices = np.where((x >= x_left) & (x <= x_right))[0]
        for i in channel_indices:
            v_z[i, j] = v_z_at_z
    
    return q, v_z, channel_left, channel_right

# Функція 3: Побудова матриць для неявної схеми
def build_matrices(q, v_z):
    N = config.nx * config.nz
    A = lil_matrix((N, N))  # Матриця для неявної частини
    B = lil_matrix((N, N))  # Матриця для явної частини
    
    alpha_x = config.D * config.dt / (2 * config.dx**2)
    alpha_z = config.D * config.dt / (2 * config.dz**2)
    beta_z = config.dt / (4 * config.dz)  # Для адвекції
    
    for j in range(config.nz):
        for i in range(config.nx):
            idx = i + j * config.nx  # Індекс у розгорнутому векторі
            
            # Діагональні елементи
            A[idx, idx] = 1 + 2 * alpha_x + 2 * alpha_z
            B[idx, idx] = 1 - 2 * alpha_x - 2 * alpha_z
            
            # Елементи для дифузії по x
            if i > 0:
                A[idx, idx - 1] = -alpha_x
                B[idx, idx - 1] = alpha_x
            if i < config.nx - 1:
                A[idx, idx + 1] = -alpha_x
                B[idx, idx + 1] = alpha_x
            
            # Елементи для дифузії та адвекції по z
            if j > 0:
                A[idx, idx - config.nx] = -alpha_z + beta_z * v_z[i, j]
                B[idx, idx - config.nx] = alpha_z - beta_z * v_z[i, j]
            if j < config.nz - 1:
                A[idx, idx + config.nx] = -alpha_z - beta_z * v_z[i, j]
                B[idx, idx + config.nx] = alpha_z + beta_z * v_z[i, j]
    
    # Перетворюємо в формат CSR для швидкого розв’язання
    A = A.tocsr()
    B = B.tocsr()
    
    return A, B

# Функція 4: Оновлення концентрації (неявна схема)
def update_concentration(q, v_z):
    A, B = build_matrices(q, v_z)
    
    # Розгортаємо q у вектор
    q_flat = q.flatten()
    
    # Обчислюємо праву частину
    rhs = B @ q_flat
    
    # Розв’язуємо систему A * q_new = rhs
    q_new_flat = spsolve(A, rhs)
    
    # Перетворюємо назад у матрицю
    q_new = q_new_flat.reshape((config.nx, config.nz))
    
    # Обмеження значень
    q_new = np.where(q_new < 0, 0, q_new)
    q_new = np.where(q_new > 1e3, 1e3, q_new)
    
    return q_new

# Функція 5: Застосування граничних умов
def apply_boundary_conditions(q):
    q_new = q.copy()
    q_new[:, -1] = config.q_top  # Верхня межа
    q_new[:, 0] = q_new[:, 1]    # Нижня межа: ∂q/∂z = 0
    q_new[0, :] = q_new[1, :]    # Ліва межа: ∂q/∂x = 0
    q_new[-1, :] = q_new[-2, :]  # Права межа: ∂q/∂x = 0
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
def visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника"):
    plt.figure(figsize=(8, 6))
    plt.contourf(X.T, Z.T, q, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='Концентрація (кг/м³)')
    
    # Позначення меж каналу (конус)
    plt.plot(channel_left, z, color='red', linestyle='--', linewidth=1.5, label='Межі каналу')
    plt.plot(channel_right, z, color='red', linestyle='--', linewidth=1.5)
    
    # Позначення ізольованих меж
    plt.axvline(x=-4, color='blue', linestyle='-', linewidth=1.5, label='Ізольовані межі')
    plt.axvline(x=4, color='blue', linestyle='-', linewidth=1.5)
    plt.axhline(y=0, color='blue', linestyle='-', linewidth=1.5)
    plt.axhline(y=6, color='blue', linestyle='-', linewidth=1.5)
    
    plt.title(f'{title} через {time / (3600 * 24):.2f} днів')
    plt.xlabel('x (м)')
    plt.ylabel('z (м)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    filename = f"results/concentration_{time / (3600 * 24):.2f}_days.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Зображення збережено як {filename}")
    plt.show()

# Основна функція моделювання
def run_simulation():
    x, z, X, Z = initialize_grid()
    q, v_z, channel_left, channel_right = initialize_conditions(x, z)
    
    snapshot_times = [1 * 24 * 3600, 10 * 24 * 3600]
    snapshots_taken = {t: False for t in snapshot_times}
    
    time = 0
    
    while time < config.max_time:
        q_new = update_concentration(q, v_z)
        q_new = apply_boundary_conditions(q_new)
        q = q_new
        time += config.dt
        
        for snapshot_time in snapshot_times:
            if not snapshots_taken[snapshot_time] and abs(time - snapshot_time) < config.dt:
                visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника (проміжний етап)")
                snapshots_taken[snapshot_time] = True
        
        if check_stopping_condition(q, time):
            break
    
    visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника (кінцевий результат)")

if __name__ == "__main__":
    run_simulation()