import numpy as np
import matplotlib.pyplot as plt
import config
import os

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
    # Визначаємо межі каналу як функцію z
    channel_left = -0.1 - (2 - 0.1) / 6 * (6 - z)
    channel_right = 0.1 + (2 - 0.1) / 6 * (6 - z)
    
    q = np.zeros((config.nx, config.nz))
    v_z = np.zeros((config.nx, config.nz))
    
    # Змінна швидкість v_z залежно від z
    for j in range(config.nz):
        z_val = z[j]
        # Швидкість змінюється лінійно від -0.02 (при z=6) до -0.001 (при z=0)
        v_z_at_z = -0.02 + (0.02 - 0.001) / 6 * (6 - z_val)
        # Визначаємо межі каналу на даній висоті
        x_left = -0.1 - (2 - 0.1) / 6 * (6 - z_val)
        x_right = 0.1 + (2 - 0.1) / 6 * (6 - z_val)
        channel_indices = np.where((x >= x_left) & (x <= x_right))[0]
        for i in channel_indices:
            v_z[i, j] = v_z_at_z
    
    return q, v_z, channel_left, channel_right

# Функція 3: Оновлення концентрації
def update_concentration(q, v_z):
    q_new = q.copy()
    for i in range(1, config.nx-1):
        for j in range(1, config.nz-1):
            adv_x = 0
            adv_z = v_z[i, j] * (q[i, j] - q[i, j-1]) / config.dz if v_z[i, j] != 0 else 0
            diff_x = config.D * (q[i+1, j] - 2*q[i, j] + q[i-1, j]) / (config.dx**2)
            diff_z = config.D * (q[i, j+1] - 2*q[i, j] + q[i, j-1]) / (config.dz**2)
            q_new[i, j] = q[i, j] + config.dt * (-adv_x - adv_z + diff_x + diff_z)
            if np.isnan(q_new[i, j]) or q_new[i, j] < 0:
                q_new[i, j] = 0
            if q_new[i, j] > 1e3:
                q_new[i, j] = 1e3
    return q_new

# Функція 4: Застосування граничних умов
def apply_boundary_conditions(q):
    q_new = q.copy()
    q_new[:, -1] = config.q_top
    q_new[:, 0] = q_new[:, 1]
    q_new[0, :] = q_new[1, :]
    q_new[-1, :] = q_new[-2, :]
    return q_new

# Функція 5: Перевірка умов зупинки
def check_stopping_condition(q, time):
    min_q = np.min(q)
    if np.isnan(min_q):
        print(f"Знайдено nan на часі {time / (3600 * 24):.2f} днів. Зупиняємо моделювання.")
        return True
    if np.all(q >= config.q_threshold):
        print(f"Увесь зразок забруднений (> {config.q_threshold} кг/м³) через {time / (3600 * 24):.2f} днів")
        return True
    if time % 10000 == 0:
        print(f"Час: {time / (3600 * 24):.2f} днів, Мінімальна концентрація: {min_q:.4f} кг/м³")
    return False

# Функція 6: Візуалізація з межами областей і збереженням
def visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, title="Концентрація забрудника"):
    plt.figure(figsize=(8, 6))
    plt.contourf(X.T, Z.T, q, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(label='Концентрація (кг/м³)')
    plt.plot(X, Z, 'k-', linewidth=0.2, alpha=0.3)
    plt.plot(X.T, Z.T, 'k-', linewidth=0.2, alpha=0.3)
    
    # Позначення меж каналу (конус)
    plt.plot(channel_left, Z[0, :], color='red', linestyle='--', linewidth=1.5, label='Межі каналу')
    plt.plot(channel_right, Z[0, :], color='red', linestyle='--', linewidth=1.5)
    
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
        # print("loop")
        q_new = update_concentration(q, v_z)
        q_new = apply_boundary_conditions(q_new)
        q = q_new
        time += config.dt
        
        for snapshot_time in snapshot_times:
            if not snapshots_taken[snapshot_time] and abs(time - snapshot_time) < config.dt:
                visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, title="Концентрація забрудника (проміжний етап)")
                snapshots_taken[snapshot_time] = True
        
        if check_stopping_condition(q, time):
            break
    
    visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, title="Концентрація забрудника (кінцевий результат)")

if __name__ == "__main__":
    run_simulation()