import numpy as np
import matplotlib.pyplot as plt
import config
import os

# Створюємо папку для збереження результатів
if not os.path.exists("results"):
    os.makedirs("results")

plt.ion()

# Ініціалізація сітки
def initialize_grid():
    x = np.linspace(-4, 4, config.nx)
    z = np.linspace(0, 6, config.nz)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z

# Визначення геометрії каналу та ініціалізація умов
def initialize_conditions(x, z):
    channel_left = np.where(z >= 3, -0.1, -0.1 - (2 - 0.1) / 3 * (3 - z))
    channel_right = np.where(z >= 3, 0.1, 0.1 + (2 - 0.1) / 3 * (3 - z))

    channel_mask = np.zeros((config.nx, config.nz), dtype=bool)
    for j in range(config.nz):
        x_left = channel_left[j]
        x_right = channel_right[j]
        channel_mask[:, j] = (x >= x_left) & (x <= x_right)

    soil_mask = ~channel_mask

    q = np.zeros((config.nx, config.nz))
    q[:, -1] = config.q_top  # Початкова концентрація на поверхні
    return q, channel_left, channel_right, channel_mask, soil_mask

# Візуалізація з межами областей і збереженням
def visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, title="Концентрація забрудника", save=True):
    plt.clf()
    # Фіксуємо vmin=0 (нульова концентрація) і vmax=0.01 (максимальна концентрація з граничних умов)
    plt.contourf(X.T, Z.T, q, levels=50, cmap='jet', alpha=0.8, vmin=0, vmax=0.01)
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

# Перевірка умови зупинки
def check_stopping_condition(q, time):
    if np.all(q >= config.q_threshold):
        print(f"Увесь зразок забруднений (> {config.q_threshold} кг/м³) через {time / (3600 * 24):.2f} днів")
        return True
    return False

# Обчислення нового розподілу концентрації
def compute_diffusion(q, soil_mask, channel_mask, time):
    q_new = q.copy()
    
    # Коефіцієнт дифузії та розміри кроку
    D = config.D
    dx2 = config.dx**2
    dz2 = config.dz**2
    dt = config.dt
    
    # Стабільність (умова КФЛ для дифузії: D * dt / dx^2 <= 0.5)
    alpha_x = D * dt / dx2
    alpha_z = D * dt / dz2
    if max(alpha_x, alpha_z) > 0.5:
        raise ValueError("Часовий крок dt занадто великий для стабільності дифузії!")
    
    # Визначаємо глибину заповнення каналу
    z = np.linspace(0, 6, config.nz)
    z_max = 6.0
    v0 = abs(config.v_z_channel)  # 0.02 м/с
    w0 = 0.2  # Базова ширина при z >= 3
    
    # Розрахунок source_depth з урахуванням ширини
    source_depth = 0.0
    if time >= 1.0:
        t = time - 1.0  # Починаємо з 1 секунди
        # Верхня частина (z від 6 до 3): постійна швидкість
        if t * v0 <= 3.0:  # 3 м за 150 с при v0 = 0.02 м/с
            source_depth = t * v0
        else:
            # Конічна частина (z від 3 до 0)
            t_cone = t - 3.0 / v0  # Час у конічній частині
            # Ширина: w(z) = 0.2 + 1.2667 * (3 - z)
            # v(z) = v0 * w0 / w(z), інтегруємо аналітично
            # dz/dt = -v0 * w0 / (0.2 + 1.2667 * (3 - z))
            # Інтеграл: source_depth = 3 + решта від конусу
            if t_cone > 0:
                # Спрощуємо: чисельно наближаємо
                z_remaining = 3.0
                while t_cone > 0 and z_remaining > 0:
                    w = 0.2 + 1.2667 * (3 - z_remaining)
                    v_z = v0 * w0 / w
                    dz = v_z * min(t_cone, dt)  # Невеликий крок для точності
                    z_remaining -= dz
                    t_cone -= min(t_cone, dt)
                source_depth = 3.0 + (3.0 - z_remaining)
    
    # Заповнюємо канал концентрацією q = 0.01 залежно від глибини
    for i in range(config.nx):
        for j in range(config.nz):
            if channel_mask[i, j] and (z_max - z[j]) <= source_depth:
                q_new[i, j] = 0.01
    
    # Дифузія в ґрунті
    for i in range(1, config.nx - 1):
        for j in range(1, config.nz - 1):
            if soil_mask[i, j]:
                q_new[i, j] = q[i, j] + D * dt * (
                    (q[i+1, j] - 2 * q[i, j] + q[i-1, j]) / dx2 +  # Дифузія по x
                    (q[i, j+1] - 2 * q[i, j] + q[i, j-1]) / dz2     # Дифузія по z
                )
    
    # Граничні умови
    q_new[:, -1] = config.q_top
    q_new[:, 0] = q_new[:, 1]
    q_new[0, :] = q_new[1, :]
    q_new[-1, :] = q_new[-2, :]
    
    return q_new

# Основна функція моделювання
def run_simulation():
    x, z, X, Z = initialize_grid()
    q, channel_left, channel_right, channel_mask, soil_mask = initialize_conditions(x, z)
    
    snapshot_times = [1 * 24 * 3600, 10 * 24 * 3600]
    snapshots_taken = {t: False for t in snapshot_times}
    
    time = 0
    visualization_interval = 5000
    
    while time < config.max_time:
        q_new = compute_diffusion(q, soil_mask, channel_mask, time)
        q = q_new
        time += config.dt
        
        # Перевірка умови зупинки перед проміжною візуалізацією
        if check_stopping_condition(q, time):
            # Зберігаємо кінцевий результат одразу після виконання умови
            visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, 
                                      title="Концентрація забрудника (кінцевий результат)", save=True)
            break
        
        # Проміжна візуалізація
        if time % visualization_interval < config.dt:
            visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, save=False)
        
        # Знімки у визначені моменти
        for snapshot_time in snapshot_times:
            if not snapshots_taken[snapshot_time] and abs(time - snapshot_time) < config.dt:
                visualize_results_with_grid(X, Z, q, time, channel_left, channel_right, z, save=True)
                snapshots_taken[snapshot_time] = True
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()