import numpy as np


def lemur_function_1(d, u, B, B_x, B_y, B_z, H, H_x, H_y, H_z, k0):
    kk = 2.0
    sigma_0 = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    N = len(d)
    K_AIR_A = np.sqrt(k0 ** 2 - kk * H + 0j)
    K_AIR_B = np.sqrt(k0 ** 2 + kk * H + 0j)
    k_air_term1 = 0.5 * (K_AIR_A + K_AIR_B) * sigma_0
    k_air_term2 = (sigma_x * H_x + sigma_y * H_y + sigma_z * H_z) / (2 * max(H, 1e-20)) * (K_AIR_A - K_AIR_B)
    k_air = k_air_term1 + k_air_term2

    r11 = np.zeros(N, dtype=complex)
    r12 = np.zeros(N, dtype=complex)
    r21 = np.zeros(N, dtype=complex)
    r22 = np.zeros(N, dtype=complex)
    t11 = np.zeros(N, dtype=complex)
    t12 = np.zeros(N, dtype=complex)
    t21 = np.zeros(N, dtype=complex)
    t22 = np.zeros(N, dtype=complex)

    for j in range(N):
        B_val = max(B[j], 1e-20)
        K_A = np.sqrt(k0 ** 2 - u[j] - kk * B[j] + 0j)
        K_B = np.sqrt(k0 ** 2 - u[j] + kk * B[j] + 0j)
        k_term1 = 0.5 * (K_A + K_B) * sigma_0
        k_term2 = (sigma_x * B_x[j] + sigma_y * B_y[j] + sigma_z * B_z[j]) / (2 * B_val) * (K_A - K_B)
        k = k_term1 + k_term2

        inv_k_air_k = np.linalg.inv(k_air + k)
        r0 = inv_k_air_k @ (k_air - k)
        t0 = inv_k_air_k @ (2 * k_air)
        t00 = inv_k_air_k @ (2 * k)

        EXP_A = np.exp(1j * K_A * d[j])
        EXP_B = np.exp(1j * K_B * d[j])
        EXP_term1 = 0.5 * (EXP_A + EXP_B) * sigma_0
        EXP_term2 = (sigma_x * B_x[j] + sigma_y * B_y[j] + sigma_z * B_z[j]) / (2 * B_val) * (EXP_A - EXP_B)
        EXP = EXP_term1 + EXP_term2

        I_minus = sigma_0 - EXP @ r0 @ EXP @ r0
        inv_I_minus = np.linalg.inv(I_minus)
        r_matrix = r0 - t00 @ EXP @ r0 @ inv_I_minus @ EXP @ t0
        t_matrix = t00 @ inv_I_minus @ EXP @ t0

        r11[j] = r_matrix[0, 0]
        r12[j] = r_matrix[0, 1]
        r21[j] = r_matrix[1, 0]
        r22[j] = r_matrix[1, 1]
        t11[j] = t_matrix[0, 0]
        t12[j] = t_matrix[0, 1]
        t21[j] = t_matrix[1, 0]
        t22[j] = t_matrix[1, 1]

    B_sub = max(B[N], 1e-10)
    K_A = np.sqrt(k0 ** 2 - u[N] - kk * B[N] + 0j)
    K_B = np.sqrt(k0 ** 2 - u[N] + kk * B[N] + 0j)
    k_term1 = 0.5 * (K_A + K_B) * sigma_0
    k_term2 = (sigma_x * B_x[N] + sigma_y * B_y[N] + sigma_z * B_z[N]) / (2 * B_sub) * (K_A - K_B)
    k_sub = k_term1 + k_term2

    inv_k_air_k_sub = np.linalg.inv(k_air + k_sub)
    r_matrix = inv_k_air_k_sub @ (k_air - k_sub)
    t_matrix = inv_k_air_k_sub @ (2 * k_air)

    R11 = [r_matrix[0, 0]]
    R12 = [r_matrix[0, 1]]
    R21 = [r_matrix[1, 0]]
    R22 = [r_matrix[1, 1]]
    T11 = [t_matrix[0, 0]]
    T12 = [t_matrix[0, 1]]
    T21 = [t_matrix[1, 0]]
    T22 = [t_matrix[1, 1]]

    for j in range(1, N + 1):
        R_prev = np.array([[R11[j - 1], R12[j - 1]],
                           [R21[j - 1], R22[j - 1]]])
        r_now = np.array([[r11[N - j], r12[N - j]],
                          [r21[N - j], r22[N - j]]])
        t_now = np.array([[t11[N - j], t12[N - j]],
                          [t21[N - j], t22[N - j]]])

        I_minus = sigma_0 - r_now @ R_prev
        inv_I_minus = np.linalg.inv(I_minus)
        R_matrix = r_now + t_now @ R_prev @ inv_I_minus @ t_now
        T_matrix = np.array([[T11[j - 1], T12[j - 1]],
                             [T21[j - 1], T22[j - 1]]]) @ inv_I_minus @ t_now

        R11.append(R_matrix[0, 0])
        R12.append(R_matrix[0, 1])
        R21.append(R_matrix[1, 0])
        R22.append(R_matrix[1, 1])
        T11.append(T_matrix[0, 0])
        T12.append(T_matrix[0, 1])
        T21.append(T_matrix[1, 0])
        T22.append(T_matrix[1, 1])

    R = np.array([[R11[N], R12[N]], [R21[N], R22[N]]])
    real_K_AIR_A = np.real(K_AIR_A)
    real_K_AIR_B = np.real(K_AIR_B)
    real_K_A = np.real(K_A)
    real_K_B = np.real(K_B)

    Phi11 = real_K_AIR_A * np.abs(R[0, 0]) ** 2 / real_K_AIR_A
    Phi12 = real_K_AIR_A * np.abs(R[0, 1]) ** 2 / real_K_AIR_B
    Phi21 = real_K_AIR_B * np.abs(R[1, 0]) ** 2 / real_K_AIR_A
    Phi22 = real_K_AIR_B * np.abs(R[1, 1]) ** 2 / real_K_AIR_B
    Reflectivity = np.array([[Phi11, Phi12], [Phi21, Phi22]])

    return np.array([
        Reflectivity[0, 0], Reflectivity[0, 1],
        Reflectivity[1, 0], Reflectivity[1, 1],
        0, 0, 0, 0, 0, 0  # Placeholders for other values
    ])