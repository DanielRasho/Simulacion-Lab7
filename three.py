#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special as spc
from scipy.stats import chi2, norm
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

np.random.seed(42)


class LCG:
    """
    Linear Congruential Generator
    X_{n+1} = (a * X_n + c) mod m
    """

    def __init__(self, seed, a, c, m):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        self.current = seed

    def next(self):
        """Genera el siguiente número en la secuencia"""
        self.current = (self.a * self.current + self.c) % self.m
        return self.current

    def generate_sample(self, n):
        sample = []
        for _ in range(n):
            sample.append(self.next())
        return np.array(sample)

    def generate_uniform(self, n):
        """Genera una muestra uniforme en [0, 1)"""
        integers = self.generate_sample(n)
        return integers / self.m

    def reset(self):
        self.current = self.seed


class MersenneTwister:
    def __init__(self, seed=5489):
        # Parámetros del MT19937
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253

        # Estado interno
        self.MT = [0] * self.n
        self.index = self.n + 1
        self.lower_mask = (1 << self.r) - 1
        self.upper_mask = (~self.lower_mask) & 0xFFFFFFFF

        # Inicializar con semilla
        self.seed_mt(seed)

    def seed_mt(self, seed):
        """Inicializa el generador con una semilla"""
        self.index = self.n
        self.MT[0] = seed & 0xFFFFFFFF
        for i in range(1, self.n):
            self.MT[i] = (
                self.f * (self.MT[i - 1] ^ (self.MT[i - 1] >> (self.w - 2))) + i
            ) & 0xFFFFFFFF

    def twist(self):
        """Genera los siguientes n valores del estado"""
        for i in range(self.n):
            x = (self.MT[i] & self.upper_mask) + (
                self.MT[(i + 1) % self.n] & self.lower_mask
            )
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA
        self.index = 0

    def extract_number(self):
        """Extrae un número aleatorio del estado"""
        if self.index >= self.n:
            self.twist()

        y = self.MT[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= y >> self.l

        self.index += 1
        return y & 0xFFFFFFFF

    def random(self):
        """Genera un número aleatorio en [0, 1)"""
        return self.extract_number() / (2**32)

    def generate_uniform(self, n):
        """Genera una muestra uniforme de tamaño n"""
        return np.array([self.random() for _ in range(n)])


m1 = 2**31 - 1  # Número primo de Mersenne
a1 = 7**5  # 16807
c1 = 0  # Generador multiplicativo
seed1 = 123456789
N = 1_000_000

lcg1 = LCG(seed1, a1, c1, m1)
sample_uniform_lcg1 = lcg1.generate_uniform(N)

mt = MersenneTwister(seed=42)
sample_uniform_mt = mt.generate_uniform(N)


class NISTTests:
    """Implementación de pruebas NIST SP 800-22"""

    @staticmethod
    def uniform_to_bits(uniform_sample, threshold=0.5):
        """Convierte muestra uniforme [0,1) a bits"""
        return (uniform_sample >= threshold).astype(int)

    @staticmethod
    def frequency_test(bits):
        """Test de Frecuencia (Monobit)"""
        n = len(bits)
        s = np.sum(2 * bits - 1)  # Convierte 0,1 a -1,1
        s_obs = np.abs(s) / np.sqrt(n)
        p_value = spc.erfc(s_obs / np.sqrt(2))
        return p_value

    @staticmethod
    def block_frequency_test(bits, block_size=128):
        """Test de Frecuencia dentro de un Bloque"""
        n = len(bits)
        num_blocks = n // block_size
        if num_blocks == 0:
            return 0.0

        block_bits = bits[: num_blocks * block_size].reshape((num_blocks, block_size))
        proportions = np.mean(block_bits, axis=1)
        chi_squared = 4 * block_size * np.sum((proportions - 0.5) ** 2)
        p_value = spc.gammaincc(num_blocks / 2, chi_squared / 2)
        return p_value

    @staticmethod
    def runs_test(bits):
        """Test de Rachas (Runs)"""
        n = len(bits)
        pi = np.mean(bits)

        # Pre-test: verificar que la proporción está cerca de 0.5
        tau = 2 / np.sqrt(n)
        if np.abs(pi - 0.5) >= tau:
            return 0.0

        # Contar rachas
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                runs += 1

        p_value = spc.erfc(
            np.abs(runs - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi))
        )
        return p_value

    @staticmethod
    def longest_run_test(bits):
        """Test de la Racha Más Larga de Unos"""
        n = len(bits)

        if n < 128:
            return 0.0
        elif n < 6272:
            M, K = 8, 3
            v_values = [1, 2, 3, 4]
            pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
        elif n < 750000:
            M, K = 128, 5
            v_values = [4, 5, 6, 7, 8, 9]
            pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        else:
            M, K = 10000, 6
            v_values = [10, 11, 12, 13, 14, 15, 16]
            pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

        N = n // M
        blocks = bits[: N * M].reshape((N, M))

        # Encontrar la racha más larga en cada bloque
        longest_runs = []
        for block in blocks:
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            longest_runs.append(max_run)

        # Contar frecuencias
        v = np.zeros(len(v_values) + 1)
        for run in longest_runs:
            if run <= v_values[0]:
                v[0] += 1
            elif run >= v_values[-1]:
                v[-1] += 1
            else:
                for i in range(len(v_values) - 1):
                    if v_values[i] < run <= v_values[i + 1]:
                        v[i + 1] += 1
                        break

        chi_squared = np.sum(
            (v - N * np.array(pi_values)) ** 2 / (N * np.array(pi_values))
        )
        p_value = spc.gammaincc((len(pi_values) - 1) / 2, chi_squared / 2)
        return p_value

    @staticmethod
    def spectral_test(bits):
        """Test Espectral (DFT)"""
        n = len(bits)
        x = 2 * bits - 1  # Convierte a +1, -1

        # Aplicar FFT
        s = np.fft.fft(x)
        modulus = np.abs(s[: n // 2])

        # Threshold
        tau = np.sqrt(np.log(1 / 0.05) * n)
        n0 = 0.95 * n / 2
        n1 = np.sum(modulus < tau)

        d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4)
        p_value = spc.erfc(np.abs(d) / np.sqrt(2))
        return p_value

    @staticmethod
    def non_overlapping_template_test(bits, template=None, block_size=968):
        """Test de Coincidencia de Plantilla No Superpuesta"""
        if template is None:
            template = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

        n = len(bits)
        m = len(template)
        N = n // block_size

        if N == 0:
            return 0.0

        mu = (block_size - m + 1) / (2**m)
        sigma_squared = block_size * ((1 / (2**m)) - ((2 * m - 1) / (2 ** (2 * m))))

        w = np.zeros(N)
        for i in range(N):
            block = bits[i * block_size : (i + 1) * block_size]
            count = 0
            j = 0
            while j <= block_size - m:
                if np.array_equal(block[j : j + m], template):
                    count += 1
                    j += m  # No superpuesto
                else:
                    j += 1
            w[i] = count

        chi_squared = np.sum((w - mu) ** 2 / sigma_squared)
        p_value = spc.gammaincc(N / 2, chi_squared / 2)
        return p_value

    @staticmethod
    def serial_test(bits, m=16):
        """Test Serial"""
        n = len(bits)

        def psi_squared(m_param, bits_param):
            if m_param == 0:
                return 0
            n_param = len(bits_param)
            counts = np.zeros(2**m_param)
            for i in range(n_param):
                pattern = 0
                for j in range(m_param):
                    pattern = (pattern << 1) | bits_param[(i + j) % n_param]
                counts[pattern] += 1
            psi = (2**m_param / n_param) * np.sum(counts**2) - n_param
            return psi

        psi2_m = psi_squared(m, bits)
        psi2_m1 = psi_squared(m - 1, bits)
        psi2_m2 = psi_squared(m - 2, bits)

        delta1 = psi2_m - psi2_m1
        delta2 = psi2_m - 2 * psi2_m1 + psi2_m2

        p_value1 = spc.gammaincc(2 ** (m - 2), delta1 / 2)
        p_value2 = spc.gammaincc(2 ** (m - 3), delta2 / 2)

        return min(p_value1, p_value2)

    @staticmethod
    def approximate_entropy_test(bits, m=10):
        """Test de Entropía Aproximada"""
        n = len(bits)

        def phi(m_param):
            counts = {}
            for i in range(n):
                pattern = tuple(bits[(i + j) % n] for j in range(m_param))
                counts[pattern] = counts.get(pattern, 0) + 1

            phi_m = 0
            for count in counts.values():
                if count > 0:
                    phi_m += count * np.log(count / n)
            return phi_m / n

        phi_m = phi(m)
        phi_m1 = phi(m + 1)
        apen = phi_m - phi_m1

        chi_squared = 2 * n * (np.log(2) - apen)
        p_value = spc.gammaincc(2 ** (m - 1), chi_squared / 2)
        return p_value

    @staticmethod
    def cumulative_sums_test(bits):
        """Test de Sumas Acumulativas"""
        n = len(bits)
        x = 2 * bits - 1
        s = np.cumsum(x)

        # Forward
        z_forward = np.max(np.abs(s))
        # Backward
        z_backward = np.max(np.abs(np.cumsum(x[::-1])))

        def p_value_calc(z):
            sum_a = 0
            start = int((-n / z + 1) / 4)
            end = int((n / z - 1) / 4) + 1
            for k in range(start, end):
                sum_a += norm.cdf((4 * k + 1) * z / np.sqrt(n)) - norm.cdf(
                    (4 * k - 1) * z / np.sqrt(n)
                )

            sum_b = 0
            start = int((-n / z - 3) / 4)
            end = int((n / z - 1) / 4) + 1
            for k in range(start, end):
                sum_b += norm.cdf((4 * k + 3) * z / np.sqrt(n)) - norm.cdf(
                    (4 * k + 1) * z / np.sqrt(n)
                )

            return 1 - sum_a + sum_b

        p_forward = p_value_calc(z_forward)
        p_backward = p_value_calc(z_backward)

        return min(p_forward, p_backward)

    @staticmethod
    def linear_complexity_test(bits, block_size=500):
        """Test de Complejidad Lineal"""
        n = len(bits)
        K = n // block_size

        if K < 1:
            return 0.0

        mu = (
            block_size / 2
            + (9 + (-1) ** (block_size + 1)) / 36
            - 1 / (2**block_size) * (block_size / 3 + 2 / 9)
        )

        t = [(-1) ** block_size * (i - mu) + 2 / 9 for i in range(7)]
        v = [0, 0, 0, 0, 0, 0, 0]

        for i in range(K):
            block = bits[i * block_size : (i + 1) * block_size]
            L = NISTTests.berlekamp_massey(block)
            t_i = (-1) ** block_size * (L - mu) + 2 / 9

            if t_i <= t[0]:
                v[0] += 1
            elif t_i <= t[1]:
                v[1] += 1
            elif t_i <= t[2]:
                v[2] += 1
            elif t_i <= t[3]:
                v[3] += 1
            elif t_i <= t[4]:
                v[4] += 1
            elif t_i <= t[5]:
                v[5] += 1
            else:
                v[6] += 1

        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        chi_squared = np.sum([(v[i] - K * pi[i]) ** 2 / (K * pi[i]) for i in range(7)])
        p_value = spc.gammaincc(3, chi_squared / 2)
        return p_value

    @staticmethod
    def berlekamp_massey(bits):
        """Algoritmo de Berlekamp-Massey para complejidad lineal"""
        n = len(bits)
        b = np.zeros(n, dtype=int)
        c = np.zeros(n, dtype=int)
        b[0] = 1
        c[0] = 1

        L = 0
        m = -1
        N = 0

        while N < n:
            d = bits[N]
            for i in range(1, L + 1):
                d ^= c[i] & bits[N - i]

            if d == 1:
                t = c.copy()
                for i in range(n - N + m):
                    c[N - m + i] ^= b[i]

                if L <= N / 2:
                    L = N + 1 - L
                    m = N
                    b = t
            N += 1

        return L


def run_nist_battery(uniform_sample, generator_name="Generator"):
    """Ejecuta batería completa de tests NIST"""
    bits = NISTTests.uniform_to_bits(uniform_sample)

    tests = {
        "Frequency (Monobit)": NISTTests.frequency_test,
        "Block Frequency": NISTTests.block_frequency_test,
        "Runs": NISTTests.runs_test,
        "Longest Run": NISTTests.longest_run_test,
        "Spectral (DFT)": NISTTests.spectral_test,
        "Non-Overlapping Template": NISTTests.non_overlapping_template_test,
        "Serial": NISTTests.serial_test,
        "Approximate Entropy": NISTTests.approximate_entropy_test,
        "Cumulative Sums": NISTTests.cumulative_sums_test,
        "Linear Complexity": NISTTests.linear_complexity_test,
    }

    results = {}
    alpha = 0.01  # Nivel de significancia

    print(f"\n{'=' * 60}")
    print(f"Pruebas NIST SP 800-22 para {generator_name}")
    print(f"{'=' * 60}")
    print(f"Tamaño de muestra: {len(uniform_sample):,}")
    print(f"Número de bits: {len(bits):,}")
    print(f"Nivel de significancia (α): {alpha}")
    print(f"{'=' * 60}\n")

    for test_name, test_func in tests.items():
        try:
            p_value = test_func(bits)
            passed = p_value >= alpha
            results[test_name] = {"p_value": p_value, "passed": passed}
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:30s}: p-value = {p_value:.6f} {status}")
        except Exception as e:
            results[test_name] = {"p_value": 0.0, "passed": False}
            print(f"{test_name:30s}: ERROR - {str(e)}")

    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    print(f"\n{'=' * 60}")
    print(
        f"Resumen: {passed_count}/{total_count} pruebas pasadas ({100 * passed_count / total_count:.1f}%)"
    )
    print(f"{'=' * 60}\n")

    return results


# Generar muestras
m1 = 2**31 - 1
a1 = 7**5
c1 = 0
seed1 = 123456789
N = 100_000  # Reducido para velocidad

print("Generando muestras...")
lcg1 = LCG(seed1, a1, c1, m1)
sample_uniform_lcg1 = lcg1.generate_uniform(N)

mt = MersenneTwister(seed=42)
sample_uniform_mt = mt.generate_uniform(N)

# Ejecutar pruebas
results_lcg = run_nist_battery(
    sample_uniform_lcg1, "Linear Congruential Generator (LCG)"
)
results_mt = run_nist_battery(sample_uniform_mt, "Mersenne Twister (MT19937)")

# Crear tabla comparativa
comparison_data = []
for test_name in results_lcg.keys():
    comparison_data.append(
        {
            "Test": test_name,
            "LCG p-value": results_lcg[test_name]["p_value"],
            "LCG Pass": "✓" if results_lcg[test_name]["passed"] else "✗",
            "MT p-value": results_mt[test_name]["p_value"],
            "MT Pass": "✓" if results_mt[test_name]["passed"] else "✗",
        }
    )

df = pd.DataFrame(comparison_data)

# Visualizar tabla
print("\n" + "=" * 80)
print("TABLA COMPARATIVA DE RESULTADOS")
print("=" * 80)
print(df.to_string(index=False))
print("=" * 80)

# Resumen estadístico
lcg_passed = sum(1 for r in results_lcg.values() if r["passed"])
mt_passed = sum(1 for r in results_mt.values() if r["passed"])
total_tests = len(results_lcg)

print(f"\nRESUMEN:")
print(
    f"  LCG:            {lcg_passed}/{total_tests} pruebas pasadas ({100 * lcg_passed / total_tests:.1f}%)"
)
print(
    f"  Mersenne Twister: {mt_passed}/{total_tests} pruebas pasadas ({100 * mt_passed / total_tests:.1f}%)"
)

print(f"\n{'=' * 80}")
print("CONCLUSIÓN:")
if mt_passed > lcg_passed:
    print(
        f"El Mersenne Twister es SUPERIOR al LCG, pasando {mt_passed - lcg_passed} prueba(s) adicional(es)."
    )
    print("El MT19937 muestra mejor calidad criptográfica y aleatoriedad estadística.")
elif lcg_passed > mt_passed:
    print(
        f"El LCG es SUPERIOR al Mersenne Twister, pasando {lcg_passed - mt_passed} prueba(s) adicional(es)."
    )
else:
    print("Ambos generadores muestran desempeño similar en las pruebas NIST.")
print(f"{'=' * 80}\n")

# Gráfico de comparación
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de barras de p-values
test_names = [t[:20] for t in df["Test"]]
x = np.arange(len(test_names))
width = 0.35

ax1.barh(x - width / 2, df["LCG p-value"], width, label="LCG", alpha=0.8)
ax1.barh(x + width / 2, df["MT p-value"], width, label="Mersenne Twister", alpha=0.8)
ax1.axvline(x=0.01, color="r", linestyle="--", linewidth=2, label="α = 0.01")
ax1.set_yticks(x)
ax1.set_yticklabels(test_names)
ax1.set_xlabel("p-value")
ax1.set_title("Comparación de p-values en Tests NIST")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico de resumen
categories = ["LCG", "Mersenne Twister"]
passed = [lcg_passed, mt_passed]
failed = [total_tests - lcg_passed, total_tests - mt_passed]

ax2.bar(categories, passed, label="Pasadas", color="green", alpha=0.7)
ax2.bar(categories, failed, bottom=passed, label="Falladas", color="red", alpha=0.7)
ax2.set_ylabel("Número de Pruebas")
ax2.set_title("Resumen de Pruebas NIST")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Añadir porcentajes
for i, (p, f) in enumerate(zip(passed, failed)):
    total = p + f
    percentage = 100 * p / total
    ax2.text(
        i,
        total / 2,
        f"{percentage:.1f}%",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
    )

plt.tight_layout()
plt.savefig("nist_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Gráfico guardado como 'nist_comparison.png'")
