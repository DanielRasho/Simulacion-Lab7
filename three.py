#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special as spc
from scipy.stats import chi2, norm
import pandas as pd
from typing import Tuple

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

np.random.seed(42)


class BitGeneratorLCG:
    """
    Generador de bits aleatorios usando Linear Congruential Generator
    Extrae bits del número generado para mayor eficiencia
    """

    def __init__(self, seed, a=None, c=None, m=None):
        """
        Inicializa el generador LCG

        Parámetros:
        - seed: Semilla inicial
        - a: Multiplicador (por defecto: 7^5 = 16807)
        - c: Incremento (por defecto: 0, generador multiplicativo)
        - m: Módulo (por defecto: 2^31 - 1, primo de Mersenne)
        """
        self.seed = seed
        self.a = a if a is not None else 7**5  # 16807
        self.c = c if c is not None else 0
        self.m = m if m is not None else 2**31 - 1
        self.current = seed
        self.bits_per_number = 31  # Bits utilizables del módulo

        print(f"LCG inicializado:")
        print(f"  a = {self.a}")
        print(f"  c = {self.c}")
        print(f"  m = {self.m}")
        print(f"  seed = {self.seed}")
        print(f"  bits por número = {self.bits_per_number}")

    def next_number(self):
        """Genera el siguiente número en la secuencia LCG"""
        self.current = (self.a * self.current + self.c) % self.m
        return self.current

    def generate_bits(self, n):
        """
        Genera N bits aleatorios

        Optimización: extrae múltiples bits de cada número generado
        """
        bits = []
        bits_needed = n

        while bits_needed > 0:
            number = self.next_number()

            # Extraer bits del número (de LSB a MSB)
            bits_to_extract = min(bits_needed, self.bits_per_number)
            for i in range(bits_to_extract):
                bits.append((number >> i) & 1)

            bits_needed -= bits_to_extract

        return np.array(bits[:n], dtype=int)

    def generate_bits_simple(self, n):
        """
        Versión simple: genera un número y toma el bit menos significativo
        (Menos eficiente pero más directo)
        """
        bits = []
        for _ in range(n):
            number = self.next_number()
            bits.append(number & 1)  # Bit menos significativo
        return np.array(bits, dtype=int)

    def reset(self):
        """Reinicia el generador a su estado inicial"""
        self.current = self.seed


class BitGeneratorMT:
    """
    Generador de bits aleatorios usando Mersenne Twister (MT19937)
    Extrae bits de los números generados
    """

    def __init__(self, seed=5489):
        """Inicializa el Mersenne Twister"""
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
        self.initial_seed = seed
        self.bits_per_number = 32

        # Inicializar con semilla
        self.seed_mt(seed)

        print(f"\nMersenne Twister inicializado:")
        print(f"  Algoritmo: MT19937")
        print(f"  Período: 2^19937 - 1")
        print(f"  seed = {seed}")
        print(f"  bits por número = {self.bits_per_number}")

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

    def generate_bits(self, n):
        """
        Genera N bits aleatorios

        Optimización: extrae 32 bits de cada número generado
        """
        bits = []
        bits_needed = n

        while bits_needed > 0:
            number = self.extract_number()

            # Extraer bits del número (de LSB a MSB)
            bits_to_extract = min(bits_needed, self.bits_per_number)
            for i in range(bits_to_extract):
                bits.append((number >> i) & 1)

            bits_needed -= bits_to_extract

        return np.array(bits[:n], dtype=int)

    def generate_bits_simple(self, n):
        """
        Versión simple: genera un número y toma el bit menos significativo
        """
        bits = []
        for _ in range(n):
            number = self.extract_number()
            bits.append(number & 1)
        return np.array(bits, dtype=int)

    def reset(self):
        """Reinicia el generador a su estado inicial"""
        self.seed_mt(self.initial_seed)


class NISTTests:
    """
    Implementación de los 15 tests estadísticos de NIST SP 800-22
    para evaluar la aleatoriedad de secuencias de bits
    """

    def __init__(self, significance_level=0.01):
        """
        Args:
            significance_level: Nivel de significancia (típicamente 0.01)
        """
        self.alpha = significance_level

    def monobit_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        Test 1: Frequency (Monobit) Test
        Verifica si la proporción de 0s y 1s es aproximadamente 1:1
        """
        n = len(bits)
        s = np.sum(2 * bits - 1)  # Convierte 0->-1, 1->1
        s_obs = np.abs(s) / np.sqrt(n)
        p_value = spc.erfc(s_obs / np.sqrt(2))
        return p_value, p_value >= self.alpha

    def frequency_within_block_test(
        self, bits: np.ndarray, M=128
    ) -> Tuple[float, bool]:
        """
        Test 2: Frequency Test within a Block
        Verifica la frecuencia de 1s dentro de bloques de M bits
        """
        n = len(bits)
        N = n // M

        if N < 1:
            return 0.0, False

        blocks = bits[: N * M].reshape(N, M)
        proportions = np.mean(blocks, axis=1)
        chi_squared = 4 * M * np.sum((proportions - 0.5) ** 2)
        p_value = spc.gammaincc(N / 2, chi_squared / 2)

        return p_value, p_value >= self.alpha

    def runs_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        Test 3: Runs Test
        Verifica la transición entre 0s y 1s
        """
        n = len(bits)
        pi = np.mean(bits)

        # Pre-test: verificar que pi esté cerca de 0.5
        tau = 2 / np.sqrt(n)
        if np.abs(pi - 0.5) >= tau:
            return 0.0, False

        # Contar runs (secuencias de bits iguales consecutivos)
        runs = 1 + np.sum(bits[:-1] != bits[1:])

        p_value = spc.erfc(
            np.abs(runs - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi))
        )

        return p_value, p_value >= self.alpha

    def longest_run_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        Test 4: Test for the Longest Run of Ones in a Block
        """
        n = len(bits)

        if n < 128:
            return 0.0, False
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
        blocks = bits[: N * M].reshape(N, M)

        # Encontrar el run más largo de 1s en cada bloque
        v = np.zeros(len(v_values))
        for block in blocks:
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0

            for i, threshold in enumerate(v_values):
                if i == len(v_values) - 1:
                    if max_run >= threshold:
                        v[i] += 1
                        break
                elif max_run == threshold:
                    v[i] += 1
                    break

        chi_squared = np.sum(
            (v - N * np.array(pi_values)) ** 2 / (N * np.array(pi_values))
        )
        p_value = spc.gammaincc(K / 2, chi_squared / 2)

        return p_value, p_value >= self.alpha

    def binary_matrix_rank_test(
        self, bits: np.ndarray, M=32, Q=32
    ) -> Tuple[float, bool]:
        """
        Test 5: Binary Matrix Rank Test
        """
        n = len(bits)
        N = n // (M * Q)

        if N == 0:
            return 0.0, False

        # Crear matrices y calcular rangos
        fm = 0  # Full rank (M)
        fm1 = 0  # Rank M-1

        for i in range(N):
            block = bits[i * M * Q : (i + 1) * M * Q].reshape(M, Q)
            rank = np.linalg.matrix_rank(block)
            if rank == M:
                fm += 1
            elif rank == M - 1:
                fm1 += 1

        chi_squared = (
            (fm - 0.2888 * N) ** 2 / (0.2888 * N)
            + (fm1 - 0.5776 * N) ** 2 / (0.5776 * N)
            + ((N - fm - fm1) - 0.1336 * N) ** 2 / (0.1336 * N)
        )

        p_value = np.exp(-chi_squared / 2)

        return p_value, p_value >= self.alpha

    def dft_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        Test 6: Discrete Fourier Transform (Spectral) Test
        """
        n = len(bits)
        s = 2 * bits - 1  # Convierte 0->-1, 1->1

        # Aplicar FFT
        S = np.fft.fft(s)
        M = np.abs(S[: n // 2])

        # Umbral
        T = np.sqrt(np.log(1 / 0.05) * n)

        # Contar picos que exceden el umbral
        N0 = 0.95 * n / 2
        N1 = len(M[M < T])

        d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
        p_value = spc.erfc(np.abs(d) / np.sqrt(2))

        return p_value, p_value >= self.alpha

    def non_overlapping_template_test(
        self, bits: np.ndarray, m=9, B=None
    ) -> Tuple[float, bool]:
        """
        Test 7: Non-overlapping Template Matching Test
        Usa un template simple para el ejemplo
        """
        n = len(bits)
        M = 1032  # Tamaño de bloque recomendado
        N = n // M

        if N == 0 or m > M:
            return 0.0, False

        # Template simple: secuencia de m unos
        template = np.ones(m, dtype=int)

        # Contar coincidencias en cada bloque
        W = []
        for i in range(N):
            block = bits[i * M : (i + 1) * M]
            count = 0
            j = 0
            while j <= len(block) - m:
                if np.array_equal(block[j : j + m], template):
                    count += 1
                    j += m  # Non-overlapping
                else:
                    j += 1
            W.append(count)

        W = np.array(W)
        mu = (M - m + 1) / (2**m)
        sigma_squared = M * ((1 / (2**m)) - ((2 * m - 1) / (2 ** (2 * m))))

        chi_squared = np.sum((W - mu) ** 2) / sigma_squared
        p_value = spc.gammaincc(N / 2, chi_squared / 2)

        return p_value, p_value >= self.alpha

    def overlapping_template_test(self, bits: np.ndarray, m=9) -> Tuple[float, bool]:
        """
        Test 8: Overlapping Template Matching Test
        """
        n = len(bits)
        M = 1032
        N = n // M

        if N == 0 or m > M:
            return 0.0, False

        # Template: m unos
        template = np.ones(m, dtype=int)

        # Parámetros
        lambda_param = (M - m + 1) / (2**m)
        eta = lambda_param / 2

        # Probabilidades para K=5
        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]

        # Contar coincidencias en cada bloque
        v = np.zeros(6)
        for i in range(N):
            block = bits[i * M : (i + 1) * M]
            count = 0
            for j in range(len(block) - m + 1):
                if np.array_equal(block[j : j + m], template):
                    count += 1

            if count <= 4:
                v[count] += 1
            else:
                v[5] += 1

        chi_squared = np.sum((v - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p_value = spc.gammaincc(5 / 2, chi_squared / 2)

        return p_value, p_value >= self.alpha

    def maurers_universal_test(
        self, bits: np.ndarray, L=7, Q=1280
    ) -> Tuple[float, bool]:
        """
        Test 9: Maurer's Universal Statistical Test
        """
        n = len(bits)
        K = (n // L) - Q

        if K <= 0:
            return 0.0, False

        # Tabla de inicialización
        T = {}

        # Fase de inicialización
        for i in range(Q):
            block = tuple(bits[i * L : (i + 1) * L])
            T[block] = i + 1

        # Fase de test
        sum_log = 0.0
        for i in range(Q, Q + K):
            block = tuple(bits[i * L : (i + 1) * L])
            distance = i + 1 - T.get(block, 0)
            T[block] = i + 1
            if distance > 0:
                sum_log += np.log2(distance)

        fn = sum_log / K

        # Valores esperados (para L=7)
        expected_value = 6.196
        variance = 3.125

        c = 0.7 - 0.8 / L + (4 + 32 / L) * (K ** (-3 / L)) / 15
        sigma = c * np.sqrt(variance / K)

        p_value = spc.erfc(np.abs((fn - expected_value) / (np.sqrt(2) * sigma)))

        return p_value, p_value >= self.alpha

    def linear_complexity_test(self, bits: np.ndarray, M=500) -> Tuple[float, bool]:
        """
        Test 10: Linear Complexity Test
        """
        n = len(bits)
        K = 6
        N = n // M

        if N == 0:
            return 0.0, False

        mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - 1 / (2**M) * (M / 3 + 2 / 9)

        # Probabilidades para K=6
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]

        v = np.zeros(7)
        for i in range(N):
            block = bits[i * M : (i + 1) * M]
            L = self._berlekamp_massey(block)
            T = (-1) ** M * (L - mu) + 2 / 9

            if T <= -2.5:
                v[0] += 1
            elif T <= -1.5:
                v[1] += 1
            elif T <= -0.5:
                v[2] += 1
            elif T <= 0.5:
                v[3] += 1
            elif T <= 1.5:
                v[4] += 1
            elif T <= 2.5:
                v[5] += 1
            else:
                v[6] += 1

        chi_squared = np.sum((v - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p_value = spc.gammaincc(K / 2, chi_squared / 2)

        return p_value, p_value >= self.alpha

    def _berlekamp_massey(self, bits: np.ndarray) -> int:
        """Algoritmo de Berlekamp-Massey para calcular complejidad lineal"""
        n = len(bits)
        b = np.array([1] + [0] * n)
        c = np.array([1] + [0] * n)
        L, m = 0, -1

        for i in range(n):
            d = bits[i]
            for j in range(1, L + 1):
                d ^= c[j] & bits[i - j]

            if d == 1:
                t = c.copy()
                for j in range(i - m, n + 1):
                    if j < len(b):
                        c[j] ^= b[j - (i - m)]

                if L <= i / 2:
                    L = i + 1 - L
                    m = i
                    b = t

        return L

    def serial_test(self, bits: np.ndarray, m=16) -> Tuple[float, bool]:
        """
        Test 11: Serial Test
        """
        n = len(bits)

        def psi_sq(m, bits):
            if m == 0:
                return 0
            counts = {}
            for i in range(n):
                pattern = tuple(
                    bits[i : i + m]
                    if i + m <= n
                    else np.concatenate([bits[i:], bits[: i + m - n]])
                )
                counts[pattern] = counts.get(pattern, 0) + 1

            sum_val = sum(count**2 for count in counts.values())
            return (2**m / n) * sum_val - n

        psi2_m = psi_sq(m, bits)
        psi2_m1 = psi_sq(m - 1, bits)
        psi2_m2 = psi_sq(m - 2, bits)

        delta1 = psi2_m - psi2_m1
        delta2 = psi2_m - 2 * psi2_m1 + psi2_m2

        p_value1 = spc.gammaincc(2 ** (m - 2), delta1 / 2)
        p_value2 = spc.gammaincc(2 ** (m - 3), delta2 / 2)

        p_value = min(p_value1, p_value2)

        return p_value, p_value >= self.alpha

    def approximate_entropy_test(self, bits: np.ndarray, m=10) -> Tuple[float, bool]:
        """
        Test 12: Approximate Entropy Test
        """
        n = len(bits)

        def phi(m):
            counts = {}
            for i in range(n):
                pattern = tuple(
                    bits[i : i + m]
                    if i + m <= n
                    else np.concatenate([bits[i:], bits[: i + m - n]])
                )
                counts[pattern] = counts.get(pattern, 0) + 1

            return sum((count / n) * np.log(count / n) for count in counts.values())

        phi_m = phi(m)
        phi_m1 = phi(m + 1)

        appen = phi_m - phi_m1
        chi_squared = 2 * n * (np.log(2) - appen)

        p_value = spc.gammaincc(2 ** (m - 1), chi_squared / 2)

        return p_value, p_value >= self.alpha

    def cumulative_sums_test(self, bits: np.ndarray, mode=0) -> Tuple[float, bool]:
        """
        Test 13: Cumulative Sums (Cusum) Test
        mode: 0 para forward, 1 para backward
        """
        n = len(bits)
        x = 2 * bits - 1  # Convierte 0->-1, 1->1

        if mode == 1:
            x = x[::-1]

        cumsum = np.cumsum(x)
        z = np.max(np.abs(cumsum))

        # Calcular p-value
        sum_a = 0.0
        sum_b = 0.0

        for k in range(int(-n / z + 1), int(n / z)):
            sum_a += norm.cdf((4 * k + 1) * z / np.sqrt(n)) - norm.cdf(
                (4 * k - 1) * z / np.sqrt(n)
            )
            sum_b += norm.cdf((4 * k + 3) * z / np.sqrt(n)) - norm.cdf(
                (4 * k + 1) * z / np.sqrt(n)
            )

        p_value = 1 - sum_a + sum_b

        return p_value, p_value >= self.alpha

    def random_excursions_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        Test 14: Random Excursions Test
        """
        n = len(bits)
        x = np.concatenate(([0], np.cumsum(2 * bits - 1), [0]))

        # Encontrar ciclos (retornos a cero)
        zero_indices = np.where(x == 0)[0]
        cycles = len(zero_indices) - 1

        if cycles < 500:
            return 0.0, False

        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        p_values = []

        for state in states:
            v = np.zeros(6)
            for i in range(cycles):
                cycle = x[zero_indices[i] : zero_indices[i + 1] + 1]
                count = np.sum(cycle == state)
                if count <= 4:
                    v[count] += 1
                else:
                    v[5] += 1

            # Probabilidades teóricas
            if state in [-1, 1]:
                pi = [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0313]
            else:
                pi = [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0740]

            chi_squared = np.sum(
                (v - cycles * np.array(pi)) ** 2 / (cycles * np.array(pi))
            )
            p_value = spc.gammaincc(5 / 2, chi_squared / 2)
            p_values.append(p_value)

        final_p_value = np.min(p_values)
        return final_p_value, final_p_value >= self.alpha

    def random_excursions_variant_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """
        Test 15: Random Excursions Variant Test
        """
        n = len(bits)
        x = np.concatenate(([0], np.cumsum(2 * bits - 1), [0]))

        # Encontrar ciclos
        zero_indices = np.where(x == 0)[0]
        J = len(zero_indices) - 1

        if J < 500:
            return 0.0, False

        states = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p_values = []

        for state in states:
            count = np.sum(x == state)
            p_value = spc.erfc(
                np.abs(count - J) / np.sqrt(2 * J * (4 * np.abs(state) - 2))
            )
            p_values.append(p_value)

        final_p_value = np.min(p_values)
        return final_p_value, final_p_value >= self.alpha

    def run_all_tests(self, bits: np.ndarray) -> pd.DataFrame:
        """
        Ejecuta todos los tests y devuelve resultados en DataFrame
        """
        results = []

        tests = [
            ("1. Frequency (Monobit)", lambda: self.monobit_test(bits)),
            (
                "2. Frequency within Block",
                lambda: self.frequency_within_block_test(bits),
            ),
            ("3. Runs", lambda: self.runs_test(bits)),
            ("4. Longest Run of Ones", lambda: self.longest_run_test(bits)),
            ("5. Binary Matrix Rank", lambda: self.binary_matrix_rank_test(bits)),
            ("6. Discrete Fourier Transform", lambda: self.dft_test(bits)),
            (
                "7. Non-overlapping Template",
                lambda: self.non_overlapping_template_test(bits),
            ),
            ("8. Overlapping Template", lambda: self.overlapping_template_test(bits)),
            ("9. Maurer's Universal", lambda: self.maurers_universal_test(bits)),
            ("10. Linear Complexity", lambda: self.linear_complexity_test(bits)),
            ("11. Serial", lambda: self.serial_test(bits)),
            ("12. Approximate Entropy", lambda: self.approximate_entropy_test(bits)),
            ("13. Cumulative Sums", lambda: self.cumulative_sums_test(bits)),
            ("14. Random Excursions", lambda: self.random_excursions_test(bits)),
            (
                "15. Random Excursions Variant",
                lambda: self.random_excursions_variant_test(bits),
            ),
        ]

        for test_name, test_func in tests:
            obj = {}
            try:
                p_value, passed = test_func()
                obj = {
                    "Test": test_name,
                    "P-Value": f"{p_value:.6f}",
                    "Result": "PASS" if passed else "FAIL",
                }
            except Exception as e:
                obj = {
                    "Test": test_name,
                    "P-Value": "ERROR",
                    "Result": f"ERROR: {str(e)[:30]}",
                }
            print(f"{obj['Test']}: {obj['Result']}\n\t- P-Value: {obj['P-Value']}")
            results.append(obj)

        return pd.DataFrame(results)


# ==================== CÓDIGO PRINCIPAL ====================


# Clases de generadores (del documento proporcionado)
class BitGeneratorLCG:
    def __init__(self, seed, a=None, c=None, m=None):
        self.seed = seed
        self.a = a if a is not None else 7**5
        self.c = c if c is not None else 0
        self.m = m if m is not None else 2**31 - 1
        self.current = seed
        self.bits_per_number = 31

    def next_number(self):
        self.current = (self.a * self.current + self.c) % self.m
        return self.current

    def generate_bits(self, n):
        bits = []
        bits_needed = n
        while bits_needed > 0:
            number = self.next_number()
            bits_to_extract = min(bits_needed, self.bits_per_number)
            for i in range(bits_to_extract):
                bits.append((number >> i) & 1)
            bits_needed -= bits_to_extract
        return np.array(bits[:n], dtype=int)

    def reset(self):
        self.current = self.seed


class BitGeneratorMT:
    def __init__(self, seed=5489):
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253
        self.MT = [0] * self.n
        self.index = self.n + 1
        self.lower_mask = (1 << self.r) - 1
        self.upper_mask = (~self.lower_mask) & 0xFFFFFFFF
        self.initial_seed = seed
        self.bits_per_number = 32
        self.seed_mt(seed)

    def seed_mt(self, seed):
        self.index = self.n
        self.MT[0] = seed & 0xFFFFFFFF
        for i in range(1, self.n):
            self.MT[i] = (
                self.f * (self.MT[i - 1] ^ (self.MT[i - 1] >> (self.w - 2))) + i
            ) & 0xFFFFFFFF

    def twist(self):
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
        if self.index >= self.n:
            self.twist()
        y = self.MT[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= y >> self.l
        self.index += 1
        return y & 0xFFFFFFFF

    def generate_bits(self, n):
        bits = []
        bits_needed = n
        while bits_needed > 0:
            number = self.extract_number()
            bits_to_extract = min(bits_needed, self.bits_per_number)
            for i in range(bits_to_extract):
                bits.append((number >> i) & 1)
            bits_needed -= bits_to_extract
        return np.array(bits[:n], dtype=int)

    def reset(self):
        self.seed_mt(self.initial_seed)


# ==================== EJECUTAR PRUEBAS ====================

print("=" * 80)
print("PRUEBAS NIST SP 800-22 PARA GENERADORES DE BITS ALEATORIOS")
print("=" * 80)

# Generar secuencias de bits
# N = 1_000_000
N = 100_000
print(f"\nGenerando {N:,} bits con cada generador...\n")

lcg = BitGeneratorLCG(seed=42)
bits_lcg = lcg.generate_bits(N)
print(f"✓ LCG: {N:,} bits generados")

mt = BitGeneratorMT(seed=42)
bits_mt = mt.generate_bits(N)
print(f"✓ Mersenne Twister: {N:,} bits generados")

# Ejecutar tests
print("\n" + "=" * 80)
print("EJECUTANDO TESTS NIST...")
print("=" * 80)

nist = NISTTests(significance_level=0.01)

print("\n[1/2] Evaluando Linear Congruential Generator (LCG)...")
results_lcg = nist.run_all_tests(bits_lcg)

print("[2/2] Evaluando Mersenne Twister (MT19937)...")
results_mt = nist.run_all_tests(bits_mt)

# Crear tabla comparativa
print("\n" + "=" * 80)
print("RESULTADOS COMPARATIVOS")
print("=" * 80)

comparison = pd.DataFrame(
    {
        "Test": results_lcg["Test"],
        "LCG P-Value": results_lcg["P-Value"],
        "LCG Result": results_lcg["Result"],
        "MT P-Value": results_mt["P-Value"],
        "MT Result": results_mt["Result"],
    }
)

print("\n" + comparison.to_string(index=False))

# Análisis de resultados
print("\n" + "=" * 80)
print("ANÁLISIS Y CONCLUSIONES")
print("=" * 80)

lcg_passed = sum(results_lcg["Result"] == "PASS")
mt_passed = sum(results_mt["Result"] == "PASS")
total_tests = len(results_lcg)

print("\nRESUMEN DE DESEMPEÑO:")
print(f"{'─' * 60}")
print("Linear Congruential Generator (LCG):")
print(
    f"  • Tests aprobados: {lcg_passed}/{total_tests} ({lcg_passed / total_tests * 100:.1f}%)"
)
print(f"  • Tests fallidos: {total_tests - lcg_passed}/{total_tests}")

print("\nMersenne Twister (MT19937):")
print(
    f"  • Tests aprobados: {mt_passed}/{total_tests} ({mt_passed / total_tests * 100:.1f}%)"
)
print(f"  • Tests fallidos: {total_tests - mt_passed}/{total_tests}")

print(f"\n{'─' * 60}")
print("CONCLUSIÓN:")
print(f"{'─' * 60}")

if mt_passed > lcg_passed:
    diff = mt_passed - lcg_passed
    print("\n✓ El Mersenne Twister (MT19937) se desempeña MEJOR que el LCG")
    print(f"  - Aprobó {diff} test(s) adicional(es)")
    print(
        f"  - Tasa de éxito: {mt_passed / total_tests * 100:.1f}% vs {lcg_passed / total_tests * 100:.1f}%"
    )
elif lcg_passed > mt_passed:
    diff = lcg_passed - mt_passed
    print("\n✓ El Linear Congruential Generator (LCG) se desempeña MEJOR que el MT")
    print(f"  - Aprobó {diff} test(s) adicional(es)")
    print(
        f"  - Tasa de éxito: {lcg_passed / total_tests * 100:.1f}% vs {mt_passed / total_tests * 100:.1f}%"
    )
else:
    print("\n⚖ Ambos generadores tienen desempeño SIMILAR")
    print(f"  - Ambos aprobaron {lcg_passed}/{total_tests} tests")
    print(f"  - Tasa de éxito idéntica: {lcg_passed / total_tests * 100:.1f}%")

print(f"\n{'═' * 80}")
print("Nivel de significancia usado: α = 0.01")
print("Un test PASA si p-value ≥ 0.01")
print(f"{'═' * 80}\n")
