import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

# ====================
# 1. KONFIGURASI & SETUP TANGKI AIR
# ====================

@dataclass
class WaterTankConfig:
    """Konfigurasi parameter tangki air"""
    # Dimensi tangki
    tank_diameter: float = 2.0          # m
    tank_height: float = 3.0             # m
    tank_volume: float = field(init=False, default=None)  # m³
    
    # Parameter pipa
    inlet_pipe_diameter: float = 0.1     # m
    outlet_pipe_diameter: float = 0.15    # m
    inlet_pipe_length: float = 5.0        # m
    outlet_pipe_length: float = 10.0       # m
    
    # Parameter aliran
    inlet_velocity: float = 1.5           # m/s
    outlet_velocity: float = 2.0           # m/s (saat gravitasi)
    pump_pressure: float = 200000.0        # Pa (tekanan pompa)
    
    # Kondisi awal
    initial_water_height: float = 0.5      # m
    initial_water_volume: float = field(init=False, default=None)  # m³
    
    # Parameter fisik
    water_density: float = 1000.0          # kg/m³
    gravity: float = 9.81                   # m/s²
    water_viscosity: float = 0.001          # Pa·s
    atmospheric_pressure: float = 101325.0  # Pa
    
    # Koefisien kerugian
    inlet_loss_coefficient: float = 0.5     # Koefisien kerugian inlet
    outlet_loss_coefficient: float = 0.8    # Koefisien kerugian outlet
    friction_factor: float = 0.02            # Faktor gesekan pipa
    
    # Kebutuhan air asrama
    building_occupants: int = 100            # Jumlah penghuni
    water_consumption_per_person: float = 150  # L/orang/hari
    daily_water_demand: float = field(init=False, default=None)  # m³/hari
    
    # Parameter simulasi
    simulation_time: float = 21600.0          # detik (6 jam, lebih realistis)
    time_step: float = 1.0                   # detik
    
    def __post_init__(self):
        """Validasi konfigurasi dan hitung atribut turunan"""
        # Hitung volume tangki
        self.tank_volume = np.pi * (self.tank_diameter/2)**2 * self.tank_height
        
        # Hitung volume air awal
        self.initial_water_volume = np.pi * (self.tank_diameter/2)**2 * self.initial_water_height
        
        # Hitung kebutuhan air harian
        self.daily_water_demand = (self.building_occupants * 
                                   self.water_consumption_per_person / 1000.0)  # m³/hari
        
        # Validasi
        if self.initial_water_height > self.tank_height:
            st.warning("Peringatan: Ketinggian air awal melebihi tinggi tangki!")
        
        if self.daily_water_demand > self.tank_volume * 24:  # Asumsi siklus pengisian per jam
            st.warning("Peringatan: Kebutuhan air melebihi kapasitas suplai tangki!")
    
    def copy(self):
        """Buat salinan konfigurasi"""
        params = {k: v for k, v in self.__dict__.items() 
                 if k not in ['tank_volume', 'initial_water_volume', 'daily_water_demand']}
        new_config = WaterTankConfig(**params)
        return new_config
    
    def update_parameter(self, parameter_name: str, value: float):
        """Update satu parameter dan hitung ulang atribut turunan"""
        if parameter_name in self.__annotations__:
            setattr(self, parameter_name, value)
            self.__post_init__()
        else:
            raise ValueError(f"Parameter {parameter_name} tidak valid")

# ====================
# 2. MODEL FISIKA TANGKI AIR
# ====================

class WaterTankPhysics:
    """Model fisika untuk sistem tangki air"""
    
    def __init__(self, config: WaterTankConfig):
        self.config = config
    
    def calculate_inlet_flow_rate(self, water_height: float) -> float:
        """
        Hitung laju aliran masuk (m³/s)
        Dipengaruhi oleh tekanan pompa dan ketinggian air
        """
        # Luas penampang pipa inlet
        inlet_area = np.pi * (self.config.inlet_pipe_diameter/2)**2
        
        # Tekanan hidrostatik dari ketinggian air
        hydrostatic_pressure = self.config.water_density * self.config.gravity * water_height
        
        # Tekanan efektif (pompa - hidrostatik - kerugian)
        effective_pressure = (self.config.pump_pressure - hydrostatic_pressure)
        
        if effective_pressure <= 0:
            return 0.0
        
        # Kecepatan aliran (Persamaan Bernoulli dengan kerugian)
        velocity = np.sqrt(2 * effective_pressure / self.config.water_density)
        
        # Faktor kerugian
        loss_factor = 1.0 / (1.0 + self.config.inlet_loss_coefficient)
        
        # Laju aliran volumetrik
        flow_rate = inlet_area * velocity * loss_factor
        
        return max(flow_rate, 0.0)
    
    def calculate_outlet_flow_rate(self, water_height: float) -> float:
        """
        Hitung laju aliran keluar (m³/s)
        Dipengaruhi oleh ketinggian air (gravitasi)
        """
        if water_height <= 0:
            return 0.0
        
        # Luas penampang pipa outlet
        outlet_area = np.pi * (self.config.outlet_pipe_diameter/2)**2
        
        # Kecepatan aliran akibat gravitasi (Torricelli dengan kerugian)
        theoretical_velocity = np.sqrt(2 * self.config.gravity * water_height)
        
        # Faktor kerugian (termasuk gesekan dan belokan)
        loss_factor = 1.0 / np.sqrt(1.0 + self.config.outlet_loss_coefficient + 
                                    self.config.friction_factor * 
                                    self.config.outlet_pipe_length / 
                                    self.config.outlet_pipe_diameter)
        
        actual_velocity = theoretical_velocity * loss_factor
        
        # Laju aliran volumetrik
        flow_rate = outlet_area * actual_velocity
        
        return flow_rate
    
    def calculate_simultaneous_flow(self, water_height: float, 
                                   inlet_active: bool = True,
                                   outlet_active: bool = True) -> Tuple[float, float]:
        """
        Hitung laju aliran saat pengisian dan pengosongan bersamaan
        """
        inlet_rate = self.calculate_inlet_flow_rate(water_height) if inlet_active else 0.0
        outlet_rate = self.calculate_outlet_flow_rate(water_height) if outlet_active else 0.0
        
        return inlet_rate, outlet_rate
    
    def calculate_tank_cross_section(self) -> float:
        """Hitung luas penampang tangki"""
        return np.pi * (self.config.tank_diameter/2)**2
    
    def height_to_volume(self, height: float) -> float:
        """Konversi ketinggian ke volume"""
        return self.calculate_tank_cross_section() * height
    
    def volume_to_height(self, volume: float) -> float:
        """Konversi volume ke ketinggian"""
        return volume / self.calculate_tank_cross_section()
    
    def calculate_optimal_tank_size(self) -> Dict:
        """
        Hitung ukuran tangki optimal berdasarkan kebutuhan
        """
        cross_section = self.calculate_tank_cross_section()
        
        # Volume minimum untuk memenuhi kebutuhan 1 hari
        min_volume = self.config.daily_water_demand * 1.5  # 1.5 hari cadangan
        
        # Tinggi minimum
        min_height = min_volume / cross_section
        
        # Diameter optimal (asumsi rasio tinggi:diameter = 1.5:1)
        optimal_diameter = (4 * min_volume / (1.5 * np.pi))**(1/3)
        optimal_height = 1.5 * optimal_diameter
        
        return {
            'min_volume': min_volume,
            'min_height': min_height,
            'optimal_diameter': optimal_diameter,
            'optimal_height': optimal_height,
            'optimal_volume': np.pi * (optimal_diameter/2)**2 * optimal_height,
            'current_diameter': self.config.tank_diameter,
            'current_height': self.config.tank_height,
            'current_volume': self.config.tank_volume
        }

# ====================
# 3. SISTEM PERSAMAAN DIFERENSIAL
# ====================

class DifferentialEquations:
    """Sistem persamaan diferensial untuk simulasi tangki air"""
    
    def __init__(self, physics_model: WaterTankPhysics):
        self.physics = physics_model
        self.config = physics_model.config
    
    def system_equations(self, t: float, y: np.ndarray, 
                        inlet_active: bool = True,
                        outlet_active: bool = True) -> np.ndarray:
        """
        Sistem persamaan diferensial:
        y = [water_height]
        
        Returns:
        dy/dt = [dh/dt]
        """
        h = y[0]
        
        # Batasi ketinggian antara 0 dan tinggi tangki
        h = np.clip(h, 0, self.config.tank_height)
        
        # Hitung laju aliran
        Q_in, Q_out = self.physics.calculate_simultaneous_flow(h, inlet_active, outlet_active)
        
        # Luas penampang tangki
        A = self.physics.calculate_tank_cross_section()
        
        # Perubahan ketinggian (dh/dt = (Q_in - Q_out) / A)
        if A > 0:
            dh_dt = (Q_in - Q_out) / A
        else:
            dh_dt = 0.0
        
        # Batasi laju perubahan
        dh_dt = np.clip(dh_dt, -0.1, 0.1)
        
        return np.array([dh_dt])
    
    def get_initial_conditions(self) -> np.ndarray:
        """Kondisi awal sistem"""
        return np.array([self.config.initial_water_height])

# ====================
# 4. SIMULATOR UTAMA TANGKI AIR (DIPERBAIKI)
# ====================

class WaterTankSimulator:
    """Simulator utama sistem tangki air"""
    
    def __init__(self, config: WaterTankConfig):
        self.config = config
        self.physics = WaterTankPhysics(config)
        self.equations = DifferentialEquations(self.physics)
        
        # Results storage
        self.time_history = None
        self.height_history = None
        self.volume_history = None
        self.inlet_rate_history = None
        self.outlet_rate_history = None
        self.results = None
    
    def run_simulation(self, inlet_active: bool = True, 
                      outlet_active: bool = True,
                      skip_metrics: bool = False) -> Dict:
        """
        Jalankan simulasi dengan opsi aktivasi inlet/outlet
        
        Parameters:
        - skip_metrics: Jika True, tidak menghitung metrik waktu pengisian/pengosongan
                        untuk menghindari rekursi
        """
        # Setup time
        t_span = (0, self.config.simulation_time)
        t_eval = np.arange(0, self.config.simulation_time, self.config.time_step)
        
        # Initial conditions
        y0 = self.equations.get_initial_conditions()
        
        # Solve ODE system
        solution = solve_ivp(
            fun=lambda t, y: self.equations.system_equations(t, y, inlet_active, outlet_active),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        # Store results
        self.time_history = solution.t / 60.0  # Convert to minutes
        self.height_history = solution.y[0]
        
        # Calculate volume
        A = self.physics.calculate_tank_cross_section()
        self.volume_history = self.height_history * A
        
        # Calculate flow rates
        self.inlet_rate_history = []
        self.outlet_rate_history = []
        
        for h in self.height_history:
            Q_in, Q_out = self.physics.calculate_simultaneous_flow(h, inlet_active, outlet_active)
            self.inlet_rate_history.append(Q_in)   # m³/s
            self.outlet_rate_history.append(Q_out) # m³/s
        
        # Calculate metrics (skip jika diminta untuk menghindari rekursi)
        if not skip_metrics:
            self.results = self._calculate_metrics(inlet_active, outlet_active)
        else:
            self.results = self._calculate_basic_metrics()
        
        return self.results
    
    def _calculate_basic_metrics(self) -> Dict:
        """Hitung metrik dasar tanpa menghitung waktu pengisian/pengosongan"""
        if self.time_history is None:
            raise ValueError("Jalankan simulasi terlebih dahulu")
        
        # Hitung total volume menggunakan metode trapezoid manual
        dt = self.config.time_step / 60.0  # Konversi ke menit untuk integrasi
        
        # Integrasi manual untuk total volume
        total_inlet = 0
        total_outlet = 0
        for i in range(1, len(self.time_history)):
            dt_actual = (self.time_history[i] - self.time_history[i-1]) * 60  # detik
            total_inlet += (self.inlet_rate_history[i-1] + self.inlet_rate_history[i]) / 2 * dt_actual
            total_outlet += (self.outlet_rate_history[i-1] + self.outlet_rate_history[i]) / 2 * dt_actual
        
        metrics = {
            # Ketinggian
            'max_height': np.max(self.height_history),
            'min_height': np.min(self.height_history),
            'final_height': self.height_history[-1],
            'avg_height': np.mean(self.height_history),
            
            # Volume
            'max_volume': np.max(self.volume_history),
            'min_volume': np.min(self.volume_history),
            'final_volume': self.volume_history[-1],
            'avg_volume': np.mean(self.volume_history),
            
            # Laju aliran
            'max_inlet_rate': np.max(self.inlet_rate_history) if self.inlet_rate_history else 0,
            'max_outlet_rate': np.max(self.outlet_rate_history) if self.outlet_rate_history else 0,
            'avg_inlet_rate': np.mean(self.inlet_rate_history) if self.inlet_rate_history else 0,
            'avg_outlet_rate': np.mean(self.outlet_rate_history) if self.outlet_rate_history else 0,
            
            # Total volume yang diproses (menggunakan integrasi manual)
            'total_inlet_volume': total_inlet,
            'total_outlet_volume': total_outlet,
            
            # Status
            'tank_full': self.height_history[-1] >= self.config.tank_height - 0.01,
            'tank_empty': self.height_history[-1] <= 0.01,
            'fill_percentage': (self.height_history[-1] / self.config.tank_height) * 100,
            
            # Placeholder untuk waktu (akan diisi nanti)
            'time_to_fill': None,
            'time_to_empty': None
        }
        
        return metrics
    
    def _calculate_metrics(self, inlet_active: bool, outlet_active: bool) -> Dict:
        """Hitung metrik kinerja tangki"""
        if self.time_history is None:
            raise ValueError("Jalankan simulasi terlebih dahulu")
        
        # Dapatkan metrik dasar
        metrics = self._calculate_basic_metrics()
        
        # Hitung waktu pengisian dan pengosongan secara terpisah (tanpa rekursi)
        if inlet_active and not outlet_active:
            # Jika ini adalah simulasi pengisian saja, hitung waktu ke penuh
            for i, h in enumerate(self.height_history):
                if h >= self.config.tank_height:
                    metrics['time_to_fill'] = self.time_history[i]
                    break
        
        if not inlet_active and outlet_active:
            # Jika ini adalah simulasi pengosongan saja, hitung waktu ke kosong
            for i, h in enumerate(self.height_history):
                if h <= 0.01:
                    metrics['time_to_empty'] = self.time_history[i]
                    break
        
        return metrics
    
    # ==================== PERBAIKAN UTAMA: FUNGSI WAKTU PENGISIAN REALISTIS ====================
    
    def calculate_fill_time(self) -> Optional[Dict]:
        """
        Hitung waktu yang dibutuhkan untuk mengisi dari kosong ke penuh
        Menggunakan integrasi numerik yang mempertimbangkan perubahan tekanan hidrostatik
        Returns dictionary dengan detail waktu pengisian
        """
        # Buat konfigurasi baru dengan ketinggian awal 0
        fill_config = self.config.copy()
        fill_config.initial_water_height = 0.0
        
        # Buat simulator baru
        fill_sim = WaterTankSimulator(fill_config)
        
        # Setup time yang lebih panjang (maksimal 24 jam)
        max_time = 24 * 3600  # 24 jam dalam detik
        t_span = (0, max_time)
        t_eval = np.arange(0, max_time, 60)  # Evaluasi setiap menit
        
        # Initial conditions
        y0 = np.array([0.0])
        
        # Solve ODE system untuk pengisian
        solution = solve_ivp(
            fun=lambda t, y: fill_sim.equations.system_equations(t, y, inlet_active=True, outlet_active=False),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        time_history = solution.t / 60.0  # Konversi ke menit
        height_history = solution.y[0]
        
        # Cari waktu saat ketinggian mencapai tinggi tangki
        fill_time = None
        fill_index = None
        
        for i, h in enumerate(height_history):
            if h >= self.config.tank_height - 0.001:  # Threshold 1 mm
                fill_time = time_history[i]
                fill_index = i
                break
        
        if fill_time is None:
            # Jika tidak mencapai penuh, ambil waktu maksimum
            fill_time = time_history[-1]
            fill_index = len(height_history) - 1
        
        # Hitung detail proses pengisian
        A = self.physics.calculate_tank_cross_section()
        inlet_area = np.pi * (self.config.inlet_pipe_diameter/2)**2
        
        # Hitung laju aliran rata-rata pada berbagai fase
        n_points = min(10, len(height_history))
        indices = np.linspace(0, fill_index, n_points, dtype=int)
        
        flow_rates = []
        heights = []
        effective_pressures = []
        
        for idx in indices:
            h = height_history[idx]
            
            # Tekanan hidrostatik
            hydrostatic = self.config.water_density * self.config.gravity * h
            effective_pressure = self.config.pump_pressure - hydrostatic
            
            # Laju aliran
            if effective_pressure > 0:
                velocity = np.sqrt(2 * effective_pressure / self.config.water_density)
                loss_factor = 1.0 / (1.0 + self.config.inlet_loss_coefficient)
                Q_in = inlet_area * velocity * loss_factor * 3600  # m³/jam
            else:
                Q_in = 0
            
            flow_rates.append(Q_in)
            heights.append(h)
            effective_pressures.append(effective_pressure / 1000)  # kPa
        
        # Hitung tekanan hidrostatik maksimum
        max_hydrostatic = self.config.water_density * self.config.gravity * self.config.tank_height / 1000  # kPa
        
        # Analisis teoritis
        # Kecepatan teoritis maksimum (saat tangki kosong)
        v_max = np.sqrt(2 * self.config.pump_pressure / self.config.water_density)
        Q_max = inlet_area * v_max * 3600  # m³/jam
        
        # Kecepatan teoritis minimum (saat tangki hampir penuh)
        effective_pressure_min = max(0, self.config.pump_pressure - 
                                    self.config.water_density * self.config.gravity * self.config.tank_height)
        if effective_pressure_min > 0:
            v_min = np.sqrt(2 * effective_pressure_min / self.config.water_density)
            Q_min = inlet_area * v_min * 3600
        else:
            v_min = 0
            Q_min = 0
        
        # Waktu pengisian teoritis (dengan asumsi laju aliran konstan rata-rata)
        Q_avg_theoretical = (Q_max + Q_min) / 2 if Q_min > 0 else Q_max / 2
        theoretical_time = self.config.tank_volume / Q_avg_theoretical if Q_avg_theoretical > 0 else float('inf')
        theoretical_time_minutes = theoretical_time * 60
        
        return {
            'fill_time_minutes': fill_time,
            'fill_time_hours': fill_time / 60,
            'fill_time_seconds': fill_time * 60,
            'reached_full': fill_index is not None,
            'final_height': height_history[fill_index] if fill_index is not None else height_history[-1],
            'fill_percentage': (height_history[fill_index] / self.config.tank_height * 100) if fill_index is not None else 0,
            'flow_rate_profile': {
                'heights': heights,
                'flow_rates': flow_rates,  # m³/jam
                'effective_pressures': effective_pressures  # kPa
            },
            'theoretical_analysis': {
                'Q_max_m3_per_jam': Q_max,
                'Q_min_m3_per_jam': Q_min,
                'Q_avg_m3_per_jam': Q_avg_theoretical,
                'theoretical_time_minutes': theoretical_time_minutes,
                'total_volume_m3': self.config.tank_volume,
                'max_hydrostatic_kPa': max_hydrostatic,
                'pump_pressure_kPa': self.config.pump_pressure / 1000
            }
        }
    
    # ==================== PERBAIKAN UTAMA: FUNGSI WAKTU PENGOSONGAN REALISTIS ====================
    
    def calculate_empty_time(self) -> Optional[Dict]:
        """
        Hitung waktu yang dibutuhkan untuk mengosongkan dari penuh ke kosong
        Menggunakan model Torricelli dengan mempertimbangkan perubahan tekanan hidrostatik
        Returns dictionary dengan detail waktu pengosongan
        """
        # Buat konfigurasi baru dengan ketinggian awal penuh
        empty_config = self.config.copy()
        empty_config.initial_water_height = self.config.tank_height
        
        # Buat simulator baru
        empty_sim = WaterTankSimulator(empty_config)
        
        # Setup time
        max_time = 24 * 3600  # 24 jam
        t_span = (0, max_time)
        t_eval = np.arange(0, max_time, 60)  # Evaluasi setiap menit
        
        # Initial conditions
        y0 = np.array([self.config.tank_height])
        
        # Solve ODE system untuk pengosongan
        solution = solve_ivp(
            fun=lambda t, y: empty_sim.equations.system_equations(t, y, inlet_active=False, outlet_active=True),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        time_history = solution.t / 60.0  # Konversi ke menit
        height_history = solution.y[0]
        
        # Cari waktu saat ketinggian mencapai 0
        empty_time = None
        empty_index = None
        
        for i, h in enumerate(height_history):
            if h <= 0.001:  # Threshold 1 mm
                empty_time = time_history[i]
                empty_index = i
                break
        
        if empty_time is None:
            empty_time = time_history[-1]
            empty_index = len(height_history) - 1
        
        # Hitung detail proses pengosongan
        A = self.physics.calculate_tank_cross_section()
        outlet_area = np.pi * (self.config.outlet_pipe_diameter/2)**2
        
        # Hitung laju aliran keluar pada berbagai ketinggian
        n_points = min(10, len(height_history))
        indices = np.linspace(0, empty_index, n_points, dtype=int)
        
        flow_rates = []
        heights = []
        velocities = []
        theoretical_velocities = []
        
        for idx in indices:
            h = height_history[idx]
            
            # Kecepatan Torricelli
            theoretical_velocity = np.sqrt(2 * self.config.gravity * h)
            
            # Faktor kerugian
            loss_factor = 1.0 / np.sqrt(1.0 + self.config.outlet_loss_coefficient + 
                                        self.config.friction_factor * 
                                        self.config.outlet_pipe_length / 
                                        self.config.outlet_pipe_diameter)
            
            actual_velocity = theoretical_velocity * loss_factor
            Q_out = outlet_area * actual_velocity * 3600  # m³/jam
            
            flow_rates.append(Q_out)
            heights.append(h)
            velocities.append(actual_velocity)
            theoretical_velocities.append(theoretical_velocity)
        
        # Analisis teoritis menggunakan rumus pengosongan tangki
        # Waktu pengosongan teoritis untuk tangki prismatis
        # t = (2 * A * sqrt(H)) / (Cd * A_outlet * sqrt(2g))
        
        Cd = 1.0 / np.sqrt(1.0 + self.config.outlet_loss_coefficient + 
                           self.config.friction_factor * 
                           self.config.outlet_pipe_length / 
                           self.config.outlet_pipe_diameter)
        
        theoretical_time = (2 * A * np.sqrt(self.config.tank_height)) / (Cd * outlet_area * np.sqrt(2 * self.config.gravity))
        theoretical_time_minutes = theoretical_time / 60
        
        # Validasi dengan rumus empiris (memperhitungkan gesekan lebih akurat)
        # Menggunakan faktor koreksi untuk pipa panjang
        friction_correction = 1.0 + 0.5 * self.config.friction_factor * (self.config.outlet_pipe_length / self.config.outlet_pipe_diameter)
        empirical_time = theoretical_time * friction_correction
        empirical_time_minutes = empirical_time / 60
        
        # Hitung waktu paruh (waktu untuk mencapai setengah ketinggian)
        half_height = self.config.tank_height / 2
        half_time = None
        for i, h in enumerate(height_history):
            if h <= half_height:
                half_time = time_history[i]
                break
        
        return {
            'empty_time_minutes': empty_time,
            'empty_time_hours': empty_time / 60,
            'empty_time_seconds': empty_time * 60,
            'half_time_minutes': half_time,
            'reached_empty': empty_index is not None,
            'final_height': height_history[empty_index],
            'drain_percentage': ((self.config.tank_height - height_history[empty_index]) / self.config.tank_height * 100),
            'flow_rate_profile': {
                'heights': heights,
                'flow_rates': flow_rates,  # m³/jam
                'velocities': velocities,  # m/s
                'theoretical_velocities': theoretical_velocities  # m/s
            },
            'theoretical_analysis': {
                'Cd': Cd,  # Koefisien discharge
                'theoretical_time_minutes': theoretical_time_minutes,
                'empirical_time_minutes': empirical_time_minutes,
                'max_velocity': velocities[0] if velocities else 0,
                'min_velocity': velocities[-1] if velocities else 0,
                'avg_flow_rate_m3_per_jam': np.mean(flow_rates) if flow_rates else 0,
                'friction_correction': friction_correction
            }
        }
    
    def validate_fill_time_analytical(self) -> Dict:
        """
        Validasi waktu pengisian menggunakan metode analitik
        Untuk tangki prismatik dengan luas penampang konstan
        """
        A_tank = self.physics.calculate_tank_cross_section()
        A_inlet = np.pi * (self.config.inlet_pipe_diameter/2)**2
        
        # Koefisien discharge untuk inlet
        Cd = 1.0 / np.sqrt(1.0 + self.config.inlet_loss_coefficient)
        
        # Fungsi untuk integrasi numerik 1/Q_in(h)
        def dh_to_dt(h):
            if h >= self.config.tank_height:
                return 0
            effective_pressure = self.config.pump_pressure - self.config.water_density * self.config.gravity * h
            if effective_pressure <= 0:
                return float('inf')
            Q = Cd * A_inlet * np.sqrt(2 * effective_pressure / self.config.water_density)
            return A_tank / Q if Q > 0 else float('inf')
        
        # Integrasi numerik dari h=0 ke h=H
        from scipy import integrate
        
        try:
            fill_time_analytical, error = integrate.quad(
                dh_to_dt, 
                0, 
                self.config.tank_height,
                limit=100,
                epsabs=1e-6
            )
            fill_time_minutes = fill_time_analytical / 60
        except Exception as e:
            fill_time_minutes = None
            error = str(e)
        
        return {
            'analytical_time_minutes': fill_time_minutes,
            'integration_error': error,
            'method': 'Integrasi Numerik Q_in(h)',
            'Cd': Cd
        }

# ====================
# 5. VISUALISASI dengan PLOTLY
# ====================

class PlotlyVisualization:
    """Kelas untuk visualisasi hasil simulasi dengan Plotly"""
    
    @staticmethod
    def plot_water_height(simulator: WaterTankSimulator, title: str = "Profil Ketinggian Air"):
        """Plot profil ketinggian air"""
        fig = go.Figure()
        
        time = simulator.time_history
        height = simulator.height_history
        config = simulator.config
        
        # Tambah garis ketinggian
        fig.add_trace(go.Scatter(
            x=time, 
            y=height,
            mode='lines',
            name='Ketinggian Air',
            line=dict(color='blue', width=3),
            hovertemplate='Waktu: %{x:.1f} menit<br>Ketinggian: %{y:.2f} m<extra></extra>'
        ))
        
        # Tambah garis referensi
        fig.add_hline(y=config.tank_height, line_dash="dash", 
                     line_color="red", opacity=0.7,
                     annotation_text=f"Tinggi Maksimum ({config.tank_height:.1f} m)")
        
        fig.add_hline(y=0, line_dash="dash", 
                     line_color="gray", opacity=0.5,
                     annotation_text="Dasar Tangki")
        
        # Tambah zona aman
        safe_level = config.tank_height * 0.2
        fig.add_hrect(y0=0, y1=safe_level,
                     fillcolor="yellow", opacity=0.1, line_width=0,
                     annotation_text="Zona Kritis")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial, sans-serif", color="darkblue")
            ),
            xaxis_title="Waktu (menit)",
            yaxis_title="Ketinggian Air (m)",
            hovermode="x unified",
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_flow_rates(simulator: WaterTankSimulator):
        """Plot laju aliran inlet dan outlet"""
        fig = go.Figure()
        
        time = simulator.time_history
        
        # Inlet flow rate
        fig.add_trace(go.Scatter(
            x=time, 
            y=simulator.inlet_rate_history,
            mode='lines',
            name='Laju Aliran Masuk',
            line=dict(color='green', width=2),
            hovertemplate='Waktu: %{x:.1f} menit<br>Q_in: %{y:.4f} m³/s'
        ))
        
        # Outlet flow rate
        fig.add_trace(go.Scatter(
            x=time, 
            y=simulator.outlet_rate_history,
            mode='lines',
            name='Laju Aliran Keluar',
            line=dict(color='red', width=2),
            hovertemplate='Waktu: %{x:.1f} menit<br>Q_in: %{y:.4f} m³/s'
        ))
        
        # Update layout
        fig.update_layout(
            title="Profil Laju Aliran",
            xaxis_title="Waktu (menit)",
            yaxis_title="Laju Aliran (m³/s)",
            hovermode="x unified",
            showlegend=True,
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_tank_status(simulator: WaterTankSimulator):
        """Plot status tangki dalam subplot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ketinggian Air vs Waktu', 
                          'Volume Air dalam Tangki',
                          'Akumulasi Volume Air', 
                          'Distribusi Laju Aliran'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        time = simulator.time_history
        config = simulator.config
        A = simulator.physics.calculate_tank_cross_section()
        
        # Plot 1: Water height
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=simulator.height_history,
                mode='lines',
                name='Ketinggian',
                line=dict(color='blue', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.2)'
            ),
            row=1, col=1
        )
        
        # Plot 2: Water volume
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=simulator.volume_history,
                mode='lines',
                name='Volume',
                line=dict(color='purple', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.2)'
            ),
            row=1, col=2
        )
        
        # Plot 3: Cumulative volume - menggunakan metode kumulatif manual
        dt = simulator.config.time_step / 60.0  # menit
        
        # Hitung volume kumulatif menggunakan integrasi trapezoid manual
        cum_in = [0]
        cum_out = [0]
        
        for i in range(1, len(time)):
            dt_actual = (time[i] - time[i-1]) * 60  # detik
            inlet_increment = (simulator.inlet_rate_history[i-1] + simulator.inlet_rate_history[i]) / 2 * dt_actual / 1000
            outlet_increment = (simulator.outlet_rate_history[i-1] + simulator.outlet_rate_history[i]) / 2 * dt_actual / 1000
            
            cum_in.append(cum_in[-1] + inlet_increment)
            cum_out.append(cum_out[-1] + outlet_increment)
        
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=cum_in,
                mode='lines',
                name='Volume Masuk Kumulatif',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=cum_out,
                mode='lines',
                name='Volume Keluar Kumulatif',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Plot 4: Flow rate distribution (bar chart)
        avg_in = np.mean(simulator.inlet_rate_history) if simulator.inlet_rate_history else 0
        avg_out = np.mean(simulator.outlet_rate_history) if simulator.outlet_rate_history else 0
        
        fig.add_trace(
            go.Bar(
                x=['Aliran Masuk', 'Aliran Keluar'],
                y=[avg_in, avg_out],
                marker=dict(color=['green', 'red']),
                text=[f'{avg_in:.2f} L/s', f'{avg_out:.2f} L/s'],
                textposition='auto',
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=1)
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=2)
        fig.update_xaxes(title_text="Waktu (menit)", row=2, col=1)
        fig.update_yaxes(title_text="Ketinggian (m)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (m³)", row=1, col=2)
        fig.update_yaxes(title_text="Volume Kumulatif (m³)", row=2, col=1)
        fig.update_yaxes(title_text="Laju Aliran (L/s)", row=2, col=2)
        
        return fig
    
    @staticmethod
    def plot_comparison_chart(simulators: List[WaterTankSimulator], 
                             labels: List[str],
                             scenario_type: str):
        """Plot perbandingan beberapa simulasi"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(f'Perbandingan Profil Ketinggian - {scenario_type}', 
                          'Perbandingan Volume Air',
                          'Perbandingan Laju Aliran Masuk', 
                          'Perbandingan Metrik Kinerja'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot 1: Height comparison
        for i, sim in enumerate(simulators):
            if sim.time_history is not None:
                fig.add_trace(
                    go.Scatter(
                        x=sim.time_history,
                        y=sim.height_history,
                        mode='lines',
                        name=labels[i],
                        line=dict(color=colors[i % len(colors)], width=2)
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Volume comparison
        for i, sim in enumerate(simulators):
            if sim.time_history is not None:
                fig.add_trace(
                    go.Scatter(
                        x=sim.time_history,
                        y=sim.volume_history,
                        mode='lines',
                        name=labels[i],
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Inlet rate comparison
        for i, sim in enumerate(simulators):
            if sim.time_history is not None:
                fig.add_trace(
                    go.Scatter(
                        x=sim.time_history,
                        y=sim.inlet_rate_history,
                        mode='lines',
                        name=labels[i],
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Metrics comparison
        metrics = ['max_height', 'min_height', 'avg_height', 'fill_percentage']
        metric_labels = ['Maks (m)', 'Min (m)', 'Rata-rata (m)', 'Terisi (%)']
        
        for i, sim in enumerate(simulators):
            if sim.results is not None:
                values = [sim.results.get(metric, 0) for metric in metrics]
                
                fig.add_trace(
                    go.Bar(
                        name=labels[i],
                        x=metric_labels,
                        y=values,
                        marker_color=colors[i % len(colors)],
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            barmode='group',
            hovermode="closest",
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=1)
        fig.update_xaxes(title_text="Waktu (menit)", row=1, col=2)
        fig.update_xaxes(title_text="Waktu (menit)", row=2, col=1)
        fig.update_yaxes(title_text="Ketinggian (m)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (m³)", row=1, col=2)
        fig.update_yaxes(title_text="Laju Aliran (L/s)", row=2, col=1)
        fig.update_yaxes(title_text="Nilai", row=2, col=2)
        
        return fig

# ====================
# 6. ANALISIS SENSITIVITAS
# ====================

class SensitivityAnalysis:
    """Analisis sensitivitas parameter"""
    
    @staticmethod
    def analyze_parameter_sensitivity(base_config: WaterTankConfig,
                                     parameter_name: str,
                                     values: List[float],
                                     inlet_active: bool = True,
                                     outlet_active: bool = True) -> Dict:
        """
        Analisis sensitivitas untuk satu parameter
        """
        results = []
        
        for value in values:
            # Create new config with modified parameter
            config = base_config.copy()
            config.update_parameter(parameter_name, value)
            
            # Run simulation with skip_metrics=True to avoid recursion
            simulator = WaterTankSimulator(config)
            simulator.run_simulation(inlet_active, outlet_active, skip_metrics=True)
            
            results.append({
                'value': value,
                'simulator': simulator,
                'metrics': simulator.results
            })
        
        return {
            'parameter': parameter_name,
            'results': results
        }

# ====================
# 7. FUNGSI TAMPILAN UNTUK HASIL WAKTU (BARU)
# ====================

def display_fill_time_results(fill_results):
    """Tampilkan hasil analisis waktu pengisian"""
    if fill_results['reached_full']:
        st.success(f"✅ Tangki mencapai penuh dalam **{fill_results['fill_time_minutes']:.1f} menit** ({fill_results['fill_time_hours']:.2f} jam)")
    else:
        st.warning(f"⚠️ Tangki belum penuh setelah {fill_results['fill_time_minutes']:.1f} menit")
        st.info(f"Ketinggian akhir: {fill_results['final_height']:.2f} m ({fill_results['fill_percentage']:.1f}%)")
    
    # Tampilkan analisis teoritis
    theo = fill_results['theoretical_analysis']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Laju Maksimum", f"{theo['Q_max_m3_per_jam']:.2f} m³/jam")
        st.metric("Tekanan Pompa", f"{theo['pump_pressure_kPa']:.1f} kPa")
    with col2:
        st.metric("Laju Minimum", f"{theo['Q_min_m3_per_jam']:.2f} m³/jam")
        st.metric("Tekanan Hidrostatik Maks", f"{theo['max_hydrostatic_kPa']:.1f} kPa")
    with col3:
        st.metric("Laju Rata-rata", f"{theo['Q_avg_m3_per_jam']:.2f} m³/jam")
        st.metric("Waktu Teoritis", f"{theo['theoretical_time_minutes']:.1f} menit")
    
    # Grafik profil laju aliran
    profile = fill_results['flow_rate_profile']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=profile['heights'], y=profile['flow_rates'],
                  mode='lines+markers', name='Laju Aliran',
                  line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=profile['heights'], y=profile['effective_pressures'],
                  mode='lines+markers', name='Tekanan Efektif',
                  line=dict(color='red', width=2, dash='dash')),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Profil Laju Aliran dan Tekanan Selama Pengisian",
        xaxis_title="Ketinggian Air (m)",
        height=400,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Laju Aliran (m³/jam)", secondary_y=False)
    fig.update_yaxes(title_text="Tekanan Efektif (kPa)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Penjelasan fisika
    st.info("""
    **Fenomena Fisika yang Terjadi:**
    - Laju aliran menurun seiring naiknya ketinggian air karena tekanan hidrostatik meningkat
    - Tekanan efektif = Tekanan pompa - Tekanan hidrostatik
    - Saat tekanan hidrostatik mendekati tekanan pompa, laju aliran mendekati nol
    - Waktu pengisian tidak linear karena laju aliran tidak konstan
    """)

def display_empty_time_results(empty_results):
    """Tampilkan hasil analisis waktu pengosongan"""
    if empty_results['reached_empty']:
        st.success(f"✅ Tangki kosong dalam **{empty_results['empty_time_minutes']:.1f} menit** ({empty_results['empty_time_hours']:.2f} jam)")
    else:
        st.warning(f"⚠️ Tangki belum kosong setelah {empty_results['empty_time_minutes']:.1f} menit")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Waktu Paruh", f"{empty_results['half_time_minutes']:.1f} menit" if empty_results['half_time_minutes'] else "N/A")
        st.metric("Koefisien Discharge", f"{empty_results['theoretical_analysis']['Cd']:.3f}")
    with col2:
        st.metric("Waktu Teoritis", f"{empty_results['theoretical_analysis']['theoretical_time_minutes']:.1f} menit")
        st.metric("Kecepatan Maks", f"{empty_results['theoretical_analysis']['max_velocity']:.2f} m/s")
    with col3:
        st.metric("Waktu Empiris", f"{empty_results['theoretical_analysis']['empirical_time_minutes']:.1f} menit")
        st.metric("Laju Rata-rata", f"{empty_results['theoretical_analysis']['avg_flow_rate_m3_per_jam']:.2f} m³/jam")
    
    # Grafik profil pengosongan
    profile = empty_results['flow_rate_profile']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=profile['heights'], y=profile['flow_rates'],
                  mode='lines+markers', name='Laju Aliran',
                  line=dict(color='red', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=profile['heights'], y=profile['velocities'],
                  mode='lines+markers', name='Kecepatan Aktual',
                  line=dict(color='orange', width=2)),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(x=profile['heights'], y=profile['theoretical_velocities'],
                  mode='lines', name='Kecepatan Teoritis',
                  line=dict(color='gray', width=2, dash='dash'),
                  opacity=0.5),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Profil Pengosongan Tangki (Hukum Torricelli)",
        xaxis_title="Ketinggian Air (m)",
        height=400,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Laju Aliran (m³/jam)", secondary_y=False)
    fig.update_yaxes(title_text="Kecepatan (m/s)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Penjelasan fisika
    st.info("""
    **Fenomena Fisika yang Terjadi:**
    - Kecepatan aliran mengikuti Hukum Torricelli: v = √(2gh)
    - Semakin rendah ketinggian air, semakin kecil kecepatan aliran
    - Koefisien discharge (Cd) < 1 akibat kerugian gesekan dan belokan pipa
    - Waktu pengosongan total = 2 × waktu paruh (untuk tangki prismatik ideal)
    """)

# ====================
# 8. APLIKASI STREAMLIT
# ====================

def create_sidebar():
    """Buat sidebar untuk input parameter"""
    st.sidebar.title("⚙️ Parameter Tangki Air")
    
    st.sidebar.subheader("Dimensi Tangki")
    tank_diameter = st.sidebar.slider("Diameter Tangki (m)", 0.5, 5.0, 2.0, 0.1)
    tank_height = st.sidebar.slider("Tinggi Tangki (m)", 1.0, 10.0, 3.0, 0.1)
    initial_height = st.sidebar.slider("Ketinggian Air Awal (m)", 0.0, tank_height, 0.5, 0.1)
    
    st.sidebar.subheader("Parameter Pipa")
    inlet_diameter = st.sidebar.slider("Diameter Pipa Inlet (cm)", 2.0, 30.0, 10.0, 1.0) / 100
    outlet_diameter = st.sidebar.slider("Diameter Pipa Outlet (cm)", 2.0, 30.0, 15.0, 1.0) / 100
    pump_pressure = st.sidebar.slider("Tekanan Pompa (bar)", 0.5, 5.0, 2.0, 0.1) * 100000
    
    st.sidebar.subheader("Kebutuhan Air Asrama")
    occupants = st.sidebar.number_input("Jumlah Penghuni", 10, 500, 100, 10)
    consumption = st.sidebar.slider("Konsumsi per Orang (L/hari)", 50, 300, 150, 10)
    
    st.sidebar.subheader("Parameter Simulasi")
    sim_time = st.sidebar.slider("Waktu Simulasi (jam)", 1, 12, 6, 1) * 3600
    
    # Advanced parameters
    with st.sidebar.expander("Parameter Lanjutan"):
        inlet_loss = st.slider("Koef. Kerugian Inlet", 0.1, 2.0, 0.5, 0.1)
        outlet_loss = st.slider("Koef. Kerugian Outlet", 0.1, 2.0, 0.8, 0.1)
        friction = st.slider("Faktor Gesekan", 0.01, 0.05, 0.02, 0.01)
    
    # Buat konfigurasi
    config = WaterTankConfig(
        tank_diameter=tank_diameter,
        tank_height=tank_height,
        initial_water_height=initial_height,
        inlet_pipe_diameter=inlet_diameter,
        outlet_pipe_diameter=outlet_diameter,
        pump_pressure=pump_pressure,
        building_occupants=occupants,
        water_consumption_per_person=consumption,
        simulation_time=sim_time,
        inlet_loss_coefficient=inlet_loss,
        outlet_loss_coefficient=outlet_loss,
        friction_factor=friction
    )
    
    return config

def display_results(simulator, results, scenario_name=""):
    """Tampilkan hasil simulasi"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ketinggian Akhir", f"{results['final_height']:.2f} m")
        st.metric("Volume Akhir", f"{results['final_volume']:.2f} m³")
    
    with col2:
        # Tampilkan waktu pengisian jika tersedia
        if results.get('time_to_fill') is not None:
            st.metric("Waktu Pengisian Penuh", f"{results['time_to_fill']:.1f} menit")
        if results.get('time_to_empty') is not None:
            st.metric("Waktu Pengosongan", f"{results['time_to_empty']:.1f} menit")
        st.metric("Persentase Terisi", f"{results['fill_percentage']:.1f}%")
    
    with col3:
        st.metric("Laju Aliran Masuk Rata-rata", f"{results['avg_inlet_rate']:.4f} m³/s")
        st.metric("Laju Aliran Keluar Rata-rata", f"{results['avg_outlet_rate']:.4f} m³/s")

    with col4:
        st.metric("Volume Masuk Total", f"{results['total_inlet_volume']:.2f} m³")
        st.metric("Volume Keluar Total", f"{results['total_outlet_volume']:.2f} m³")
        
        if results['tank_full']:
            st.success("✅ Tangki Penuh")
        elif results['tank_empty']:
            st.warning("⚠️ Tangki Kosong")
        else:
            st.info("ℹ️ Tangki dalam proses")

def main():
    """Aplikasi utama Streamlit"""
    st.set_page_config(
        page_title="Simulasi Tangki Air Asrama",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("💧 Simulasi Sistem Tangki Air Asrama")
    st.markdown("""
    Aplikasi ini mensimulasikan sistem tangki air untuk kebutuhan asrama. 
    Anda dapat menganalisis waktu pengisian, pengosongan, dan profil ketinggian air 
    dalam berbagai skenario operasi.
    """)
    
    # Sidebar untuk input parameter
    config = create_sidebar()
    
    # Pilihan skenario
    st.subheader("🎮 Pilih Skenario Simulasi")
    scenario = st.radio(
        "Mode Operasi:",
        ["Pengisian Saja", "Pengosongan Saja", "Pengisian & Pengosongan Bersamaan"],
        horizontal=True
    )
    
    # Tentukan status inlet/outlet berdasarkan skenario
    if scenario == "Pengisian Saja":
        inlet_active, outlet_active = True, False
        scenario_desc = "Hanya Pengisian"
    elif scenario == "Pengosongan Saja":
        inlet_active, outlet_active = False, True
        scenario_desc = "Hanya Pengosongan"
    else:
        inlet_active, outlet_active = True, True
        scenario_desc = "Pengisian & Pengosongan Bersamaan"
    
    # Jalankan simulasi utama
    with st.spinner("Menjalankan simulasi..."):
        simulator = WaterTankSimulator(config)
        results = simulator.run_simulation(inlet_active, outlet_active, skip_metrics=False)
    
    # Tampilkan hasil
    st.success("✅ Simulasi selesai!")
    display_results(simulator, results, scenario_desc)
    
    # Tab untuk visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Profil Ketinggian", 
        "📊 Analisis Aliran", 
        "🕒 Waktu Pengisian/Pengosongan",
        "🔍 Analisis Sensitivitas",
        "📋 Data & Optimalisasi"
    ])
    
    with tab1:
        st.subheader(f"Profil Ketinggian Air - {scenario_desc}")
        fig_height = PlotlyVisualization.plot_water_height(simulator)
        st.plotly_chart(fig_height, use_container_width=True)
        
        # Informasi tambahan
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Informasi Tangki:**
            - Volume Total: {config.tank_volume:.2f} m³
            - Kapasitas: {config.tank_volume * 1000:.0f} Liter
            - Ketinggian Maksimum: {config.tank_height:.2f} m
            - Ketinggian Saat Ini: {results['final_height']:.2f} m
            """)
        
        with col2:
            st.info(f"""
            **Kebutuhan Air Asrama:**
            - Jumlah Penghuni: {config.building_occupants} orang
            - Konsumsi per Orang: {config.water_consumption_per_person} L/hari
            - Total Kebutuhan: {config.daily_water_demand:.2f} m³/hari
            - Kebutuhan per Jam: {config.daily_water_demand/24:.2f} m³/jam
            """)
    
    with tab2:
        st.subheader("Analisis Laju Aliran")
        
        # Plot flow rates
        fig_flow = PlotlyVisualization.plot_flow_rates(simulator)
        st.plotly_chart(fig_flow, use_container_width=True)
        
        # Status tangki
        fig_status = PlotlyVisualization.plot_tank_status(simulator)
        st.plotly_chart(fig_status, use_container_width=True)
    
    with tab3:
        st.subheader("Analisis Waktu Pengisian dan Pengosongan")
        
        st.markdown("""
        **Penjelasan Fisika:**
        - Waktu pengisian tidak konstan karena tekanan hidrostatik meningkat seiring ketinggian air
        - Laju aliran masuk menurun saat tangki semakin penuh (tekanan pompa melawan tekanan hidrostatik)
        - Laju aliran keluar mengikuti hukum Torricelli: v = √(2gh), dipengaruhi ketinggian air
        - Koefisien kerugian (loss coefficient) memperhitungkan gesekan dan belokan pipa
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**⏱️ Waktu Pengisian (dari kosong ke penuh)**")
            if st.button("Hitung Waktu Pengisian Detail", key="btn_fill_detail"):
                with st.spinner("Menghitung waktu pengisian dengan model hidrostatik..."):
                    fill_results = simulator.calculate_fill_time()
                    display_fill_time_results(fill_results)
                    
                    # Tampilkan grafik pengisian
                    fill_config = config.copy()
                    fill_config.initial_water_height = 0.0
                    fill_sim = WaterTankSimulator(fill_config)
                    fill_sim.run_simulation(inlet_active=True, outlet_active=False, skip_metrics=True)
                    
                    fig_fill = PlotlyVisualization.plot_water_height(fill_sim, "Profil Pengisian Tangki (Realistis)")
                    st.plotly_chart(fig_fill, use_container_width=True)
        
        with col2:
            st.info("**⏱️ Waktu Pengosongan (dari penuh ke kosong)**")
            if st.button("Hitung Waktu Pengosongan Detail", key="btn_empty_detail"):
                with st.spinner("Menghitung waktu pengosongan dengan model Torricelli..."):
                    empty_results = simulator.calculate_empty_time()
                    display_empty_time_results(empty_results)
                    
                    # Tampilkan grafik pengosongan
                    empty_config = config.copy()
                    empty_config.initial_water_height = config.tank_height
                    empty_sim = WaterTankSimulator(empty_config)
                    empty_sim.run_simulation(inlet_active=False, outlet_active=True, skip_metrics=True)
                    
                    fig_empty = PlotlyVisualization.plot_water_height(empty_sim, "Profil Pengosongan Tangki (Realistis)")
                    st.plotly_chart(fig_empty, use_container_width=True)
        
        # Tambahkan validasi analitik
        st.subheader("📐 Validasi dengan Rumus Analitik")
        if st.button("Validasi dengan Metode Analitik"):
            validation = simulator.validate_fill_time_analytical()
            if validation['analytical_time_minutes']:
                st.info(f"""
                **Hasil Validasi:**
                - Waktu pengisian (integrasi numerik): {validation['analytical_time_minutes']:.2f} menit
                - Error integrasi: {validation['integration_error']:.2e}
                - Koefisien discharge inlet: {validation['Cd']:.3f}
                - Metode: {validation['method']}
                
                **Interpretasi:**
                Hasil ini memvalidasi bahwa model numerik yang digunakan konsisten dengan 
                persamaan diferensial analitik untuk sistem tangki prismatik.
                """)
            else:
                st.error("Validasi gagal - mungkin parameter di luar batas fisik")
    
    with tab4:
        st.subheader("Analisis Sensitivitas Parameter")
        
        # Pilih parameter untuk analisis sensitivitas
        param_options = {
            "Diameter Tangki": "tank_diameter",
            "Tinggi Tangki": "tank_height",
            "Diameter Pipa Inlet": "inlet_pipe_diameter",
            "Diameter Pipa Outlet": "outlet_pipe_diameter",
            "Tekanan Pompa": "pump_pressure"
        }
        
        selected_param = st.selectbox(
            "Pilih parameter untuk analisis sensitivitas:",
            list(param_options.keys())
        )
        
        param_name = param_options[selected_param]
        
        # Buat range nilai berdasarkan parameter
        current_val = getattr(config, param_name)
        
        if param_name in ["inlet_pipe_diameter", "outlet_pipe_diameter"]:
            # Untuk diameter pipa, buat range dalam cm
            current_cm = current_val * 100
            values_cm = [current_cm * 0.5, current_cm * 0.75, current_cm, 
                        current_cm * 1.25, current_cm * 1.5]
            values = [v/100 for v in values_cm]
            display_values = [f"{v:.1f} cm" for v in values_cm]
        elif param_name == "pump_pressure":
            # Untuk tekanan pompa, buat range dalam bar
            current_bar = current_val / 100000
            values_bar = [current_bar * 0.5, current_bar * 0.75, current_bar,
                         current_bar * 1.25, current_bar * 1.5]
            values = [v * 100000 for v in values_bar]
            display_values = [f"{v:.2f} bar" for v in values_bar]
        else:
            # Untuk parameter lainnya
            values = [current_val * 0.5, current_val * 0.75, current_val,
                     current_val * 1.25, current_val * 1.5]
            display_values = [f"{v:.2f} m" for v in values]
        
        # Jalankan analisis
        if st.button("Jalankan Analisis Sensitivitas", type="primary"):
            with st.spinner(f"Menjalankan analisis untuk {selected_param}..."):
                analysis = SensitivityAnalysis.analyze_parameter_sensitivity(
                    config, param_name, values, inlet_active, outlet_active
                )
                
                # Buat dataframe hasil
                analysis_data = []
                for i, result in enumerate(analysis['results']):
                    analysis_data.append({
                        'Nilai': display_values[i],
                        'Ketinggian Akhir (m)': result['metrics']['final_height'],
                        'Volume Akhir (m³)': result['metrics']['final_volume'],
                        'Laju Inlet Rata-rata (L/s)': result['metrics']['avg_inlet_rate'],
                        'Laju Outlet Rata-rata (L/s)': result['metrics']['avg_outlet_rate'],
                        'Persentase Terisi (%)': result['metrics']['fill_percentage']
                    })
                
                df_analysis = pd.DataFrame(analysis_data)
                st.dataframe(df_analysis.style.format({
                    'Ketinggian Akhir (m)': '{:.2f}',
                    'Volume Akhir (m³)': '{:.2f}',
                    'Laju Inlet Rata-rata (L/s)': '{:.2f}',
                    'Laju Outlet Rata-rata (L/s)': '{:.2f}',
                    'Persentase Terisi (%)': '{:.1f}'
                }), use_container_width=True)
                
                # Buat grafik sensitivitas
                fig_sens = go.Figure()
                
                metrics_to_plot = [
                    ('Ketinggian Akhir (m)', 'blue'),
                    ('Volume Akhir (m³)', 'green'),
                    ('Laju Inlet Rata-rata (L/s)', 'red')
                ]
                
                for metric, color in metrics_to_plot:
                    fig_sens.add_trace(go.Scatter(
                        x=list(range(len(values))),
                        y=[d[metric] for d in analysis_data],
                        mode='lines+markers',
                        name=metric,
                        line=dict(color=color, width=2)
                    ))
                
                fig_sens.update_layout(
                    title=f"Sensitivitas {selected_param}",
                    xaxis=dict(
                        title=selected_param,
                        ticktext=display_values,
                        tickvals=list(range(len(display_values)))
                    ),
                    yaxis_title="Nilai Metrik",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)
    
    with tab5:
        st.subheader("Data Simulasi dan Optimalisasi Tangki")
        
        # Data simulasi
        data = {
            'Waktu (menit)': simulator.time_history,
            'Ketinggian (m)': simulator.height_history,
            'Volume (m³)': simulator.volume_history,
            'Laju Inlet (L/s)': simulator.inlet_rate_history,
            'Laju Outlet (L/s)': simulator.outlet_rate_history
        }
        
        df = pd.DataFrame(data)
        
        st.dataframe(df.style.format({
            'Waktu (menit)': '{:.1f}',
            'Ketinggian (m)': '{:.2f}',
            'Volume (m³)': '{:.2f}',
            'Laju Inlet (L/s)': '{:.2f}',
            'Laju Outlet (L/s)': '{:.2f}'
        }), use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name="data_tangki_air.csv",
            mime="text/csv"
        )
        
        # Optimalisasi ukuran tangki
        st.subheader("🎯 Optimalisasi Ukuran Tangki")
        
        physics = WaterTankPhysics(config)
        optimal = physics.calculate_optimal_tank_size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Ukuran Tangki Saat Ini:**
            - Diameter: {optimal['current_diameter']:.2f} m
            - Tinggi: {optimal['current_height']:.2f} m
            - Volume: {optimal['current_volume']:.2f} m³ ({optimal['current_volume'] * 1000:.0f} Liter)
            """)
        
        with col2:
            st.success(f"""
            **Ukuran Tangki Optimal (Rekomendasi):**
            - Diameter: {optimal['optimal_diameter']:.2f} m
            - Tinggi: {optimal['optimal_height']:.2f} m
            - Volume: {optimal['optimal_volume']:.2f} m³ ({optimal['optimal_volume'] * 1000:.0f} Liter)
            """)
        
        # Evaluasi kecukupan
        if config.tank_volume >= optimal['min_volume']:
            st.success(f"✅ Kapasitas tangki saat ini ({config.tank_volume:.2f} m³) mencukupi untuk kebutuhan ({optimal['min_volume']:.2f} m³)")
        else:
            st.warning(f"⚠️ Kapasitas tangki saat ini ({config.tank_volume:.2f} m³) kurang dari kebutuhan minimal ({optimal['min_volume']:.2f} m³)")
        
        # Rekomendasi jadwal pengisian
        st.subheader("📅 Rekomendasi Jadwal Pengisian")
        
        # Hitung waktu operasi pompa yang diperlukan
        required_daily_volume = config.daily_water_demand
        # Gunakan ketinggian rata-rata untuk estimasi kapasitas pompa
        avg_inlet_rate = physics.calculate_inlet_flow_rate(config.tank_height/2) * 3600  # m³/jam
        
        if avg_inlet_rate > 0:
            pump_hours_needed = required_daily_volume / avg_inlet_rate
            
            st.info(f"""
            **Analisis Kebutuhan Pompa:**
            - Kebutuhan Air Harian: {required_daily_volume:.2f} m³
            - Kapasitas Pompa Rata-rata: {avg_inlet_rate:.2f} m³/jam
            - Waktu Operasi Pompa yang Diperlukan: {pump_hours_needed:.1f} jam/hari
            
            **Rekomendasi Jadwal:**
            - Sesi 1: {pump_hours_needed/2:.1f} jam (pagi)
            - Sesi 2: {pump_hours_needed/2:.1f} jam (sore)
            
            **Catatan:** Waktu operasi ini sudah mempertimbangkan penurunan efisiensi
            akibat tekanan hidrostatik saat tangki hampir penuh.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Catatan:** Simulasi ini menggunakan model fisika yang mempertimbangkan:
    - Tekanan hidrostatik yang berubah terhadap ketinggian air
    - Hukum Torricelli untuk aliran keluar
    - Koefisien kerugian akibat gesekan dan belokan pipa
    - Perubahan laju aliran selama proses pengisian/pengosongan
    
    Hasil simulasi ini lebih realistis dibandingkan model dengan laju aliran konstan.
    """)

if __name__ == "__main__":
    main()