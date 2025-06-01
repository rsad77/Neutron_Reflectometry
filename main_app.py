import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QTableWidget, QTableWidgetItem, QFileDialog, QGridLayout, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.integrate import trapezoid
from smoothing import smooth_data
from material_database import MATERIALS
from structure_generator import generate_periodic_with_defects, generate_fibonacci_with_defects
from lemur import lemur_function_1
from export_analysis import export_to_excel, integrate_reflectivity, find_peaks_analysis
u_norm=2e-6 * 1e20 * 4 * np.pi

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


    def plot(self, wavelengths, reflectivity):
        self.ax.clear()
        self.ax.semilogy(wavelengths, reflectivity)
        self.ax.set_xlabel('Wavelength (Å)')
        self.ax.set_ylabel('Reflectivity')
        self.ax.set_title('Neutron Reflectivity')
        self.ax.grid(True, which="both", ls="--")
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Reflectivity Calculator")
        self.setGeometry(100, 100, 1000, 600)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Input tab
        self.input_tab = QWidget()
        self.tabs.addTab(self.input_tab, "Input Parameters")
        self.setup_input_tab()

        # Results tab
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Results")
        self.setup_results_tab()

        # Initialize data
        self.wavelengths = []
        self.reflectivity = []
        self.peak_data = []
        self.integral_value = 0.0

        self.smoothed_reflectivity = None


    def setup_input_tab(self):
        layout = QVBoxLayout(self.input_tab)

        # Structure type selection
        group_structure = QGroupBox("Structure Type")
        layout_structure = QHBoxLayout(group_structure)

        self.structure_type = QComboBox()
        self.structure_type.addItems(["Periodic", "Fibonacci"])
        layout_structure.addWidget(QLabel("Structure:"))
        layout_structure.addWidget(self.structure_type)

        self.num_periods = QSpinBox()
        self.num_periods.setRange(1, 250)
        self.num_periods.setValue(77)
        layout_structure.addWidget(QLabel("Number of Periods:"))
        layout_structure.addWidget(self.num_periods)

        self.fibonacci_order = QSpinBox()
        self.fibonacci_order.setRange(1, 20)
        self.fibonacci_order.setValue(11)
        layout_structure.addWidget(QLabel("Fibonacci Order:"))
        layout_structure.addWidget(self.fibonacci_order)

        layout.addWidget(group_structure)

        # Layer thicknesses
        group_thickness = QGroupBox("Layer Thicknesses (Å)")
        layout_thickness = QHBoxLayout(group_thickness)

        self.thickness_A = QDoubleSpinBox()
        self.thickness_A.setRange(1, 1000)
        self.thickness_A.setValue(48.5)
        layout_thickness.addWidget(QLabel("Thickness A:"))
        layout_thickness.addWidget(self.thickness_A)

        self.thickness_B = QDoubleSpinBox()
        self.thickness_B.setRange(1, 1000)
        self.thickness_B.setValue(30)
        layout_thickness.addWidget(QLabel("Thickness B:"))
        layout_thickness.addWidget(self.thickness_B)

        self.thickness_coating = QDoubleSpinBox()
        self.thickness_coating.setRange(0, 1000)
        self.thickness_coating.setValue(100)
        layout_thickness.addWidget(QLabel("Coating Thickness:"))
        layout_thickness.addWidget(self.thickness_coating)

        self.thickness_pre = QDoubleSpinBox()
        self.thickness_pre.setRange(0, 1000)
        self.thickness_pre.setValue(400)
        layout_thickness.addWidget(QLabel("Pre-layer Thickness:"))
        layout_thickness.addWidget(self.thickness_pre)

        layout.addWidget(group_thickness)

        # Material selection
        group_materials = QGroupBox("Material Selection")
        layout_materials = QHBoxLayout(group_materials)

        self.material_A = QComboBox()
        self.material_B = QComboBox()
        self.material_coating = QComboBox()
        self.material_pre = QComboBox()
        self.material_substrate = QComboBox()

        materials_list = list(MATERIALS.keys())
        self.material_A.addItems(materials_list)
        self.material_B.addItems(materials_list)
        self.material_coating.addItems(materials_list)
        self.material_pre.addItems(materials_list)
        self.material_substrate.addItems(materials_list)
        # Установка значений по умолчанию
        self.material_coating.setCurrentText('Nb')
        self.material_pre.setCurrentText('Nb')
        self.material_substrate.setCurrentText('Al2O3')
        self.material_A.setCurrentText('Ni')
        self.material_B.setCurrentText('Zr')

        layout_materials.addWidget(QLabel("Material A:"))
        layout_materials.addWidget(self.material_A)
        layout_materials.addWidget(QLabel("Material B:"))
        layout_materials.addWidget(self.material_B)
        layout_materials.addWidget(QLabel("Coating:"))
        layout_materials.addWidget(self.material_coating)
        layout_materials.addWidget(QLabel("Pre-layer:"))
        layout_materials.addWidget(self.material_pre)
        layout_materials.addWidget(QLabel("Substrate:"))
        layout_materials.addWidget(self.material_substrate)

        layout.addWidget(group_materials)

        # Add this to the setup_input_tab method, after the material selection group
        # Magnetic field group
        group_magnetic = QGroupBox("Magnetic Field Parameters")
        layout_magnetic = QGridLayout(group_magnetic)

        self.H_x = QDoubleSpinBox()
        self.H_x.setRange(-100, 100)
        self.H_x.setValue(0.0)
        layout_magnetic.addWidget(QLabel("H_x (T):"), 0, 0)
        layout_magnetic.addWidget(self.H_x, 0, 1)

        self.H_y = QDoubleSpinBox()
        self.H_y.setRange(-100, 100)
        self.H_y.setValue(0.0)
        layout_magnetic.addWidget(QLabel("H_y (T):"), 1, 0)
        layout_magnetic.addWidget(self.H_y, 1, 1)

        self.H_z = QDoubleSpinBox()
        self.H_z.setRange(-100, 100)
        self.H_z.setValue(0.000000001)
        layout_magnetic.addWidget(QLabel("H_z (T):"), 2, 0)
        layout_magnetic.addWidget(self.H_z, 2, 1)

        self.theta = QDoubleSpinBox()
        self.theta.setRange(0, 90)
        self.theta.setValue(5e-3)
        self.theta.setDecimals(6)
        layout_magnetic.addWidget(QLabel("Theta (rad):"), 3, 0)
        layout_magnetic.addWidget(self.theta, 3, 1)

        # Normalization constants
        self.u_norm = 2e-6 * 1e20 * 4 * np.pi
        self.d_norm = (2 * np.pi) / np.sqrt(self.u_norm) * 1e10
        self.M_norm = self.u_norm / (1e-4 * 1.674927471e-27 * 1.91304272 * 5.050783699e-27 / (1.054571800e-34) ** 2)

        layout.addWidget(group_magnetic)

        # Wavelength range
        group_wavelength = QGroupBox("Wavelength Range (Å)")
        layout_wavelength = QHBoxLayout(group_wavelength)

        self.wavelength_min = QDoubleSpinBox()
        self.wavelength_min.setRange(0.001, 10.0)
        self.wavelength_min.setValue(0.001)
        layout_wavelength.addWidget(QLabel("Min:"))
        layout_wavelength.addWidget(self.wavelength_min)
        self.wavelength_min.setDecimals(3)
        self.wavelength_min.setSingleStep(0.001)

        self.wavelength_max = QDoubleSpinBox()
        self.wavelength_max.setRange(0.000001, 10.0)
        self.wavelength_max.setValue(5)
        layout_wavelength.addWidget(QLabel("Max:"))
        layout_wavelength.addWidget(self.wavelength_max)
        self.wavelength_max.setDecimals(3)
        self.wavelength_max.setSingleStep(0.001)

        self.wavelength_step = QDoubleSpinBox()
        self.wavelength_step.setRange(0.000000001, 1)
        self.wavelength_step.setValue(0.01)
        layout_wavelength.addWidget(QLabel("Step:"))
        layout_wavelength.addWidget(self.wavelength_step)
        self.wavelength_step.setDecimals(9)
        self.wavelength_step.setSingleStep(0.000000001)

        layout.addWidget(group_wavelength)

        # Defect parameters
        group_defects = QGroupBox("Layer Defects Simulation")
        layout_defects = QGridLayout(group_defects)

        # Defect probability
        self.defect_prob = QDoubleSpinBox()
        self.defect_prob.setRange(0, 1)
        self.defect_prob.setValue(0.0)
        self.defect_prob.setSingleStep(0.01)
        layout_defects.addWidget(QLabel("Layer Skip Probability:"), 0, 0)
        layout_defects.addWidget(self.defect_prob, 0, 1)

        # Thickness variation
        self.thickness_dev_A = QDoubleSpinBox()
        self.thickness_dev_A.setRange(0, 100)
        self.thickness_dev_A.setValue(0.0)
        self.thickness_dev_A.setSingleStep(1)
        layout_defects.addWidget(QLabel("Thickness Deviation A (Å):"), 1, 0)
        layout_defects.addWidget(self.thickness_dev_A, 1, 1)

        self.thickness_dev_B = QDoubleSpinBox()
        self.thickness_dev_B.setRange(0, 100)
        self.thickness_dev_B.setValue(0.0)
        self.thickness_dev_B.setSingleStep(1)
        layout_defects.addWidget(QLabel("Thickness Deviation B (Å):"), 2, 0)
        layout_defects.addWidget(self.thickness_dev_B, 2, 1)

        # Random seed
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 1000000)
        self.random_seed.setValue(42)
        layout_defects.addWidget(QLabel("Random Seed:"), 3, 0)
        layout_defects.addWidget(self.random_seed, 3, 1)

        layout.addWidget(group_defects)

        # Replacement parameters
        layout_defects.addWidget(QLabel("A->A' Probability:"), 4, 0)
        self.replace_prob_A = QDoubleSpinBox()
        self.replace_prob_A.setRange(0, 1)
        self.replace_prob_A.setValue(0.0)
        self.replace_prob_A.setSingleStep(0.01)
        layout_defects.addWidget(self.replace_prob_A, 4, 1)

        layout_defects.addWidget(QLabel("B->B' Probability:"), 5, 0)
        self.replace_prob_B = QDoubleSpinBox()
        self.replace_prob_B.setRange(0, 1)
        self.replace_prob_B.setValue(0.0)
        self.replace_prob_B.setSingleStep(0.01)
        layout_defects.addWidget(self.replace_prob_B, 5, 1)

        # Alternative materials
        layout_defects.addWidget(QLabel("Material A':"), 6, 0)
        self.material_A_alt = QComboBox()
        self.material_A_alt.addItems(materials_list)
        self.material_A_alt.setCurrentText('NiH')  # Default value
        layout_defects.addWidget(self.material_A_alt, 6, 1)

        layout_defects.addWidget(QLabel("Material B':"), 7, 0)
        self.material_B_alt = QComboBox()
        self.material_B_alt.addItems(materials_list)
        self.material_B_alt.setCurrentText('ZrH')  # Default value
        layout_defects.addWidget(self.material_B_alt, 7, 1)

        # Buttons
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate)
        layout.addWidget(self.calculate_button)

        group_smoothing = QGroupBox("Smoothing")
        layout_smoothing = QHBoxLayout(group_smoothing)

        self.smoothing_enabled = QCheckBox("Enable Smoothing")
        self.smoothing_enabled.setChecked(False)

        self.smoothing_method = QComboBox()
        self.smoothing_method.addItems(['savgol', 'moving_average'])
        self.smoothing_method.setCurrentText('savgol')

        self.smoothing_window = QSpinBox()
        self.smoothing_window.setRange(3, 51)
        self.smoothing_window.setValue(5)
        self.smoothing_window.setSingleStep(2)

        self.smoothing_order = QSpinBox()
        self.smoothing_order.setRange(1, 5)
        self.smoothing_order.setValue(2)
        self.smoothing_order.setEnabled(False)  # Only for savgol

        # Connect method change to order widget
        self.smoothing_method.currentTextChanged.connect(
            lambda method: self.smoothing_order.setEnabled(method == 'savgol')
        )

        layout_smoothing.addWidget(self.smoothing_enabled)
        layout_smoothing.addWidget(QLabel("Method:"))
        layout_smoothing.addWidget(self.smoothing_method)
        layout_smoothing.addWidget(QLabel("Window:"))
        layout_smoothing.addWidget(self.smoothing_window)
        layout_smoothing.addWidget(QLabel("Order:"))
        layout_smoothing.addWidget(self.smoothing_order)

        layout.addWidget(group_smoothing)

    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)

        # Plot
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget, 3)

        # Controls
        controls_layout = QHBoxLayout()

        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_data)
        controls_layout.addWidget(self.export_button)

        self.analyze_button = QPushButton("Analyze Peaks")
        self.analyze_button.clicked.connect(self.analyze_peaks)
        controls_layout.addWidget(self.analyze_button)

        self.integrate_button = QPushButton("Integrate")
        self.integrate_button.clicked.connect(self.integrate)
        controls_layout.addWidget(self.integrate_button)

        layout.addLayout(controls_layout)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Wavelength (Å)", "Intensity", "Width"])
        layout.addWidget(self.results_table, 1)

        # Integral result
        self.integral_label = QLabel("Integral: 0.0")
        self.integral_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.integral_label)

    def calculate(self):
        # Get parameters
        structure_type = self.structure_type.currentText()
        num_periods = self.num_periods.value()
        fib_order = self.fibonacci_order.value()
        dA = self.thickness_A.value()
        dB = self.thickness_B.value()
        d_coating = self.thickness_coating.value()
        d_pre = self.thickness_pre.value()

        # Get defect parameters
        defect_prob = self.defect_prob.value()
        dev_A = self.thickness_dev_A.value()
        dev_B = self.thickness_dev_B.value()
        replace_prob_A = self.replace_prob_A.value()
        replace_prob_B = self.replace_prob_B.value()
        seed = self.random_seed.value()

        # Get materials
        mat_A = MATERIALS[self.material_A.currentText()]
        mat_B = MATERIALS[self.material_B.currentText()]
        mat_A_alt = MATERIALS[self.material_A_alt.currentText()]
        mat_B_alt = MATERIALS[self.material_B_alt.currentText()]
        mat_coating = MATERIALS[self.material_coating.currentText()]
        mat_pre = MATERIALS[self.material_pre.currentText()]
        mat_substrate = MATERIALS[self.material_substrate.currentText()]

        # Build layer structure
        d_list = []
        u_list = []
        M_x_list = []
        M_y_list = []
        M_z_list = []

        # Add coating layer
        d_list.append(d_coating)
        u_list.append(mat_coating.u_real * 1e-6 + 1j * mat_coating.u_imag * 1e-6)
        M_x_list.append(mat_coating.M_x)
        M_y_list.append(mat_coating.M_y)
        M_z_list.append(mat_coating.M_z)

        # Generate main structure
        if structure_type == "Periodic":
            structure = generate_periodic_with_defects(
                dA, dB, num_periods, defect_prob, dev_A, dev_B, seed,
                replace_prob_A, replace_prob_B
            )
        else:
            structure = generate_fibonacci_with_defects(
                fib_order, dA, dB, defect_prob, dev_A, dev_B, seed,
                replace_prob_A, replace_prob_B
            )

        self.current_structure = structure

        # Add main structure layers
        for layer_type, thickness in structure:
            d_list.append(thickness)
            if layer_type == 'A':
                material = mat_A
            elif layer_type == 'A_alt':
                material = mat_A_alt
            elif layer_type == 'B':
                material = mat_B
            elif layer_type == 'B_alt':
                material = mat_B_alt

            u_list.append(material.u_real * 1e-6 + 1j * material.u_imag * 1e-6)
            M_x_list.append(material.M_x)
            M_y_list.append(material.M_y)
            M_z_list.append(material.M_z)

        # Add pre-layer
        d_list.append(d_pre)
        u_list.append(mat_pre.u_real * 1e-6 + 1j * mat_pre.u_imag * 1e-6)
        M_x_list.append(mat_pre.M_x)
        M_y_list.append(mat_pre.M_y)
        M_z_list.append(mat_pre.M_z)

        # Add substrate (thickness doesn't matter for substrate)
        d_list.append(1e10)  # Large number for semi-infinite substrate
        u_list.append(mat_substrate.u_real * 1e-6 + 1j * mat_substrate.u_imag * 1e-6)
        M_x_list.append(mat_substrate.M_x)
        M_y_list.append(mat_substrate.M_y)
        M_z_list.append(mat_substrate.M_z)

        # Convert to numpy arrays
        d_array = np.array(d_list)
        u_array = np.array(u_list)
        M_x_array = np.array(M_x_list)
        M_y_array = np.array(M_y_list)
        M_z_array = np.array(M_z_list)

        # Normalization constants from дополнения.txt
        u_norm = 2e-6 * 1e20 * 4 * np.pi
        d_norm = (2 * np.pi) / np.sqrt(u_norm) * 1e10
        M_norm = u_norm / (1e-4 * 1.674927471e-27 * 1.91304272 * 5.050783699e-27 / (1.054571800e-34) ** 2)

        #Normalize parameters
        u_normalized = u_array * 1e20 * 4 * np.pi/u_norm
        d_normalized = d_array/d_norm
        M_x_normalized = M_x_array/M_norm
        M_y_normalized = M_y_array/M_norm
        M_z_normalized = M_z_array/M_norm

       # External magnetic field (normalized)
        H_x = self.H_x.value() / M_norm
        H_y = self.H_y.value() / M_norm
        H_z = self.H_z.value() / M_norm
        H = np.sqrt(H_x ** 2 + H_y ** 2 + H_z ** 2)

        # Total B field (B = H + M)
        B_x = H_x + M_x_normalized
        B_y = H_y + M_y_normalized
        B_z = H_z + M_z_normalized
        B = np.sqrt(B_x ** 2 + B_y ** 2 + B_z ** 2)

        # Calculate reflectivity
        theta = self.theta.value()
        wavelengths = np.linspace(self.wavelength_min.value(),
                                  self.wavelength_max.value(),
                                  int((self.wavelength_max.value() - self.wavelength_min.value()) / self.wavelength_step.value()) + 1)

        reflectivity = np.zeros_like(wavelengths, dtype=float)

        for i, wl in enumerate(wavelengths):
            k0 = 2 * np.pi * np.sin(theta) / (wl * np.sqrt(u_norm)) * 1e10
            try:
                # Get the reflectivity value
                reflectivity[i] = np.real(lemur_function_1(
                    d_normalized,
                    u_normalized,
                    B, B_x, B_y, B_z,
                    H, H_x, H_y, H_z,
                    k0
                ))
            except Exception as e:
                print(f"Error at wavelength {wl}: {str(e)}")
                reflectivity[i] = 0  # Default value if calculation fails

            # Ensure we have valid finite values
        reflectivity = np.nan_to_num(reflectivity, nan=0.0, posinf=0.0, neginf=0.0)

        self.wavelengths = wavelengths
        self.reflectivity = reflectivity

        # Plot results
        if len(wavelengths) == len(reflectivity):
            self.plot_widget.plot(wavelengths, reflectivity)
        else:
            print(f"Dimension mismatch: wavelengths ({len(wavelengths)}), reflectivity ({len(reflectivity)})")

        # Calculate integral and peaks
            self.integral_value = trapezoid(reflectivity, wavelengths)
            self.peak_data = find_peaks_analysis(wavelengths, reflectivity)

        # Update UI
            self.integral_label.setText(f"Integral: {self.integral_value:.6e}")
            self.update_peaks_table()

        # Print structure info
        print(f"Generated structure with {len(structure)} layers")
        for i, (layer_type, thickness) in enumerate(structure):
            material = layer_type.replace('_alt', "'")
            print(f"Layer {i + 1}: {material}, thickness = {thickness:.2f} Å")
            # Применение сглаживания если включено
            if self.smoothing_enabled.isChecked():
                self.smoothed_reflectivity = smooth_data(
                    reflectivity,
                    window_size=self.smoothing_window.value(),
                    polynomial_order=self.smoothing_order.value(),
                    method=self.smoothing_method.currentText()
                )
                print("Normalized u:", u_normalized)
                print("Normalized d:", d_normalized)
                print("Normalized M:", M_x_normalized, M_y_normalized, M_z_normalized)
                print("First k0 value:", 2 * np.pi * np.sin(theta) / (wavelengths[0] * np.sqrt(u_norm)) * 1e10)
                print("First reflectivity value:", reflectivity[0])

                # Построение графиков
                self.plot_widget.ax.clear()
                self.plot_widget.ax.semilogy(wavelengths, reflectivity,
                                             label="Original", alpha=0.7)
                self.plot_widget.ax.semilogy(wavelengths, self.smoothed_reflectivity,
                                             label="Smoothed", linewidth=2)
                self.plot_widget.ax.legend()
                self.plot_widget.ax.set_xlabel('Wavelength (Å)')
                self.plot_widget.ax.set_ylabel('Reflectivity')
                self.plot_widget.ax.set_title('Neutron Reflectivity')
                self.plot_widget.ax.grid(True, which="both", ls="--")
                self.plot_widget.canvas.draw()
            else:
                # Построение только оригинальных данных
                self.plot_widget.plot(wavelengths, reflectivity)

    def export_data(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Excel File", "", "Excel Files (*.xlsx)"
        )
        if filename:
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'

            # Check if we have smoothed data to export
            smoothed_reflectivity = None
            if hasattr(self, 'smoothed_reflectivity') and self.smoothed_reflectivity is not None:
                smoothed_reflectivity = self.smoothed_reflectivity

            integral, peaks = export_to_excel(
                self.wavelengths,
                self.reflectivity,
                self.current_structure,
                filename,
                smoothed_reflectivity=smoothed_reflectivity  # Pass explicitly
            )
            self.integral_label.setText(f"Integral: {integral:.6e}")
            self.peak_data = peaks
            self.update_peaks_table()

    def update_peaks_table(self):
        """Update peaks table with analyzed data"""
        self.results_table.setRowCount(len(self.peak_data))
        for i, peak in enumerate(self.peak_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{peak['Wavelength']:.3f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{peak['Intensity']:.4e}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{peak.get('Width', 0):.3f}"))

    def analyze_peaks(self):
        self.peak_data = find_peaks_analysis(
            self.wavelengths, self.reflectivity
        )
        self.results_table.setRowCount(len(self.peak_data))
        for i, peak in enumerate(self.peak_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{peak['Wavelength']:.3f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{peak['Intensity']:.4e}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{peak['Width']:.4f}"))

    def integrate(self):
        self.integral_value = integrate_reflectivity(
            self.wavelengths, self.reflectivity
        )
        self.integral_label.setText(f"Integral: {self.integral_value:.6e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
