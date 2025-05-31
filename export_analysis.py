import pandas as pd
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.integrate import trapezoid


def find_peaks_analysis(wavelengths, reflectivity, height=0.01, distance=10):
    """Find peaks in reflectivity data"""
    peaks, properties = find_peaks(reflectivity, height=height, distance=distance)
    peak_data = []
    for i in peaks:
        peak_data.append({
            'Wavelength': wavelengths[i],
            'Intensity': reflectivity[i],
            'Width': properties['widths'][0] if 'widths' in properties else 0
        })
    return peak_data


def integrate_reflectivity(wavelengths, reflectivity):
    """Calculate integral of reflectivity curve"""
    return trapezoid(reflectivity, wavelengths)


def calculate_thickness_stats(structure):
    """Calculate thickness statistics for the structure"""
    stats = {}
    material_thickness = defaultdict(float)
    total_thickness = 0.0

    for layer_type, thickness in structure:
        material = layer_type.replace('_alt', "'")
        material_thickness[material] += thickness
        total_thickness += thickness

    stats['Total structure thickness'] = total_thickness
    for material, thickness in material_thickness.items():
        stats[f'Total {material} thickness'] = thickness

    return stats


def export_to_excel(wavelengths, reflectivity, structure, filename, smoothed_reflectivity=None):
    """
    Export reflectivity data with comprehensive analysis

    Args:
        wavelengths: Array of wavelength values
        reflectivity: Array of reflectivity values
        structure: List of tuples (layer_type, thickness)
        filename: Output Excel file path
        smoothed_reflectivity: Optional array of smoothed reflectivity values
    """
    # Create main data dictionary
    data_dict = {
        'Wavelength (Å)': wavelengths,
        'Reflectivity': reflectivity
    }

    # Add smoothed data if provided
    if smoothed_reflectivity is not None:
        data_dict['Smoothed Reflectivity'] = smoothed_reflectivity

    reflectivity_df = pd.DataFrame(data_dict)

    # Calculate all additional data
    thickness_stats = calculate_thickness_stats(structure)
    peaks = find_peaks_analysis(wavelengths, reflectivity)
    integral = integrate_reflectivity(wavelengths, reflectivity)

    # Create other DataFrames
    thickness_df = pd.DataFrame.from_dict(thickness_stats, orient='index',
                                          columns=['Thickness (Å)'])

    peaks_df = pd.DataFrame(peaks)
    if not peaks_df.empty:
        peaks_df = peaks_df[['Wavelength', 'Intensity', 'Width']]
        peaks_df.columns = ['Wavelength (Å)', 'Intensity', 'FWHM (Å)']

    integral_df = pd.DataFrame({
        'Parameter': ['Integral'],
        'Value': [integral]
    })

    # Create Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Save reflectivity data
        reflectivity_df.to_excel(writer, sheet_name='Reflectivity', index=False)

        # Save thickness statistics
        thickness_df.to_excel(writer, sheet_name='Thickness Info')

        # Save peaks information
        if not peaks_df.empty:
            peaks_df.to_excel(writer, sheet_name='Peaks Analysis', index=False)

        # Save integral value
        integral_df.to_excel(writer, sheet_name='Integral', index=False)

        # Adjust column widths
        for sheet in writer.sheets:
            worksheet = writer.sheets[sheet]
            for column in worksheet.columns:
                max_length = max(len(str(cell.value)) for cell in column)
                worksheet.column_dimensions[column[0].column_letter].width = max_length + 2

    return integral, peaks
