import numpy as np
import matplotlib.pyplot as plt


def carbon_scattering_factor(q):
    """
    Compute the electron scattering factor f_e(q) for carbon,
    using a 3-Lorentz + 3-Gaussian form, i.e. Kirkland's Eq. (C.15):
    
        f_e(q) = sum_{i=1..3} [ a_i / (q^2 + b_i ) ]
               + sum_{i=1..3} [ c_i * exp( - d_i * q^2 ) ]
    
    where q is in units of A^-1.
    
    Parameters
    ----------
    q : float or ndarray
        Magnitude of scattering vector in A^-1.
    
    Returns
    -------
    f_vals : float or ndarray
        Electron scattering factor for each q.
    """
    
    # --- Carbon parameter values (a_i, b_i, c_i, d_i) ---
    a = [2.53148417e-001, 1.66953010e-001, 1.60250013e-001]     # (units: Å^-1)
    b = [2.08761775e-001, 5.69804854e+000, 2.08774166e-001]     # (units: Å^-2)
    c = [1.41633354e-001, 3.60244465e-001, 8.66192838e-004]     # (units: Å^)
    d = [1.34112859e+000, 3.81665326e+000, 4.12204523e-002]     # (units: Å^2)
    # ------------------------------------------------------------
    
    # Convert q to an array
    q_arr = np.atleast_1d(q)
    
    # Build zero array matching q shape
    f_vals = np.zeros_like(q_arr, dtype=float)
    
    # Lorentz terms: sum_{i=1..3} a_i / (q^2 + b_i)
    lorentz_sum = np.zeros_like(q_arr, dtype=float)
    for i in range(3):
        lorentz_sum += a[i] / (q_arr**2 + b[i])
    
    # Gaussian terms: sum_{i=1..3} c_i * exp(-d_i * q^2)
    gauss_sum = np.zeros_like(q_arr, dtype=float)
    for i in range(3):
        gauss_sum += c[i] * np.exp(- d[i] * (q_arr**2))
    
    f_vals = lorentz_sum + gauss_sum
    
    # If a single float is passed, return float
    if np.isscalar(q):
        return f_vals[0]
    else:
        return f_vals



def compute_slice_potential(atom_count, carbon_scatter_func, 
                            qx_vals, qy_vals):
    """
    Given a 2D array of 'atom_count' in real space (Poisson random draws),
    multiply by the carbon scattering factor in reciprocal space 
    and return the resulting 'slice potential' in real space.
    
    Parameters
    ----------
    atom_count : 2D float array
        Number of atoms per pixel in this slice 
        (Poisson-distributed).
    carbon_scatter_func : callable
        A function that returns f_e(q) for carbon, 
        e.g. carbon_scattering_factor(q).
    qx_vals, qy_vals : 1D arrays
        Frequency axes in cycles/Angstrom. 
        Must be fftshifted so that zero freq is in center.
    
    Returns
    -------
    slice_potential : 2D float array
        The resulting real-space potential from this slice.
    """
    nx, ny = atom_count.shape
    
    # Forward FFT of the random distribution
    F_atoms = np.fft.fft2(atom_count)
    F_atoms = np.fft.fftshift(F_atoms)
    
    # Build mesh grid of q
    QX, QY = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    q_mag = np.sqrt(QX**2 + QY**2)  # A^-1
    
    # Evaluate carbon scattering factor at each q
    f_e = carbon_scatter_func(q_mag)
    
    # Multiply in reciprocal space
    F_atoms *= f_e
    
    # Inverse FFT
    F_atoms = np.fft.ifftshift(F_atoms)
    slice_potential = np.fft.ifft2(F_atoms).real
    
    return slice_potential



def simulate_one_slice(nx, ny, slice_thickness_nm, mean_density, 
                       qx_vals, qy_vals, carbon_scatter_func):
    """
    Create Poisson random distribution of carbon atoms for a single slice 
    and compute the slice potential.
    
    Parameters
    ----------
    nx, ny : int
        Image size in pixels.
    slice_thickness_nm : float
        Thickness (nm) of this slice.
    mean_density : float
        Average number of atoms per (pixel * nm).
        i.e. atoms per pixel per nm thickness.
    qx_vals, qy_vals : 1D arrays
        Frequency axes in cycles/Angstrom (fftshifted).
    carbon_scatter_func : callable
        Function that returns the carbon scattering factor, 
        e.g. carbon_scattering_factor.
    
    Returns
    -------
    slice_potential : 2D float array
        The resulting real-space potential from this slice.
    """
    # Poisson parameter = mean_density * slice_thickness
    lam_slice = mean_density * slice_thickness_nm
    
    # Draw random Poisson
    atom_count = np.random.poisson(lam_slice, size=(nx, ny))
    
    # Compute potential for this slice
    slice_pot = compute_slice_potential(atom_count, carbon_scatter_func, 
                                        qx_vals, qy_vals)
    return slice_pot


def simulate_amorphous_carbon_film(nx, ny, delta_x_A, 
                                   total_thickness_nm, slice_thickness_nm, 
                                   mean_density,
                                   carbon_scatter_func):
    """
    Simulate an amorphous carbon film by stacking multiple slices 
    of random carbon distributions in real space, applying the 
    scattering factor in reciprocal space, and summing the potentials.
    
    Parameters
    ----------
    nx, ny : int
        2D grid size in pixels.
    delta_x_A : float
        Pixel size in Angstroms (A/pixel).
    total_thickness_nm : float
        Total thickness of the carbon film in nm.
    slice_thickness_nm : float
        Thickness of each slice in nm.
    mean_density : float
        Mean # of atoms per (pixel * nm). 
        For example, if your pixel covers 0.04 nm^2 in real space, 
        and your 3D density is ~0.6 atoms/nm^3, you'd do:
          mean_density = 0.6 * 0.04 = 0.024  atoms/pixel/nm
    carbon_scatter_func : callable
        Function that returns the carbon scattering factor f_e(q).
    
    Returns
    -------
    film_potential : 2D float array
        The final 2D projected potential of the amorphous carbon film.
    """
    # Number of slices
    n_slices = int(np.rint(total_thickness_nm / slice_thickness_nm))
    
    # Create the reciprocal-space frequency axes in cycles/Angstrom
    # and shift so zero freq is in center
    qx_vals = np.fft.fftfreq(nx, d=delta_x_A)
    qx_vals = np.fft.fftshift(qx_vals)
    qy_vals = np.fft.fftfreq(ny, d=delta_x_A)
    qy_vals = np.fft.fftshift(qy_vals)
    
    film_potential = np.zeros((nx, ny), dtype=float)
    
    for i in range(n_slices):
        slice_pot = simulate_one_slice(nx, ny, slice_thickness_nm, 
                                       mean_density, 
                                       qx_vals, qy_vals,
                                       carbon_scatter_func)
        film_potential += slice_pot
    
    return film_potential


# ------------------ MAIN USAGE -------------------
if __name__ == "__main__":
    # Parameters
    nx, ny = 256, 256          # image size
    delta_x_A = 0.2            # 0.2 A/pixel 
    total_thickness_nm = 10.0  # 10 nm film
    slice_thickness_nm = 1.0   # each slice is 1 nm
    # Suppose we want ~1.0 atoms/nm^3 in the 3D volume of the film 
    # => in 2D per slice: 
    #   pixel area = (0.2 A)^2 = 0.04 A^2 = 0.04e-2 nm^2 = 4e-4 nm^2
    #   so each pixel in 1nm slice covers 4e-4 nm^3
    #   1.0 atoms/nm^3 * 4e-4 nm^3 = 4e-4 atoms/pixel
    # => mean_density = 0.0004 
    mean_density = 0.0004
    
    # Actually the typical density for amorphous carbon 
    # might be higher, but let's keep it small for a test. 
    # (Tune as needed!)
    
    # Now run the full simulation
    film_pot = simulate_amorphous_carbon_film(
        nx, ny,
        delta_x_A,
        total_thickness_nm,
        slice_thickness_nm,
        mean_density,
        carbon_scattering_factor
    )
    
    # film_pot now holds the random carbon film's projected potential
    print("Amorphous carbon film potential shape:", film_pot.shape)
    print("Film potential - min:", film_pot.min(), 
          "max:", film_pot.max(), 
          "mean:", film_pot.mean())
    
    # Visualize the film potential:
    plt.figure(dpi=100)
    plt.imshow(film_pot, cmap='gray')
    plt.title("Simulated Amorphous Carbon Potential")
    plt.colorbar(label='Potential (arb. units)')
    plt.show()
