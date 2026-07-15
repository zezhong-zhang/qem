
# from shapely.affinity import scale

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from qem.viz.color import get_unique_colors


class CrystalAnalyzerPlotMixin:
    """Mixin providing plotting methods for :class:`CrystalAnalyzer`."""

    def plot(self):
        vmin = np.percentile(self.image, 5)
        vmax = np.percentile(self.image, 95)
        plt.imshow(self.image, cmap="gray", vmin=vmin, vmax=vmax)
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.scatter(
                self.peak_positions[mask, 0],
                self.peak_positions[mask, 1],
                label=element,
                color=next(color_iterator),
            )
        plt.xlim(0, self.image.shape[1])
        plt.ylim(0, self.image.shape[0])
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    def plot_unitcell(self, mode: str = "affine"):
        if mode == "perfect":
            unitcell_transformed = self.unit_cell_transformed["perfect"].copy()
            origin, a, b = self.origin, self.a_vector["perfect"], self.b_vector["perfect"]
            unitcell_transformed.positions[:, :2] += origin * self.dx
        else:
            unitcell_transformed = self.unit_cell_transformed["affine"].copy()
            origin, a, b = self.origin, self.a_vector["affine"], self.b_vector["affine"]
            unitcell_transformed.positions[:, :2] += origin * self.dx

        plt.subplots()
        plt.imshow(self.image, cmap="gray")
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask_element = self.atom_types == atom_type
            element = self.elements[atom_type]
            current_color = np.array(next(color_iterator)).reshape(1, -1)
            plt.scatter(
                self.peak_positions[:, 0][mask_element],
                self.peak_positions[:, 1][mask_element],
                label=element,
                c=current_color,
            )
        for element in self.get_unitcell_elements():
            current_color = np.array(next(color_iterator)).reshape(1, -1)
            mask_unitcell_element = self.is_element_in_unit_cell(
                self.unit_cell, element
            )
            plt.scatter(
                unitcell_transformed.positions[:, 0][mask_unitcell_element] / self.dx,
                unitcell_transformed.positions[:, 1][mask_unitcell_element] / self.dx,
                edgecolors="k",
                c=current_color,
                alpha=0.8,
                label=element + " unitcell",
            )
        plt.tight_layout()
        plt.legend()
        plt.setp(plt.gca(), aspect="equal", adjustable="box")
        plt.gca().add_artist(self.scalebar)
        # plt.gca().invert_yaxis()

        # plot the a and b vectors
        plt.arrow(
            origin[0],
            origin[1],
            a[0],
            a[1],
            color="k",
            head_width=5,
            head_length=5,
        )
        plt.arrow(
            origin[0],
            origin[1],
            b[0],
            b[1],
            color="k",
            head_width=5,
            head_length=5,
        )
        # label the a and b vectors
        plt.text(
            origin[0] + a[0],
            origin[1] + a[1],
            "a",
            fontsize=20,
        )
        plt.text(
            origin[0] + b[0],
            origin[1] + b[1],
            "b",
            fontsize=20,
        )

    def plot_displacement(
        self, mode: str = "local", cut_off: float = 5.0, units: str = "A"
    ):
        if mode == "local":
            displacement = self.atomic_columns.get_local_displacement(cut_off, units)
        else:
            displacement = self.atomic_columns.get_column_displacement(units)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(
            self.atomic_columns.x,
            self.atomic_columns.y,
            c=np.linalg.norm(displacement, axis=1),
            cmap="plasma",
        )
        cbar = plt.colorbar()
        cbar.set_label(f"Displacement ({units})")
        plt.quiver(
            self.atomic_columns.x,
            self.atomic_columns.y,
            displacement[:, 0],
            displacement[:, 1],
            scale=1,
            scale_units="xy",
        )
        plt.gca().add_artist(self.scalebar)
        plt.axis("off")

    def plot_strain(self, cut_off: float = 5.0, save: bool = False):
        epsilon_xx, epsilon_yy, epsilon_xy, omega_xy = self.get_strain(cut_off)
        plt.subplots(2, 2, constrained_layout=True)
        plt.subplot(2, 2, 1)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(
            self.atomic_columns.x, self.atomic_columns.y, c=epsilon_xx, cmap="coolwarm"
        )
        plt.axis("off")
        plt.gca().add_artist(self.scalebar)
        plt.colorbar()
        # bounds = np.abs(epsilon_xx).max()
        # get the 95 percentile of the strain
        bounds = np.percentile(np.abs(epsilon_xx), 95)
        plt.clim(-bounds, bounds)
        plt.title(r"$\epsilon_{xx}$")
        # plt.tight_layout()
        plt.subplot(2, 2, 2)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(
            self.atomic_columns.x, self.atomic_columns.y, c=epsilon_yy, cmap="coolwarm"
        )
        plt.colorbar()
        # bounds = np.abs(epsilon_yy).max()
        bounds = np.percentile(np.abs(epsilon_yy), 95)
        plt.clim(-bounds, bounds)
        plt.axis("off")
        plt.title(r"$\epsilon_{yy}$")
        # plt.tight_layout()
        plt.subplot(2, 2, 3)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(
            self.atomic_columns.x, self.atomic_columns.y, c=epsilon_xy, cmap="coolwarm"
        )
        plt.colorbar()
        # bounds = np.abs(epsilon_xy).max()
        bounds = np.percentile(np.abs(epsilon_xy), 95)
        plt.clim(-bounds, bounds)
        plt.axis("off")
        plt.title(r"$\epsilon_{xy}$")
        # plt.tight_layout()
        plt.subplot(2, 2, 4)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(
            self.atomic_columns.x, self.atomic_columns.y, c=omega_xy, cmap="coolwarm"
        )
        plt.colorbar()
        # bounds = np.abs(omega_xy).max()
        bounds = np.percentile(np.abs(omega_xy), 95)
        plt.clim(-bounds, bounds)
        plt.axis("off")
        plt.title(r"$\omega_{xy}$")
        # plt.tight_layout()
        if save:
            plt.savefig("strain_map.png", dpi=300)
            plt.savefig("strain_map.svg")

    def plot_polarization(self, a_element: str, b_element: str, cutoff_radius: float = 5.0, save: bool = False, exclude_border: bool = False, border_pixel: int = 10, vector_scale: float = 10.0):
        """Plot the polarization vectors and magnitudes.

        Args:
            a_element (str): Element for A atoms (e.g., 'Sr')
            b_element (str): Element for B atoms (e.g., 'Ti')
            cutoff_radius (float, optional): Radius to search for surrounding A atoms. Defaults to 5.0, unit: Å.
            save (bool, optional): Whether to save the plot. Defaults to False.
        """
        # Calculate polarization
        pol_data = self.measure_polarization(a_element, b_element, cutoff_radius)

        if exclude_border:
            border_mask = (pol_data['positions'][:, 0] < border_pixel) | (pol_data['positions'][:, 0] > self.image.shape[1] - border_pixel) | (pol_data['positions'][:, 1] < border_pixel) | (pol_data['positions'][:, 1] > self.image.shape[0] - border_pixel) # mask the border within border_pixel
            pol_data['positions'] = pol_data['positions'][~border_mask]
            pol_data['polarization'] = pol_data['polarization'][~border_mask]
            pol_data['magnitude'] = pol_data['magnitude'][~border_mask]


        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Vector field
        ax1.imshow(self.image, cmap='gray')
        # Plot A atoms
        # a_mask = self.atom_types == self.elements.index(a_element)
        # ax1.scatter(self.peak_positions[a_mask, 0],
        #            self.peak_positions[a_mask, 1],
        #            c='blue', alpha=0.5, label='A sites')

        # Plot polarization vectors
        valid_mask = ~np.isnan(pol_data['magnitude'])
        ax1.quiver(pol_data['positions'][valid_mask, 0],
                  pol_data['positions'][valid_mask, 1],
                  pol_data['polarization'][valid_mask, 0] *vector_scale,
                  pol_data['polarization'][valid_mask, 1] *vector_scale,
                  scale=2, scale_units='xy',
                  color='red', label='B sites polarization')

        ax1.set_title('Polarization Vectors')
        ax1.legend()
        ax1.add_artist(self.scalebar)

        # Plot 2: Magnitude map
        ax2.imshow(self.image, cmap='gray')
        scatter = ax2.scatter(pol_data['positions'][:, 0],
                            pol_data['positions'][:, 1],
                            c=pol_data['magnitude'] * self.dx,
                            cmap='plasma',
                            label='B sites')
        plt.colorbar(scatter, ax=ax2, label='Polarization magnitude (Å)')
        ax2.set_title('Polarization Magnitude')
        ax2.legend()
        ax2.add_artist(self.scalebar)

        plt.tight_layout()

        if save:
            plt.savefig('polarization_map.png', dpi=300, bbox_inches='tight')
            plt.savefig('polarization_map.svg', bbox_inches='tight')

    def plot_oxygen_tilt(self, a_type: int, o_type: int, cutoff_radius: float = 5.0, save: bool = False):
        """Plot the oxygen tilt angles and directions.

        Args:
            a_type (int): Atom type label for A site atoms
            o_type (int): Atom type label for oxygen atoms
            cutoff_radius (float, optional): Radius to search for nearby atoms. Defaults to 5.0.
            save (bool, optional): Whether to save the plot. Defaults to False.
        """
        # Calculate tilt
        tilt_data = self.measure_oxygen_tilt(a_type, o_type, cutoff_radius)

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

        # Plot 1: O-O tilt lines colored by angle
        ax1.imshow(self.image, cmap='gray')
        a_mask = self.atom_types == a_type
        ax1.scatter(self.peak_positions[a_mask, 0],
                self.peak_positions[a_mask, 1],
                c='blue', alpha=0.5, label=f'{a_type} atoms')
        valid_mask = ~np.isnan(tilt_data['tilt_angles'])
        oo_pairs = tilt_data['oo_pairs'][valid_mask]
        angles = tilt_data['tilt_angles'][valid_mask]
        from matplotlib.collections import LineCollection
        if len(oo_pairs) > 0:
            lines = oo_pairs
            line_colors = angles
            lc = LineCollection(lines, cmap='viridis', array=line_colors, linewidths=2)
            ax1.add_collection(lc)
            plt.colorbar(lc, ax=ax1, label='Tilt angle (degrees)')
        ax1.set_title('Oxygen Tilt Lines')
        ax1.legend()
        ax1.add_artist(self.scalebar)

        # Plot 2: Tilt angle map (as before)
        ax2.imshow(self.image, cmap='gray')
        scatter = ax2.scatter(tilt_data['positions'][:, 0],
                            tilt_data['positions'][:, 1],
                            c=tilt_data['tilt_angles'],
                            cmap='viridis',
                            label=f'{o_type} atoms')
        plt.colorbar(scatter, ax=ax2, label='Tilt angle (degrees)')
        ax2.set_title('Oxygen Tilt Angles')
        ax2.legend()
        ax2.add_artist(self.scalebar)

        # Plot 3: Histogram of tilt angles
        ax3.hist(angles, bins=30, color='gray', edgecolor='black')
        ax3.set_xlabel('Tilt angle (degrees)')
        ax3.set_ylabel('Count')
        ax3.set_title('Tilt Angle Distribution')

        plt.tight_layout()
        if save:
            plt.savefig('oxygen_tilt_map.png', dpi=300, bbox_inches='tight')
            plt.savefig('oxygen_tilt_map.svg', bbox_inches='tight')

    def plot_lattice_parameter_unitcell(
        self, units='A', min_dist:float=0.1, show_lattice:bool=False,
        boundary_thresh:int=20, line_plot_direction:str=None,
        line_plot_averaging_window:float=None, line_plot_style:str='confidence_interval',
        save: bool = False,
    ):
        """
        Plot local lattice parameters using adaptive cell origins.
        The lattice parameter is defined as the distance between neighboring origins in a and b directions.
        """
        adaptive_cells = self.get_origin_offset("adaptive")
        origins = np.array(list(adaptive_cells.values()))  # shape (N, 2)

        # Get direction unit vectors
        a_vec = self.a_vector['perfect']
        b_vec = self.b_vector['perfect']
        a_hat = a_vec / np.linalg.norm(a_vec)
        b_hat = b_vec / np.linalg.norm(b_vec)

        lines_a = []
        values_a = []
        lines_b = []
        values_b = []

        mask = np.ones(len(origins), dtype=bool)
        mask[origins[:,0] < boundary_thresh] = False
        mask[origins[:,0] > self.image.shape[1] - boundary_thresh] = False
        mask[origins[:,1] < boundary_thresh] = False
        mask[origins[:,1] > self.image.shape[0] - boundary_thresh] = False
        origins = origins[mask]

        for origin in origins:
            # skip the boundary
            x, y = origin
            rel = origins - origin
            # Project onto a and b directions
            proj_a = rel @ a_hat
            proj_b = rel @ b_hat
            # Find the closest neighbor in +a direction (exclude self
            mask_a = (proj_a > min_dist) & (np.abs(proj_b) < np.linalg.norm(b_vec)/2) & (np.abs(proj_a) < np.linalg.norm(a_vec)*1.5)
            if np.any(mask_a):
                j = np.argmin(np.where(mask_a, proj_a, np.inf))
                lines_a.append([origin, origins[j]])
                values_a.append(np.linalg.norm(origins[j] - origin) * self.dx)
            # Find the closest neighbor in +b direction (exclude self)
            mask_b = (proj_b > min_dist) & (np.abs(proj_a) < np.linalg.norm(a_vec)/2) & (np.abs(proj_b) < np.linalg.norm(b_vec)*1.5)
            if np.any(mask_b):
                j = np.argmin(np.where(mask_b, proj_b, np.inf))
                lines_b.append([origin, origins[j]])
                values_b.append(np.linalg.norm(origins[j] - origin) * self.dx)

        if line_plot_direction:
            # Determine projection vector and axis name
            if line_plot_direction == 'a':
                proj_hat, axis_name = a_hat, 'a'
            elif line_plot_direction == 'b':
                proj_hat, axis_name = b_hat, 'b'
            else:
                raise ValueError("line_plot_direction must be 'a' or 'b'")

            # Setup plot
            fig, (ax_a, ax_b) = plt.subplots(2, 1, sharex=True)

            # Data to process: [ (data_lines, data_values, axis_to_plot_on, color, param_name), ... ]
            datasets = [
                (lines_a, values_a, ax_a, 'blue', 'a'),
                (lines_b, values_b, ax_b, 'green', 'b')
            ]

            for lines, values, ax, color, param_name in datasets:
                if not lines:
                    ax.text(0.5, 0.5, f'No data for parameter {param_name}', ha='center', va='center', transform=ax.transAxes)
                    continue

                # --- Moving Average Calculation ---
                line_midpoints = np.array([np.mean(line, axis=0) for line in lines])
                values_arr = np.array(values)
                projected_dist = (line_midpoints @ proj_hat) * self.dx

                # Default window size
                window = line_plot_averaging_window
                if window is None:
                    # Default to one unit cell dimension of the projection axis
                    if line_plot_direction == 'a':
                        window_pixels = np.linalg.norm(self.a_vector['perfect'])
                    else: # 'b'
                        window_pixels = np.linalg.norm(self.b_vector['perfect'])
                    window = window_pixels * self.dx

                sort_indices = np.argsort(projected_dist)
                sorted_dist = projected_dist[sort_indices]
                sorted_values = values_arr[sort_indices]

                # --- Clustering and Stats Calculation ---
                plot_cluster_dist_means = np.array([])
                plot_cluster_value_means = np.array([])
                plot_cluster_value_stds = np.array([])
                plot_cluster_value_cis = np.array([])

                if sorted_dist.size > 0:
                    clusters_data = []
                    current_cluster_dists = [sorted_dist[0]]
                    current_cluster_values = [sorted_values[0]]
                    cluster_start_dist = sorted_dist[0]

                    for i in range(1, len(sorted_dist)):
                        current_dist = sorted_dist[i]
                        current_val = sorted_values[i]

                        if current_dist - cluster_start_dist <= window:
                            current_cluster_dists.append(current_dist)
                            current_cluster_values.append(current_val)
                        else:
                            if current_cluster_values:
                                mean_d = np.mean(current_cluster_dists)
                                mean_v = np.mean(current_cluster_values)
                                std_v = np.std(current_cluster_values) if len(current_cluster_values) > 0 else 0
                                ci_v = 0
                                if len(current_cluster_values) > 1:
                                    sem = np.std(current_cluster_values, ddof=1) / np.sqrt(len(current_cluster_values))
                                    ci_v = 1.96 * sem
                                clusters_data.append((mean_d, mean_v, std_v, ci_v))

                            cluster_start_dist = current_dist
                            current_cluster_dists = [current_dist]
                            current_cluster_values = [current_val]

                    if current_cluster_values: # Finalize the last cluster
                        mean_d = np.mean(current_cluster_dists)
                        mean_v = np.mean(current_cluster_values)
                        std_v = np.std(current_cluster_values) if len(current_cluster_values) > 0 else 0
                        ci_v = 0
                        if len(current_cluster_values) > 1:
                            sem = np.std(current_cluster_values, ddof=1) / np.sqrt(len(current_cluster_values))
                            ci_v = 1.96 * sem
                        clusters_data.append((mean_d, mean_v, std_v, ci_v))

                    if clusters_data:
                        plot_cluster_dist_means = np.array([c[0] for c in clusters_data])
                        plot_cluster_value_means = np.array([c[1] for c in clusters_data])
                        plot_cluster_value_stds = np.array([c[2] for c in clusters_data])
                        plot_cluster_value_cis = np.array([c[3] for c in clusters_data])

                # --- Plotting ---
                ax.scatter(sorted_dist, sorted_values, alpha=0.2, color='gray', label='Raw Data', s=10)

                if plot_cluster_dist_means.size > 0:
                    if line_plot_style == 'confidence_interval':
                        ax.errorbar(plot_cluster_dist_means, plot_cluster_value_means,
                                    yerr=plot_cluster_value_cis, fmt='o', color=color,
                                    capsize=3, markersize=5, elinewidth=1.5,
                                    label=f'Parameter {param_name} (cluster mean & 95% CI)')
                    elif line_plot_style == 'error_bars':
                        ax.errorbar(plot_cluster_dist_means, plot_cluster_value_means,
                                    yerr=plot_cluster_value_stds, fmt='o', color=color,
                                    capsize=3, markersize=5, elinewidth=1.5,
                                    label=f'Parameter {param_name} (cluster mean ± std)')
                    else:
                        raise ValueError("line_plot_style must be 'confidence_interval' or 'error_bars'")
                else:
                    if sorted_dist.size > 0: # Raw data might exist even if no clusters formed (e.g. single point)
                         ax.text(0.5, 0.4, f'Not enough data to form clusters for {param_name}', ha='center', va='center', transform=ax.transAxes, fontsize=9)
                    # else: (handled by the initial check for `if not lines:`) -> ax.text(0.5, 0.5, f'No data for parameter {param_name}'...)

                ax.set_ylabel(f'Lattice parameter {param_name} ({units})')
                ax.legend()
                ax.grid(True)

            fig.suptitle(f'Lattice Parameter Profile along {axis_name}-direction', fontsize=16)
            ax_b.set_xlabel(f'Distance along {axis_name}-axis ({units})')
            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
            plt.show()

        # Atom coloring by type
        x = self.atomic_columns.x
        y = self.atomic_columns.y
        atom_types = self.atomic_columns.atom_types
        unique_types = np.unique(atom_types)
        color_map = {atype: color for atype, color in zip(unique_types, get_unique_colors(), strict=False)}
        atom_colors = [color_map[atype] for atype in atom_types]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap="gray")
        lc_a = LineCollection(lines_a, array=np.array(values_a), cmap='Blues', linewidths=2)
        plt.gca().add_collection(lc_a)
        if show_lattice:
            plt.scatter(x, y, c=atom_colors, s=15, edgecolor='k', linewidth=0.5)
        cbar_a = plt.colorbar(lc_a)
        cbar_a.set_label(f'Lattice parameter a ({units})')
        plt.axis('off')
        if hasattr(self, 'scalebar'):
            plt.gca().add_artist(self.scalebar)
        plt.title('Lattice parameter a map')
        # set the plot to the view of the image
        #plt.xlim(0, self.image.shape[1])
        #plt.ylim(0, self.image.shape[0])

        plt.subplot(1, 2, 2)
        plt.imshow(self.image, cmap="gray")
        lc_b = LineCollection(lines_b, array=np.array(values_b), cmap='Greens', linewidths=2)
        plt.gca().add_collection(lc_b)
        if show_lattice:
            plt.scatter(x, y, c=atom_colors, s=15, edgecolor='k', linewidth=0.5)
        cbar_b = plt.colorbar(lc_b)
        cbar_b.set_label(f'Lattice parameter b ({units})')
        plt.axis('off')
        if hasattr(self, 'scalebar'):
            plt.gca().add_artist(self.scalebar)
        plt.title('Lattice parameter b map')
        # set the plot to the view of the image
        #plt.xlim(0, self.image.shape[1])
        #plt.ylim(0, self.image.shape[0])
        plt.show()
        if save:
            plt.savefig("lattice_parameter_map.svg")
            plt.savefig("lattice_parameter_map.png", dpi=300)


    def plot_lattice_parameter_nearest(self, units='A', show_lattice:bool=False, angle_thresh:float=0.95, dist_min_a:float=1, dist_min_b:float=1, dist_max_a:float=None, dist_max_b:float=None, boundary_thresh:int=5):
        """
        Plot local lattice parameters using all nearest neighbors in the a and b directions
        within an angular and distance cutoff.

        Args:
            units (str, optional): Unit of the lattice parameter. Defaults to 'A'.
            show_lattice (bool, optional): Whether to show the lattice. Defaults to False.
            angle_thresh (float, optional): Angular cutoff. Defaults to 0.95.
            dist_min_a (float, optional): Minimum distance in a direction in A. Defaults to 1.
            dist_min_b (float, optional): Minimum distance in b direction in A. Defaults to 1.
            dist_max_a (float, optional): Maximum distance in a direction in A. Defaults to 3.
            dist_max_b (float, optional): Maximum distance in b direction in A. Defaults to 3.
            boundary_thresh (int, optional): Boundary threshold in pixels. Defaults to 20.
        """
        a_vec = self.a_vector['perfect']
        b_vec = self.b_vector['perfect']
        a_hat = a_vec / np.linalg.norm(a_vec)
        b_hat = b_vec / np.linalg.norm(b_vec)

        if dist_max_a is None:
            # Guess a reasonable maximum distance (e.g., 1.5 × norm of a_vec)
            dist_max_a  = 1.3 * np.linalg.norm(a_vec) * self.dx
        if dist_max_b is None:
            # Guess a reasonable maximum distance (e.g., 1.5 × norm of b_vec)
            dist_max_b  = 1.3 * np.linalg.norm(b_vec) * self.dx

        lines_a, values_a = [], []
        lines_b, values_b = [], []

        # skip boundary
        mask = np.ones(len(self.peak_positions), dtype=bool)
        mask[self.peak_positions[:,0] < boundary_thresh] = False
        mask[self.peak_positions[:,0] > self.image.shape[1] - boundary_thresh] = False
        mask[self.peak_positions[:,1] < boundary_thresh] = False
        mask[self.peak_positions[:,1] > self.image.shape[0] - boundary_thresh] = False

        peak_masked = self.peak_positions[mask]


        for i in range(len(peak_masked)):
            x,y = peak_masked[i]
            rel = peak_masked - np.array([x,y])
            dists = np.linalg.norm(rel, axis=1) * self.dx
            # Exclude self
            valid_a  = (dists > dist_min_a) & (dists < dist_max_a)
            valid_b = (dists > dist_min_b) & (dists < dist_max_b)

            # For a direction
            cos_a = np.abs((rel @ a_hat) / (np.linalg.norm(rel, axis=1) + 1e-12))
            mask_a = valid_a & (cos_a > angle_thresh)
            for j in np.where(mask_a)[0]:
                lines_a.append([peak_masked[i], peak_masked[j]])
                values_a.append(dists[j])

            # For b direction
            cos_b = np.abs((rel @ b_hat) / (np.linalg.norm(rel, axis=1) + 1e-12))
            mask_b = valid_b & (cos_b > angle_thresh)
            for j in np.where(mask_b)[0]:
                lines_b.append([peak_masked[i], peak_masked[j]])
                values_b.append(dists[j])

        # Atom coloring by type
        x = self.atomic_columns.x
        y = self.atomic_columns.y
        atom_types = self.atomic_columns.atom_types
        unique_types = np.unique(atom_types)
        color_map = {atype: color for atype, color in zip(unique_types, get_unique_colors(), strict=False)}
        atom_colors = [color_map[atype] for atype in atom_types]

        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap="gray")
        lc_a = LineCollection(lines_a, array=np.array(values_a), cmap='Blues', linewidths=2)
        plt.gca().add_collection(lc_a)
        if show_lattice:
            plt.scatter(x, y, c=atom_colors, s=15, edgecolor='k', linewidth=0.5)
        cbar_a = plt.colorbar(lc_a)
        cbar_a.set_label(f'Lattice parameter a (nearest, {units})')
        plt.axis('off')
        if hasattr(self, 'scalebar'):
            plt.gca().add_artist(self.scalebar)
        plt.title('Lattice parameter a (nearest) map')
        #plt.xlim(0, self.image.shape[1])
        #plt.ylim(0, self.image.shape[0])

        plt.subplot(1, 2, 2)
        plt.imshow(self.image, cmap="gray")
        lc_b = LineCollection(lines_b, array=np.array(values_b), cmap='Greens', linewidths=2)
        plt.gca().add_collection(lc_b)
        if show_lattice:
            plt.scatter(x, y, c=atom_colors, s=15, edgecolor='k', linewidth=0.5)
        cbar_b = plt.colorbar(lc_b)
        cbar_b.set_label(f'Lattice parameter b (nearest, {units})')
        plt.axis('off')
        if hasattr(self, 'scalebar'):
            plt.gca().add_artist(self.scalebar)
        plt.title('Lattice parameter b (nearest) map')
        #plt.xlim(0, self.image.shape[1])
        #plt.ylim(0, self.image.shape[0])
        plt.show()
